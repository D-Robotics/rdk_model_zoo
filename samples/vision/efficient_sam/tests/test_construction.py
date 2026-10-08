# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Construct and execute the public SAM pipeline through SDK boundaries."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from samples.vision.efficient_sam.runtime.python import main
from samples.vision.efficient_sam.runtime.python.pipeline import EfficientSAMPipeline
from utils.py_utils.runtime_meta import MetadataMismatchError
from utils.py_utils.sam_binding import resolve_selection
from utils.py_utils.tests.test_sam_binding import FakeRuntime, metadata


class ConstructionTests(unittest.TestCase):
    def make_sdk_pair(self, target="s100", *, bad_decoder=False):
        selection = resolve_selection("efficient_sam", target)
        encoder = FakeRuntime(metadata("efficient_sam", "encoder", target))
        decoder = FakeRuntime(metadata("efficient_sam", "decoder", target))
        # A known winning candidate with zero logits pins the sample threshold.
        decoder.outputs["low_res_masks"][:] = 0.0
        decoder.outputs["iou_predictions"][:] = np.array([0.1, 0.8, 0.2]).reshape(
            decoder.outputs["iou_predictions"].shape)
        if bad_decoder:
            decoder.input_shapes["image_embeddings"] = (1, 3)
        events = []
        runtimes = {str(selection.encoder_model_path): encoder,
                    str(selection.decoder_model_path): decoder}
        def factory(path):
            events.append(Path(path).name)
            return runtimes[path]
        return selection, encoder, decoder, factory, events

    def construct(self, selection):
        self.assertTrue(callable(getattr(EfficientSAMPipeline, "from_models", None)),
                        "The pipeline must own construction of its SDK models")
        return EfficientSAMPipeline.from_models(selection)

    def test_named_construction_predicts_once_and_schedules_both_models(self):
        for target in ("x5", "s100", "s100p", "s600"):
            with self.subTest(target=target):
                selection, encoder, decoder, factory, events = self.make_sdk_pair(target)
                with patch("utils.py_utils.platforms.require_execution_target") as gate, patch(
                    "utils.py_utils.sam_runner._default_runtime_factory", return_value=factory):
                    pipeline = self.construct(selection)
                    pipeline.set_scheduling_params(priority=7, bpu_cores=None if target == "x5" else [0])
                    image = np.arange(17 * 31 * 3, dtype=np.uint8).reshape(17, 31, 3)
                    before = image.copy()
                    result = pipeline.predict(image)
                gate.assert_called_once_with(target)
                self.assertEqual(events, [selection.encoder_model_path.name,
                                          selection.decoder_model_path.name])
                self.assertEqual(len(encoder.calls), 1)
                self.assertEqual(len(decoder.calls), 1)
                inputs = encoder.calls[0] if target == "x5" else encoder.calls[0]["encoder"]
                tensor = inputs["batched_images"]
                self.assertEqual(tensor.shape, (1, 3, 512, 512))
                self.assertTrue(np.isfinite(tensor).all())
                self.assertGreater(np.count_nonzero(tensor), 100)
                decoder_inputs = decoder.calls[0] if target == "x5" else decoder.calls[0]["decoder"]
                self.assertTrue(np.all(decoder_inputs["image_embeddings"] == 3.0))

                self.assertEqual(result["mask_index"], 1)
                self.assertAlmostEqual(result["iou"], 0.8, places=6)
                self.assertEqual(result["mask"].shape, (512, 512))
                self.assertEqual(bool(result["mask"].all()), True)
                np.testing.assert_array_equal(image, before)
                for stage, runtime in (("encoder", encoder), ("decoder", decoder)):
                    expected = {"priority": {stage: 7}}
                    if target != "x5":
                        expected["bpu_cores"] = {stage: [0]}
                    self.assertEqual(runtime.schedules, [expected])

    def test_wrong_board_fails_before_sdk_factory_or_inference(self):
        selection, encoder, decoder, factory, events = self.make_sdk_pair()
        with patch("utils.py_utils.platforms.require_execution_target",
                   side_effect=ValueError("Target mismatch")), patch(
            "utils.py_utils.sam_runner._default_runtime_factory", return_value=factory) as sdk:
            with self.assertRaisesRegex(ValueError, "Target mismatch"):
                self.construct(selection)
        sdk.assert_not_called()
        self.assertEqual(events, [])
        self.assertEqual(encoder.calls + decoder.calls, [])

    def test_invalid_pair_metadata_fails_before_scheduling_or_prediction(self):
        selection, encoder, decoder, factory, events = self.make_sdk_pair(bad_decoder=True)
        with patch("utils.py_utils.platforms.require_execution_target"), patch(
            "utils.py_utils.sam_runner._default_runtime_factory", return_value=factory):
            with self.assertRaisesRegex(MetadataMismatchError, "decoder"):
                self.construct(selection)
        self.assertEqual(len(events), 2)
        self.assertEqual(encoder.calls + decoder.calls, [])
        self.assertEqual(encoder.schedules + decoder.schedules, [])

    def test_cli_uses_the_pipeline_owned_construction(self):
        selection, encoder, decoder, factory, events = self.make_sdk_pair()
        self.assertTrue(callable(getattr(EfficientSAMPipeline, "from_models", None)),
                        "CLI construction must use the public model factory")
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.py_utils.platforms.require_execution_target"), patch(
            "utils.py_utils.sam_runner._default_runtime_factory", return_value=factory), patch.object(
            EfficientSAMPipeline, "from_models", wraps=EfficientSAMPipeline.from_models) as construction, patch.object(
            main, "read_bgr_image", return_value=np.full((19, 23, 3), 42, dtype=np.uint8)), redirect_stdout(StringIO()):
            status = main.main(["--target", "s100", "--img-save-path", directory + "/overlay.jpg",
                                "--mask-save-path", directory + "/mask.png"])
            self.assertEqual(status, 0)
            construction.assert_called_once()
            self.assertEqual(construction.call_args.args[0].target, "s100")
            self.assertTrue(Path(directory, "mask.png").is_file())
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(len(decoder.calls), 1)
        self.assertEqual(encoder.schedules, [{"priority": {"encoder": 0}, "bpu_cores": {"encoder": [0]}}])
        self.assertEqual(decoder.schedules, [{"priority": {"decoder": 0}, "bpu_cores": {"decoder": [0]}}])


if __name__ == "__main__":
    unittest.main()
