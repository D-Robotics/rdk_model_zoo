"""Canonical stage spellings, thin entry and explicit construction wiring.

The readable-runtime design requires the stage classes to expose the
canonical ``preprocess``/``infer``/``postprocess`` spellings (legacy names
stay compatible, error attribution keeps its established wording), the
entry to parse through a local ``cli`` module, and ``main`` to construct
the frontend/runtime bundle visibly and drive ``pipeline.predict`` per
utterance itself.
"""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from samples.speech.paraformer.tests import test_pipeline as fixtures


class CanonicalStageTests(unittest.TestCase):
    def fixture(self):
        fixture = fixtures.PipelineTests()
        fixture.setUp()
        return fixture, fixture.make_pipeline()

    def test_canonical_stage_spellings_match_legacy(self):
        fixture, pipeline = self.fixture()
        enc, pred, dec = (
            pipeline.encoder_stage,
            pipeline.predictor_stage,
            pipeline.decoder_stage,
        )

        prepared = enc.preprocess(fixture.features)
        legacy = enc.pre_process(fixture.features)
        np.testing.assert_array_equal(
            prepared.tensors["speech"], legacy.tensors["speech"])

        context = enc.postprocess(enc.infer(prepared.tensors))
        np.testing.assert_array_equal(
            context, enc.post_process(enc.forward(prepared.tensors)))

        prepared_pred = pred.preprocess(context)
        alphas, hidden = pred.postprocess(pred.infer(prepared_pred.tensors))
        np.testing.assert_array_equal(
            alphas, pred.post_process(pred.forward(prepared_pred.tensors))[0])

        from samples.speech.paraformer.runtime.python.cif import cif_numpy

        acoustic, count = cif_numpy(alphas, hidden, real_T=4)
        prepared_dec = dec.preprocess(context, count, acoustic)
        decoded = dec.postprocess(dec.infer(prepared_dec.tensors), prepared_dec.context)
        self.assertIsInstance(decoded.text, str)

    def test_predict_routes_through_canonical_stage_spellings(self):
        fixture, pipeline = self.fixture()
        calls = {"pre": 0, "inf": 0, "post": 0}
        enc = pipeline.encoder_stage
        original = (enc.preprocess, enc.infer, enc.postprocess)

        def counting_preprocess(features):
            calls["pre"] += 1
            return original[0](features)

        def counting_infer(tensors):
            calls["inf"] += 1
            return original[1](tensors)

        def counting_postprocess(outputs):
            calls["post"] += 1
            return original[2](outputs)

        enc.preprocess = counting_preprocess
        enc.infer = counting_infer
        enc.postprocess = counting_postprocess
        result = pipeline.predict(fixture.features, 4)
        self.assertEqual(result.token_count, 4)
        self.assertEqual(calls, {"pre": 1, "inf": 1, "post": 1})

    def test_infer_failure_keeps_established_error_wording(self):
        fixture, pipeline = self.fixture()
        original = RuntimeError("transport failed")
        pipeline.encoder = Mock(side_effect=original)
        with self.assertRaises(ValueError) as caught:
            pipeline.predict(fixture.features, 4)
        error = caught.exception
        self.assertEqual(getattr(error, "stage", None), "encoder")
        self.assertEqual(getattr(error, "operation", None), "forward")
        self.assertIs(error.__cause__, original)


class ThinEntryTests(unittest.TestCase):
    def test_main_reexports_local_cli_parser(self):
        from samples.speech.paraformer.runtime.python import cli, main

        self.assertIs(main.build_parser, cli.build_parser)
        for name in ("normalize_args", "resolve_selections_for", "print_resolution"):
            self.assertTrue(hasattr(cli, name), name)

    def test_main_constructs_bundle_and_runs_pipeline_predict(self):
        from samples.speech.paraformer.runtime.python import main

        name = "samples.speech.paraformer.runtime.python.main"
        self.assertIsNotNone(importlib.util.find_spec(name))
        predict = Mock(return_value=SimpleNamespace(
            text="中", token_ids=(3,), token_count=1,
            timings_ms={"encoder": 1.0, "predictor": 1.0, "cif": 1.0, "decoder": 1.0},
            decoder_executed=True))
        bundle = SimpleNamespace(
            runners=(SimpleNamespace(binding=SimpleNamespace(metadata={})),),
            predict=predict,
            set_scheduling_params=lambda **kwargs: None)
        frontend = SimpleNamespace(
            pre_process=lambda audio, rate: SimpleNamespace(
                tensor=np.zeros((1, 400, 560), np.float32), valid_frames=7,
                original_frames=7, truncated=False, sample_count=len(audio)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"fixture audio read by injected loader")
            out = root / "run"
            tokens = (
                Path(__file__).resolve().parents[4]
                / "samples/speech/paraformer/tests/fixtures/published-tokens.json"
            )
            args = ["--target", "s100", "--audio-file", str(audio),
                    "--output-dir", str(out), "--tokens-path", str(tokens)]
            from samples.speech.paraformer.runtime.python.model_binding import (
                resolve_selections,
            )

            for selected in resolve_selections("s100"):
                path = root / f"{selected.stage}.hbm"
                path.write_bytes(b"host-only SDK double; not a compiled HBM")
                args += [
                    f"--{selected.stage}-model-path", str(path),
                    f"--{selected.stage}-asset-id", selected.asset.reference,
                ]
            stream = io.StringIO()
            with patch("utils.py_utils.platforms.require_execution_target",
                       return_value=None), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                return_value=frontend), patch(
                "samples.speech.paraformer.runtime.python.pipeline.ParaformerPipeline.from_models",
                return_value=bundle) as load, patch(
                "samples.speech.paraformer.runtime.python.input_io.read_audio",
                return_value=(np.zeros(16000, np.float32), 16000)), contextlib.redirect_stdout(stream):
                self.assertEqual(main.main(args), 0)
            load.assert_called_once()
            predict.assert_called_once()
            self.assertEqual(predict.call_args.args[0].shape, (1, 400, 560))
            self.assertEqual(predict.call_args.args[1], 7)
            report = json.loads((out / "result.json").read_text())
            self.assertEqual(report["utterances"][0]["text"], "中")
            self.assertTrue(report["inference_executed"])


if __name__ == "__main__":
    unittest.main()
