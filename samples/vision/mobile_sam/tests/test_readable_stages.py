# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Canonical local stage API, box prompt handling and explicit orchestration.

The readable-runtime design requires the local pipeline to expose the
encoder and decoder stages with the canonical ``preprocess``/``infer``/
``postprocess`` spellings, keep the established point/box prompt API
(``predict(image, box=...)``), and show the real encode → decode
composition with per-stage error attribution — while the SAM math itself
stays in the shared stage implementations.
"""

from __future__ import annotations

import unittest

import numpy as np

from utils.py_utils.runtime_meta import RuntimeMetadata
from utils.py_utils.sam_binding import bind_model, resolve_selection
from utils.py_utils.sam_stages import StageError
from utils.py_utils.sam_tensor_io import DEFAULT_BOX


def make_binding(target: str = "s100"):
    selection = resolve_selection("mobile_sam", target)
    encoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "encoder",
        "input_names": ("normalized_images",),
        "input_shapes": {"normalized_images": (1, 3, 512, 512)},
        "input_dtypes": {"normalized_images": "float32"},
        "output_names": ("image_embeddings",),
        "output_shapes": {"image_embeddings": (1, 256, 32, 32)},
        "output_dtypes": {"image_embeddings": "float16"},
        "output_quants": {"image_embeddings": {"scale": 0.1, "zero_point": 3}},
    })
    box_shape = (1, 4, 1, 1) if target == "x5" else (1, 4)
    decoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "decoder",
        "input_names": ("image_embeddings", "boxes"),
        "input_shapes": {"image_embeddings": (1, 256, 32, 32), "boxes": box_shape},
        "input_dtypes": {"image_embeddings": "float32", "boxes": "float32"},
        "output_names": ("low_res_masks", "iou_predictions"),
        "output_shapes": {"low_res_masks": (1, 3, 128, 128), "iou_predictions": (1, 3)},
        "output_dtypes": {"low_res_masks": "float16", "iou_predictions": "float16"},
        "output_quants": {
            "low_res_masks": {"scale": 0.1, "zero_point": 3},
            "iou_predictions": {"scale": 0.1, "zero_point": 3},
        },
    })
    return bind_model(selection, encoder_meta, decoder_meta)


class FakeStageRunner:
    def __init__(self, outputs=None, *, fail=False):
        self.outputs = outputs or {}
        self.fail = fail
        self.calls = []

    def __call__(self, tensors):
        self.calls.append(tensors)
        if self.fail:
            raise RuntimeError("fixture failure")
        return self.outputs


def fixtures(*, encoder_fail=False, decoder_fail=False, target="s100"):
    from samples.vision.mobile_sam.runtime.python.pipeline import MobileSAMPipeline

    embedding = np.ones((1, 256, 32, 32), dtype=np.float16)
    low = np.ones((1, 3, 128, 128), dtype=np.float16)
    iou = np.array([[0.1, 0.8, 0.2]], dtype=np.float16)
    runner = type("Runner", (), {
        "encoder": FakeStageRunner(
            None if encoder_fail else {"image_embeddings": embedding}, fail=encoder_fail),
        "decoder": FakeStageRunner(
            None if decoder_fail else {"low_res_masks": low, "iou_predictions": iou},
            fail=decoder_fail),
    })()
    return MobileSAMPipeline(runner, make_binding(target)), runner


class ReadableStageTests(unittest.TestCase):
    def test_stages_expose_canonical_api_matching_legacy(self):
        pipeline, runner = fixtures()
        image = np.zeros((17, 31, 3), dtype=np.uint8)

        prepared = pipeline.encoder.preprocess(image)
        legacy = pipeline.encoder.pre_process(image)
        self.assertEqual(prepared.context, legacy.context)
        np.testing.assert_array_equal(
            prepared.tensors["normalized_images"], legacy.tensors["normalized_images"])

        raw = pipeline.encoder.infer(prepared)
        embedding = pipeline.encoder.postprocess(raw)
        self.assertEqual(embedding.shape, (1, 256, 32, 32))

        prepared_dec = pipeline.decoder.preprocess(embedding)
        raw_dec = pipeline.decoder.infer(prepared_dec)
        result = pipeline.decoder.postprocess(raw_dec)
        self.assertEqual(result["mask"].shape, (512, 512))

    def test_predict_equals_explicit_stage_chain_and_keeps_box_prompt(self):
        pipeline, runner = fixtures()
        image = np.zeros((17, 31, 3), dtype=np.uint8)
        custom_box = (10.0, 12.0, 200.0, 300.0)

        via_predict = pipeline.predict(image, box=custom_box)
        embedding = pipeline.encoder.postprocess(
            pipeline.encoder.infer(pipeline.encoder.preprocess(image)))
        explicit = pipeline.decoder.postprocess(
            pipeline.decoder.infer(pipeline.decoder.preprocess(embedding, box=custom_box)))
        np.testing.assert_array_equal(via_predict["mask"], explicit["mask"])

        # The custom box reached the decoder tensors; the default did not.
        decoder_boxes = runner.decoder.calls[-1]["boxes"].reshape(-1)
        np.testing.assert_array_equal(decoder_boxes, np.asarray(custom_box))
        self.assertFalse(np.array_equal(decoder_boxes, np.asarray(DEFAULT_BOX)))

        # Without an explicit box, the established default prompt is used.
        pipeline.predict(image)
        np.testing.assert_array_equal(
            runner.decoder.calls[-1]["boxes"].reshape(-1), np.asarray(DEFAULT_BOX))

    def test_encode_decode_helpers_and_consecutive_calls(self):
        pipeline, runner = fixtures()
        image_a = np.zeros((17, 31, 3), dtype=np.uint8)
        image_b = np.zeros((29, 11, 3), dtype=np.uint8)

        embedding = pipeline.encode_image(image_a)
        first = pipeline.decode_masks(embedding)
        again = pipeline.predict(image_a)
        np.testing.assert_array_equal(first["mask"], again["mask"])
        third = pipeline.predict(image_b)
        self.assertEqual(first["mask_index"], third["mask_index"])
        self.assertEqual(len(runner.encoder.calls), 3)
        self.assertEqual(len(runner.decoder.calls), 3)

    def test_stage_errors_attribute_encoder_and_decoder(self):
        encoder_fail, _ = fixtures(encoder_fail=True)
        with self.assertRaises(StageError) as caught:
            encoder_fail.predict(np.zeros((17, 31, 3), dtype=np.uint8))
        self.assertEqual(caught.exception.stage, "encoder")

        decoder_fail, runner = fixtures(decoder_fail=True)
        with self.assertRaises(StageError) as caught:
            decoder_fail.predict(np.zeros((17, 31, 3), dtype=np.uint8))
        self.assertEqual(caught.exception.stage, "decoder")
        self.assertEqual(len(runner.encoder.calls), 1)


if __name__ == "__main__":
    unittest.main()
