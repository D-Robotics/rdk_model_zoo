# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Canonical local stage API and explicit encoder→decoder orchestration.

The readable-runtime design requires the local pipeline to expose the
encoder and decoder stages with the canonical ``preprocess``/``infer``/
``postprocess`` spellings, and ``predict`` to show the real
encode → decode composition with per-stage error attribution — while the
SAM math itself stays in the shared stage implementations.
"""

from __future__ import annotations

import unittest

import numpy as np

from samples._shared.runtime_meta import RuntimeMetadata
from samples._shared.sam_binding import bind_model, resolve_selection
from samples._shared.sam_stages import StageError


def make_binding(target: str = "s100"):
    selection = resolve_selection("efficient_sam", target)
    encoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "encoder",
        "input_names": ("batched_images",),
        "input_shapes": {"batched_images": (1, 3, 512, 512)},
        "input_dtypes": {"batched_images": "float32"},
        "output_names": ("image_embeddings",),
        "output_shapes": {"image_embeddings": (1, 256, 32, 32)},
        "output_dtypes": {"image_embeddings": "float16"},
        "output_quants": {"image_embeddings": {"scale": 0.1, "zero_point": 3}},
    })
    decoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "decoder",
        "input_names": ("image_embeddings",),
        "input_shapes": {"image_embeddings": (1, 256, 32, 32)},
        "input_dtypes": {"image_embeddings": "float32"},
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


def fixtures(*, encoder_fail=False, decoder_fail=False):
    from samples.vision.efficient_sam.runtime.python.pipeline import EfficientSAMPipeline

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
    return EfficientSAMPipeline(runner, make_binding()), runner


class ReadableStageTests(unittest.TestCase):
    def test_stages_expose_canonical_api_matching_legacy(self):
        pipeline, runner = fixtures()
        image = np.zeros((17, 31, 3), dtype=np.uint8)

        prepared = pipeline.encoder.preprocess(image)
        legacy = pipeline.encoder.pre_process(image)
        self.assertEqual(prepared.context, legacy.context)
        np.testing.assert_array_equal(
            prepared.tensors["batched_images"], legacy.tensors["batched_images"])

        raw = pipeline.encoder.infer(prepared)
        self.assertIs(raw["image_embeddings"], runner.encoder.outputs["image_embeddings"])
        embedding = pipeline.encoder.postprocess(raw)
        self.assertEqual(embedding.dtype, np.float32)

        prepared_dec = pipeline.decoder.preprocess(embedding)
        raw_dec = pipeline.decoder.infer(prepared_dec)
        result = pipeline.decoder.postprocess(raw_dec)
        self.assertEqual(result["mask"].shape, (512, 512))

    def test_predict_equals_explicit_stage_chain(self):
        pipeline, runner = fixtures()
        image = np.zeros((17, 31, 3), dtype=np.uint8)

        via_predict = pipeline.predict(image)
        embedding = pipeline.encoder.postprocess(
            pipeline.encoder.infer(pipeline.encoder.preprocess(image)))
        explicit = pipeline.decoder.postprocess(
            pipeline.decoder.infer(pipeline.decoder.preprocess(embedding)))
        np.testing.assert_array_equal(via_predict["mask"], explicit["mask"])
        self.assertEqual(via_predict["iou"], explicit["iou"])
        self.assertEqual(via_predict["mask_index"], explicit["mask_index"])
        self.assertEqual(len(runner.encoder.calls), 2)
        self.assertEqual(len(runner.decoder.calls), 2)

    def test_encode_decode_helpers_and_consecutive_calls(self):
        pipeline, runner = fixtures()
        image_a = np.zeros((17, 31, 3), dtype=np.uint8)
        image_b = np.zeros((29, 11, 3), dtype=np.uint8)

        embedding = pipeline.encode_image(image_a)
        self.assertEqual(embedding.shape, (1, 256, 32, 32))
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
        # The decoder failure happened after exactly one encoder call.
        self.assertEqual(len(runner.encoder.calls), 1)


if __name__ == "__main__":
    unittest.main()
