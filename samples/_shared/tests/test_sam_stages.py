# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline source-regression and ownership tests for shared SAM stages."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples._shared.sam_binding import bind_model, resolve_selection
from samples._shared.sam_stages import DecoderStage, EncoderStage, SAMPipeline, StageError
from samples.vision.efficient_sam.runtime.python.pipeline import EfficientSAMPipeline
from samples.vision.mobile_sam.runtime.python.pipeline import MobileSAMPipeline


ROOT = Path(__file__).resolve().parents[2].parent


def make_binding(sample: str, target: str = "s100", *, output_dtype: str = "float16", box_shape=None, space: int = 128):
    selection = resolve_selection(sample, target)
    encoder_inputs = {"batched_images" if sample == "efficient_sam" else "normalized_images": (1, 3, 512, 512)}
    encoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "encoder",
        "input_names": tuple(encoder_inputs), "input_shapes": encoder_inputs,
        "input_dtypes": {name: "float32" for name in encoder_inputs},
        "output_names": ("image_embeddings",),
        "output_shapes": {"image_embeddings": (1, 256, 32, 32)},
        "output_dtypes": {"image_embeddings": output_dtype},
        "output_quants": {"image_embeddings": {"scale": 0.1, "zero_point": 3}},
    })
    decoder_inputs = {"image_embeddings": (1, 256, 32, 32)}
    if sample == "mobile_sam":
        decoder_inputs["boxes"] = box_shape or ((1, 4, 1, 1) if target == "x5" else (1, 4))
    decoder_meta = RuntimeMetadata.from_mapping({
        "model_name": "decoder",
        "input_names": tuple(decoder_inputs), "input_shapes": decoder_inputs,
        "input_dtypes": {name: "float32" for name in decoder_inputs},
        "output_names": ("low_res_masks", "iou_predictions"),
        "output_shapes": {
            "low_res_masks": (1, 3, 128 if target == "x5" else space, 128 if target == "x5" else space),
            "iou_predictions": (1, 3, 1, 1) if target == "x5" else (1, 3),
        },
        "output_dtypes": {"low_res_masks": output_dtype, "iou_predictions": output_dtype},
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


def load_source(name: str, path: Path):
    fake = types.ModuleType("hbm_runtime")

    class FakeRuntime:
        def __init__(self, _path):
            self.model_names = [name]

        def set_scheduling_params(self, **_kwargs):
            return None

    fake.HB_HBMRuntime = FakeRuntime
    old = sys.modules.get("hbm_runtime")
    sys.modules["hbm_runtime"] = fake
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("hbm_runtime", None)
        else:
            sys.modules["hbm_runtime"] = old


def source_class(sample: str, platform: str):
    path = ROOT / "platforms" / platform / "samples" / "vision" / sample / "runtime/python" / (
        "efficient_sam.py" if sample == "efficient_sam" else "mobile_sam.py")
    return load_source(f"source_{platform}_{sample}", path)


class SAMStageTests(unittest.TestCase):
    def test_all_eight_pair_bindings_are_usable(self):
        for sample in ("efficient_sam", "mobile_sam"):
            for target in ("x5", "s100", "s100p", "s600"):
                with self.subTest(sample=sample, target=target):
                    binding = make_binding(sample, target)
                    self.assertEqual(binding.encoder.sample, sample)
                    self.assertEqual(binding.decoder.target, target)

    def test_four_source_preprocess_paths_match_exactly(self):
        image = np.arange(17 * 31 * 3, dtype=np.uint8).reshape(17, 31, 3)
        for sample in ("efficient_sam", "mobile_sam"):
            for platform in ("x5", "s"):
                with self.subTest(sample=sample, platform=platform):
                    source = source_class(sample, platform)
                    if sample == "efficient_sam":
                        source_obj = source.EfficientSAMSegment(source.EfficientSAMConfig("e", "d"))
                        expected = source_obj.pre_process(image)["batched_images"]
                    else:
                        source_obj = source.MobileSAMSegment(source.MobileSAMConfig("e", "d"))
                        expected = source_obj.pre_process(image)["normalized_images"]
                    binding = make_binding(sample, "x5")
                    actual = EncoderStage(FakeStageRunner(), binding.encoder).pre_process(image).tensors
                    np.testing.assert_array_equal(actual[next(iter(actual))], expected)

    def test_four_source_postprocess_paths_match_threshold_and_resize(self):
        low = np.linspace(-1, 1, 3 * 128 * 128, dtype=np.float16).reshape(1, 3, 128, 128)
        low[0, 1, 0, 0] = 0.0
        iou = np.array([[0.1, 0.9, 0.2]], dtype=np.float16)
        for sample in ("efficient_sam", "mobile_sam"):
            for platform in ("x5", "s"):
                with self.subTest(sample=sample, platform=platform):
                    source = source_class(sample, platform)
                    source_obj = (source.EfficientSAMSegment(source.EfficientSAMConfig("e", "d"))
                                  if sample == "efficient_sam" else
                                  source.MobileSAMSegment(source.MobileSAMConfig("e", "d")))
                    source_result = source_obj.post_process({"decoder": {
                        "low_res_masks": low, "iou_predictions": iou,
                    }})
                    binding = make_binding(sample, "x5", output_dtype="float16")
                    actual = DecoderStage(FakeStageRunner(), binding.decoder).post_process({
                        "low_res_masks": low, "iou_predictions": iou.reshape(1, 3, 1, 1),
                    })
                    np.testing.assert_array_equal(actual["mask"], source_result["mask"])
                    self.assertEqual(actual["mask_index"], source_result["mask_index"])
                    self.assertEqual(actual["iou"], source_result["iou"])

    def test_encoder_post_casts_owned_and_keeps_raw_identity(self):
        binding = make_binding("efficient_sam", output_dtype="float16")
        raw = np.zeros((1, 256, 32, 32), dtype=np.float16)
        outputs = {"image_embeddings": raw}
        result = EncoderStage(FakeStageRunner(), binding.encoder).post_process(outputs)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.owndata)
        self.assertIs(outputs["image_embeddings"], raw)

    def test_efficient_decoder_rejects_unsupported_runtime_box(self):
        embedding = np.zeros((1, 256, 32, 32), dtype=np.float32)
        for target in ("x5", "s100", "s100p", "s600"):
            with self.subTest(target=target):
                binding = make_binding("efficient_sam", target)
                stage = DecoderStage(FakeStageRunner(), binding.decoder)
                self.assertEqual(set(stage.pre_process(embedding).tensors), {"image_embeddings"})
                with self.assertRaisesRegex(ValueError, "fixed.*prompt.*box"):
                    stage.pre_process(embedding, box=(0, 0, 1, 1))

    def test_decoder_box_shape_follows_x5_or_s_metadata(self):
        embedding = np.zeros((1, 256, 32, 32), dtype=np.float32)
        for target, shape in (("x5", (1, 4, 1, 1)), ("s100", (1, 4))):
            with self.subTest(target=target):
                binding = make_binding("mobile_sam", target)
                prepared = DecoderStage(FakeStageRunner(), binding.decoder).pre_process(embedding)
                self.assertEqual(prepared.tensors["boxes"].shape, shape)
                with self.assertRaises(ValueError):
                    DecoderStage(FakeStageRunner(), binding.decoder).pre_process(embedding, box=(0, 1, 0, 511))

    def test_forward_rejects_wrong_dtype_nonfinite_and_preserves_valid_arrays(self):
        binding = make_binding("efficient_sam", output_dtype="float16")
        valid = np.zeros((1, 256, 32, 32), dtype=np.float16)
        runner = FakeStageRunner({"image_embeddings": valid})
        stage = EncoderStage(runner, binding.encoder)
        prepared = stage.pre_process(np.zeros((17, 31, 3), dtype=np.uint8))
        raw = stage.forward(prepared)
        self.assertIs(raw["image_embeddings"], valid)
        runner.outputs = {"image_embeddings": valid.astype(np.float32)}
        with self.assertRaises(MetadataMismatchError):
            stage.forward(prepared)
        with self.assertRaises(MetadataMismatchError):
            stage.post_process({"image_embeddings": valid.astype(np.float32)})
        bad = valid.copy()
        bad.flat[0] = np.nan
        runner.outputs = {"image_embeddings": bad}
        with self.assertRaises(MetadataMismatchError):
            stage.forward(prepared)
        with self.assertRaises(MetadataMismatchError):
            stage.forward({"batched_images": prepared.tensors["batched_images"].astype(np.float64)})

    def test_explicit_stage_chain_equals_predict_for_a_b_a_contexts(self):
        binding = make_binding("mobile_sam", "s100", output_dtype="float16")
        embedding = np.ones((1, 256, 32, 32), dtype=np.float16)
        low = np.ones((1, 3, 128, 128), dtype=np.float16)
        iou = np.array([[0.1, 0.8, 0.2]], dtype=np.float16)

        class Runner:
            def __init__(self):
                self.encoder = FakeStageRunner({"image_embeddings": embedding})
                self.decoder = FakeStageRunner({"low_res_masks": low, "iou_predictions": iou})

        runner = Runner()
        pipeline = SAMPipeline(runner, binding)
        image_a = np.zeros((17, 31, 3), dtype=np.uint8)
        image_b = np.zeros((29, 11, 3), dtype=np.uint8)
        prepared = pipeline.encoder.pre_process(image_a)
        encoded = pipeline.encoder.post_process(pipeline.encoder.forward(prepared))
        explicit = pipeline.decoder.post_process(pipeline.decoder.forward(pipeline.decoder.pre_process(encoded)))
        composed = pipeline.predict(image_a)
        self.assertEqual(explicit["mask_index"], composed["mask_index"])
        np.testing.assert_array_equal(explicit["mask"], composed["mask"])
        self.assertEqual(prepared.context.source_shape, (17, 31, 3))
        result_a = pipeline.predict(image_a)
        result_b = pipeline.predict(image_b)
        result_a_again = pipeline.predict(image_a)
        np.testing.assert_array_equal(result_a["mask"], result_a_again["mask"])
        self.assertEqual(result_a["mask_index"], result_b["mask_index"])
        self.assertEqual(len(runner.encoder.calls), 5)
        self.assertEqual(len(runner.decoder.calls), 5)

    def test_pipeline_attributes_encoder_and_decoder_failures(self):
        binding = make_binding("efficient_sam")
        image = np.zeros((17, 31, 3), dtype=np.uint8)

        class EncoderFail:
            encoder = FakeStageRunner(fail=True)
            decoder = FakeStageRunner()

        with self.assertRaisesRegex(StageError, "encoder"):
            SAMPipeline(EncoderFail(), binding).predict(image)
        self.assertEqual(EncoderFail.decoder.calls, [])

        embedding = np.zeros((1, 256, 32, 32), dtype=np.float16)
        class DecoderFail:
            encoder = FakeStageRunner({"image_embeddings": embedding})
            decoder = FakeStageRunner(fail=True)

        with self.assertRaisesRegex(StageError, "decoder"):
            SAMPipeline(DecoderFail(), binding).predict(image)

    def test_sample_pipelines_are_thin_shared_wrappers(self):
        self.assertTrue(issubclass(EfficientSAMPipeline, SAMPipeline))
        self.assertTrue(issubclass(MobileSAMPipeline, SAMPipeline))

    def test_sample_wrappers_fix_public_prompt_and_binding(self):
        image = np.zeros((17, 31, 3), dtype=np.uint8)
        efficient_binding = make_binding("efficient_sam")
        mobile_binding = make_binding("mobile_sam")
        runner = type("Runner", (), {
            "encoder": FakeStageRunner({"image_embeddings": np.zeros((1, 256, 32, 32), dtype=np.float16)}),
            "decoder": FakeStageRunner({
                "low_res_masks": np.zeros((1, 3, 128, 128), dtype=np.float16),
                "iou_predictions": np.zeros((1, 3, 1, 1), dtype=np.float16),
            }),
        })()
        with self.assertRaisesRegex(ValueError, "efficient_sam"):
            EfficientSAMPipeline(runner, mobile_binding)
        with self.assertRaisesRegex(ValueError, "mobile_sam"):
            MobileSAMPipeline(runner, efficient_binding)
        with self.assertRaises(TypeError):
            EfficientSAMPipeline(runner, efficient_binding).predict(image, box=(0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
