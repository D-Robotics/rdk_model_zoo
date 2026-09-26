"""Real runner/binding fixtures: transforms belong to post_process, never forward."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    LTRBDetectionContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import (
    YoloDetect,
    YoloDetectConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.tensor_io import normalize_dtype
from samples.vision.ultralytics_yolo.runtime.python.decode import decode_dfl


def fixture(*, sdk=False, channels=False, ltrb=False):
    shapes = {
        f"{kind}_{stride}": (1, 64 // stride, 64 // stride, c)
        for stride in (8, 16, 32)
        for kind, c in [("cls", 1), ("box", 4 if ltrb else 64)]
    }
    raw = {
        name: np.full(
            shape,
            8 if ltrb and name.startswith("box") else -24,
            dtype=np.float32 if ltrb else np.int8,
        )
        for name, shape in shapes.items()
    }
    raw["cls_8"][0, 3, 3, 0] = 16
    if not ltrb:
        for stride in (8, 16, 32):
            raw[f"box_{stride}"][..., [1, 17, 33, 49]] = 16
    quants = {}
    for name, shape in shapes.items():
        if sdk:
            scale = (
                np.linspace(0.125, 0.375, shape[-1], dtype=np.float32)
                if channels
                else np.array([0.25], np.float32)
            )
            quants[name] = SimpleNamespace(
                quant_type=SimpleNamespace(name="NONE" if ltrb else "SCALE"),
                scale=np.array([], np.float32) if ltrb else scale,
                zero_point=np.array([3], np.int32),
                axis=3,
            )
        else:
            quants[name] = {"scale": 0.25, "zero_point": 3}
    runtime = SimpleNamespace(
        model_names=["m"],
        input_names={"m": ["image"]},
        input_shapes={"m": {"image": (1, 3, 64, 64)}},
        input_dtypes={"m": {"image": np.dtype("uint8")}},
        output_names={"m": list(shapes)},
        output_shapes={"m": shapes},
        output_dtypes={"m": {n: a.dtype for n, a in raw.items()}},
        run=lambda tensors: {"m": raw},
    )
    setattr(runtime, "output_quants" if sdk else "output_quantization", {"m": quants})
    contract = (LTRBDetectionContract if ltrb else DFLDetectionContract)(
        classes=1, strides=(8, 16, 32)
    )
    metadata = RuntimeMetadata.from_runtime(runtime)
    binding = bind_model(
        ModelSelection("fixture.bin", target="x5", contract=contract), metadata
    )
    return ModelRunner(runtime, binding, metadata), raw, quants, contract


class ForwardPurity(unittest.TestCase):
    def test_scalar_quantized_forward_preserves_original_arrays_and_values(self):
        runner, physical, _, contract = fixture()
        task = YoloDetect(
            YoloDetectConfig(
                "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
            ),
            runner=runner,
        )
        with patch.object(runner.model, "run", wraps=runner.model.run) as run:
            outputs = task.forward(
                {"m": {"image": np.zeros(64 * 64 * 3 // 2, np.uint8)}}
            )
        self.assertEqual(run.call_count, 1)
        for name, value in physical.items():
            self.assertIs(outputs[name], value)
            self.assertEqual(outputs[name].dtype, np.int8)
        expected = decode_dfl(
            {n: (v.astype(np.float32) - 3) * 0.25 for n, v in physical.items()},
            contract,
            input_size=(64, 64),
            score_thres=0.25,
            nms_thres=0.45,
        )
        actual = task.post_process(outputs, 64, 64)
        # All selected fixture boxes are interior, so inverse mapping is identity.
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
        self.assertEqual(physical["cls_8"][0, 3, 3, 0], 16)

    def test_sdk_channel_scale_and_scalar_offset_apply_only_in_postprocess(self):
        runner, physical, quants, contract = fixture(sdk=True, channels=True)
        task = YoloDetect(
            YoloDetectConfig(
                "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
            ),
            runner=runner,
        )
        outputs = task.forward({})
        for name, value in physical.items():
            self.assertIs(outputs[name], value)
        expected = {
            n: (a.astype(np.float32) - 3) * quants[n].scale.reshape(1, 1, 1, -1)
            for n, a in physical.items()
        }
        decoded = decode_dfl(
            expected, contract, input_size=(64, 64), score_thres=0.25, nms_thres=0.45
        )
        result = task.post_process(outputs, 64, 64)
        for a, b in zip(result, decoded):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)

    def test_sdk_none_descriptor_does_not_broaden_ltrb_quantized_contract(self):
        runner, physical, _, _ = fixture(sdk=True, ltrb=True)
        outputs = runner({})
        for name, value in physical.items():
            self.assertIs(outputs[name], value)
        bad = replace(runner.metadata, output_quantization={"cls_8": {"scale": 0.25}})
        with self.assertRaises(BindingError):
            bind_model(runner.binding.selection, bad)

    def test_sdk_integer_enum_names_are_numeric_not_numpy_string_dtypes(self):
        for token, expected in [
            ("S8", "int8"),
            ("S16", "int16"),
            ("S32", "int32"),
            ("F16", "float16"),
        ]:
            with self.subTest(token=token):
                self.assertEqual(
                    normalize_dtype(SimpleNamespace(name=token)), np.dtype(expected)
                )

    def test_result_arrays_remain_owned_after_runtime_reuses_raw_buffers(self):
        runner, physical, _, contract = fixture(sdk=True, channels=True)
        task = YoloDetect(
            YoloDetectConfig(
                "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
            ),
            runner=runner,
        )
        result = task.predict(np.zeros((64, 64, 3), np.uint8))
        snapshot = tuple(value.copy() for value in result)
        self.assertGreater(len(result.scores), 0)
        for value in physical.values():
            value.fill(-24)
        self.assertEqual(len(task.predict(np.zeros((64, 64, 3), np.uint8)).scores), 0)
        for before, after in zip(snapshot, result):
            np.testing.assert_array_equal(before, after)

    def test_raw_carrier_cannot_be_rebound_or_hide_shape_changes(self):
        runner, physical, _, contract = fixture()
        other, _, _, _ = fixture()
        raw = runner({})
        with self.assertRaises(BindingError):
            other.binding.read_outputs(raw)
        physical["cls_8"].shape = (64, 1)
        with self.assertRaises(BindingError):
            runner.binding.read_outputs(raw)

    def test_non_real_numeric_metadata_is_rejected_even_with_quantization(self):
        runner, _, _, _ = fixture()
        for dtype in (np.complex64, np.bool_):
            meta = replace(
                runner.metadata,
                output_dtypes={name: np.dtype(dtype) for name in runner.output_names},
            )
            with self.subTest(dtype=dtype), self.assertRaises(BindingError):
                bind_model(runner.binding.selection, meta)

    def test_invalid_channel_scale_axis_and_nonpositive_scale_fail_before_run(self):
        runner, _, _, _ = fixture()
        for descriptor in [
            {"scale": [0.1, 0.2], "zero_point": 0, "axis": 3},
            {"scale": [0.1] * 64, "zero_point": 0, "axis": 5},
            {"scale": -0.1, "zero_point": 0},
        ]:
            meta = replace(
                runner.metadata,
                output_quantization={
                    **runner.metadata.output_quantization,
                    "box_8": descriptor,
                },
            )
            with self.subTest(descriptor=descriptor), self.assertRaises(BindingError):
                bind_model(runner.binding.selection, meta)
