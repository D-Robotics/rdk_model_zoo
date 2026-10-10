"""Pose stage contracts: raw transport, named heads, visibility and geometry."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import warnings
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    DFLPoseContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.backend import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.pose import (
    YoloPose,
    YoloPoseConfig,
)


def fixture(*, target="s100", quantized=False):
    raw = {
        f"{kind}_{stride}": np.full(
            (1, 64 // stride, 64 // stride, channels),
            -24 if kind in ("cls", "box") else 4,
            np.int16,
        )
        for stride in (8, 16, 32)
        for kind, channels in [("cls", 1), ("box", 64), ("kpts", 51)]
    }
    raw["cls_8"][0, 3, 3, 0] = 20
    for stride in (8, 16, 32):
        raw[f"box_{stride}"][..., [1, 17, 33, 49]] = 20
        raw[f"kpts_{stride}"][..., 2::3] = np.linspace(-20, 20, 17, dtype=np.int16)
    quants = {
        name: SimpleNamespace(
            quant_type=SimpleNamespace(name="SCALE"),
            scale=np.linspace(0.125, 0.375, value.shape[-1], dtype=np.float32),
            zero_point=np.array([3], np.int32),
            axis=3,
        )
        for name, value in raw.items()
    }
    if not quantized:
        raw = {
            name: (value.astype(np.float32) - 3)
            * quants[name].scale.reshape(1, 1, 1, -1)
            for name, value in raw.items()
        }
        quants = {}
    ins = (
        {"image": (1, 3, 64, 64)}
        if target == "x5"
        else {"y": (1, 64, 64, 1), "uv": (1, 32, 32, 2)}
    )
    runtime = SimpleNamespace(
        model_names=["m"],
        input_names={"m": list(ins)},
        input_shapes={"m": ins},
        input_dtypes={"m": {n: "U8" for n in ins}},
        output_names={"m": list(reversed(raw))},
        output_shapes={"m": {n: v.shape for n, v in raw.items()}},
        output_dtypes={"m": {n: v.dtype for n, v in raw.items()}},
        output_quants={"m": quants},
        run=lambda inputs: {"m": raw},
    )
    metadata = RuntimeMetadata.from_runtime(runtime)
    binding = bind_model(
        ModelSelection(
            "fixture.hbm", target=target, task="pose", contract=DFLPoseContract()
        ),
        metadata,
    )
    runner = ModelRunner(runtime, binding, metadata)
    return (
        YoloPose(YoloPoseConfig("fixture.hbm", nms_thres=0.7), runner=runner),
        raw,
        quants,
    )


class PoseBinding(unittest.TestCase):
    def assert_result_equal(self, a, b):
        self.assertEqual(len(a), 5)
        self.assertEqual(len(b), 5)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-5)

    def test_float_forward_and_reference_decode(self):
        for target in ("x5", "s100", "s600"):
            with self.subTest(target=target):
                task, physical, _ = fixture(target=target)
                reference, _, _ = fixture(target=target, quantized=False)
                with patch.object(task.model, "run", wraps=task.model.run) as run:
                    raw = task.forward({})
                self.assertEqual(run.call_count, 1)
                for name, array in physical.items():
                    self.assertIs(raw[name], array)
                result = task.post_process(raw, 64, 64)
                self.assertEqual(result[3].shape, (1, 17, 2))
                self.assertEqual(result[4].shape, (1, 17, 1))
                self.assert_result_equal(
                    result, reference.post_process(reference.forward({}), 64, 64)
                )

    def test_visibility_is_once_activated(self):
        task, raw, _ = fixture(quantized=False)
        raw["kpts_8"][0, 3, 3, 2::3] = np.array([-1000, 0, 1000] + [2] * 14, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = task.post_process(task.forward({}), 64, 64)
        np.testing.assert_array_equal(result[4][0, :3, 0], [0, 0.5, 1])

    def test_actual_integer_geometry_and_interleaved_context(self):
        task, raw, _ = fixture(quantized=False)
        raw["kpts_8"][0, 3, 3, 0::3] = 0.25
        raw["kpts_8"][0, 3, 3, 1::3] = 0.25
        for resize in (0, 1):
            task.cfg.resize_type = resize
            image = np.zeros((31, 73, 3), np.uint8)
            pa = task.pre_process(image)
            pb = task.pre_process(np.zeros((67, 23, 3), np.uint8))
            result = task.post_process(task.forward(pa.tensors), transform=pa.transform)
            transform = pa.transform
            expected = np.array(
                [
                    (28 - transform.padding[0]) / transform.scale_x,
                    (28 - transform.padding[1]) / transform.scale_y,
                ]
            )
            expected = np.clip(expected, [0, 0], [73, 31])
            np.testing.assert_allclose(
                result[3][0], np.tile(expected, (17, 1)), rtol=1e-6
            )
            self.assert_result_equal(result, task.predict(image))
            self.assertEqual(pb.transform.original_size, (67, 23))
            self.assert_result_equal(
                task.post_process(task.forward(pa), 73, 31), result
            )
            with self.assertRaises(ValueError):
                task.post_process(task.forward({}))
            with self.assertRaises(ValueError):
                task.post_process(task.forward({}), 99, 99, transform=pa.transform)

    def test_nms_keeps_each_skeleton_with_its_detection(self):
        task, raw, _ = fixture(quantized=False)
        for stride in (8, 16, 32):
            raw[f"cls_{stride}"].fill(-20)
        raw["cls_8"][0, 3, 3, 0] = 8
        raw["cls_8"][
            0, 3, 4, 0
        ] = 6  # Overlapping neighbour should be suppressed at low IoU threshold.
        raw["kpts_8"][0, 3, 3, 2::3] = 3
        raw["kpts_8"][0, 3, 4, 2::3] = -3
        result = task.post_process(task.forward({}), 64, 64, nms_thres=0.01)
        self.assertEqual(len(result[0]), 1)
        np.testing.assert_array_equal(
            result[4], np.full((1, 17, 1), 1 / (1 + np.exp(-3)), np.float32)
        )

    def test_empty_owned_results_and_strict_metadata(self):
        task, raw, _ = fixture()
        result = task.post_process(task.forward({}), 64, 64)
        saved = [a.copy() for a in result]
        for a in raw.values():
            a.fill(-24)
        for a, b in zip(result, saved):
            np.testing.assert_array_equal(a, b)
        empty = task.post_process(task.forward({}), 64, 64)
        self.assertEqual(
            [a.shape for a in empty], [(0, 4), (0,), (0,), (0, 17, 2), (0, 17, 1)]
        )
        self.assertEqual(
            [a.dtype for a in empty],
            [np.float32, np.float32, np.int64, np.float32, np.float32],
        )
        for edits in [
            {"output_quantization": {"box_8": {"scale": 0.25}}},
            {
                "output_shapes": {
                    **task.binding.metadata.output_shapes,
                    "kpts_8": (1, 8, 8, 50),
                }
            },
        ]:
            with self.assertRaises(ValueError):
                bind_model(
                    task.binding.selection, replace(task.binding.metadata, **edits)
                )
        other, _, _ = fixture()
        with self.assertRaises(ValueError):
            task.post_process(other.forward({}), 64, 64)
        with self.assertRaises(ValueError):
            task.post_process({n: a.astype(np.int8) for n, a in raw.items()}, 64, 64)
        with self.assertRaises(ValueError):
            DFLPoseContract(nkpt=16)
        with self.assertRaises(ValueError):
            task.pre_process(np.zeros((0, 1, 3), np.uint8))
        for threshold in (float("nan"), 0, 1):
            with self.assertRaises(ValueError):
                task.post_process(task.forward({}), 64, 64, score_thres=threshold)


if __name__ == "__main__":
    unittest.main()
