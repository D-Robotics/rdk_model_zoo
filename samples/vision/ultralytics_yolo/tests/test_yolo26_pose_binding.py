"""YOLO26 direct boxes/point offsets share transport, never DFL point formulas."""

from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
import unittest, warnings
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    LTRBPoseContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo26_pose import (
    YOLO26Pose,
    YOLO26PoseConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_pose import YoloPose


def fixture(target="s100"):
    arrays = {
        f"{kind}_{stride}": np.full(
            (1, 64 // stride, 64 // stride, c),
            -20 if kind == "cls" else 0.25,
            np.float32,
        )
        for stride in (8, 16, 32)
        for kind, c in [("cls", 1), ("box", 4), ("kpts", 51)]
    }
    arrays["cls_8"][0, 3, 3, 0] = 6
    for stride in (8, 16, 32):
        arrays[f"box_{stride}"].fill(1)
    ins = (
        {"image": (1, 3, 64, 64)}
        if target == "x5"
        else {"y": (1, 64, 64, 1), "uv": (1, 32, 32, 2)}
    )
    metadata = RuntimeMetadata(
        "m",
        tuple(ins),
        ins,
        tuple(reversed(arrays)),
        {k: v.shape for k, v in arrays.items()},
        {n: np.uint8 for n in ins},
        {k: v.dtype for k, v in arrays.items()},
    )
    selection = ModelSelection(
        "stub", target=target, task="pose", contract=LTRBPoseContract()
    )
    runner = ModelRunner(
        SimpleNamespace(run=lambda inputs: {"m": arrays}),
        bind_model(selection, metadata),
    )
    return (
        YOLO26Pose(YOLO26PoseConfig("stub"), runner=runner),
        arrays,
        metadata,
        selection,
    )


class DirectPoseBinding(unittest.TestCase):
    def test_identity_direct_formula_and_once_activated_scores(self):
        for target in ("x5", "s100", "s100p", "s600"):
            model, arrays, _, _ = fixture(target)
            arrays["kpts_8"][0, 3, 3, 2::3] = [-1000, 0, 1000] + [2] * 14
            raw = model.forward({})
            self.assertIs(raw["box_8"], arrays["box_8"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                result = model.post_process(raw, 64, 64)
            np.testing.assert_allclose(result[0], [[20, 20, 36, 36]])
            np.testing.assert_allclose(result[3], 30)
            np.testing.assert_array_equal(result[4][0, :3, 0], [0, 0.5, 1])
            self.assertIs(YOLO26Pose.pre_process, YoloPose.pre_process)
            self.assertIs(YOLO26Pose.post_process, YoloPose.post_process)
            for a in result:
                self.assertTrue(a.flags.owndata)

    def test_nms_pairing_empty_and_buffer_lifetime(self):
        model, arrays, _, _ = fixture()
        arrays["cls_8"][0, 3, 4, 0] = 8
        arrays["box_8"].fill(3)
        arrays["kpts_8"][0, 3, 4, 0::3] = -0.25
        result = model.post_process(model.forward({}), 64, 64, nms_thres=0.3)
        self.assertEqual(len(result[0]), 1)
        np.testing.assert_allclose(result[3][..., 0], 34)
        saved = [a.copy() for a in result]
        for stride in (8, 16, 32):
            arrays[f"cls_{stride}"].fill(-20)
        arrays["kpts_8"].fill(99)
        for a, b in zip(result, saved):
            np.testing.assert_array_equal(a, b)
        empty = model.post_process(model.forward({}), 64, 64)
        self.assertEqual(
            [a.shape for a in empty], [(0, 4), (0,), (0,), (0, 17, 2), (0, 17, 1)]
        )
        self.assertEqual(empty[2].dtype, np.int64)

    def test_explicit_context_uses_actual_integer_geometry(self):
        model, _, _, _ = fixture()
        img = np.zeros((37, 59, 3), np.uint8)
        first = model.pre_process(img)
        second = model.pre_process(np.zeros((93, 17, 3), np.uint8))
        result = model.post_process(model.forward(first), transform=first.transform)
        np.testing.assert_allclose(
            result[3][0], np.tile([30 * 59 / 64, 18 * 37 / 40], (17, 1)), rtol=1e-6
        )
        for a, b in zip(result, model.predict(img)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            model.post_process(model.forward({}), 59, 37, transform=second.transform)

    def test_malformed_metadata_and_incompatible_protocol_fail(self):
        model, arrays, meta, selection = fixture()
        for bad in (
            replace(meta, output_dtypes={**meta.output_dtypes, "box_8": np.int32}),
            replace(meta, output_quantization={"box_8": {"quant_type": "SCALE"}}),
            replace(meta, output_shapes={**meta.output_shapes, "box_8": (1, 8, 8, 64)}),
        ):
            with self.assertRaises(ValueError):
                bind_model(selection, bad)
        from test_pose_binding import fixture as dfl_fixture

        dfl, _, _ = dfl_fixture()
        with self.assertRaises(ValueError):
            YOLO26Pose(YOLO26PoseConfig("stub"), runner=dfl.runner)
        arrays["kpts_8"][0, 0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            model.forward({})


if __name__ == "__main__":
    unittest.main()
