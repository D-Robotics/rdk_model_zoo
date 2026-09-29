"""Rotated boxes: bound raw data, platform NMS policy, angles and explicit context."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    LTRBOBBContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo26_obb import (
    YOLO26OBB,
    YOLO26OBBConfig,
)


def fixture(target="s100"):
    raw = {
        f"{kind}_{stride}": np.full(
            (1, 64 // stride, 64 // stride, c), -20 if kind == "cls" else 0, np.float32
        )
        for stride in (8, 16, 32)
        for kind, c in [("cls", 15), ("box", 4), ("angle", 1)]
    }
    raw["cls_8"][0, 3, 3, 2] = 6
    raw["box_8"][0, 3, 3] = [1, 1, 3, 1]
    raw["angle_8"][0, 3, 3, 0] = np.pi / 2
    ins = (
        {"image": (1, 3, 64, 64)}
        if target == "x5"
        else {"y": (1, 64, 64, 1), "uv": (1, 32, 32, 2)}
    )
    meta = RuntimeMetadata(
        "m",
        tuple(ins),
        ins,
        tuple(reversed(raw)),
        {n: v.shape for n, v in raw.items()},
        {n: np.uint8 for n in ins},
        {n: v.dtype for n, v in raw.items()},
    )
    selection = ModelSelection(
        "stub", target=target, task="obb", contract=LTRBOBBContract()
    )
    runner = ModelRunner(
        SimpleNamespace(run=lambda inputs: {"m": raw}), bind_model(selection, meta)
    )
    return YOLO26OBB(YOLO26OBBConfig("stub"), runner=runner), raw, meta, selection


class OBBBinding(unittest.TestCase):
    def test_raw_radians_and_owned_scalar_records(self):
        for target in ("x5", "s100", "s100p", "s600"):
            model, raw, _, _ = fixture(target)
            out = model.forward({})
            self.assertIs(out["angle_8"], raw["angle_8"])
            result = model.post_process(out, 64, 64)
            np.testing.assert_allclose(
                result[0]["rrect"][:4], [28, 36, 32, 16], atol=1e-5
            )
            self.assertAlmostEqual(np.cos(2 * result[0]["rrect"][4]), -1, places=6)
            self.assertEqual(result[0]["id"], 2)
            saved = result[0].copy()
            raw["box_8"].fill(0)
            raw["angle_8"].fill(0)
            self.assertEqual(result[0], saved)

    def test_x5_classwise_and_s_agnostic_nms(self):
        counts = []
        for target in ("x5", "s100"):
            model, raw, _, _ = fixture(target)
            raw["angle_8"].fill(0)
            raw["box_8"].fill(3)
            raw["cls_8"][0, 3, 4, 3] = 5
            counts.append(
                len(model.post_process(model.forward({}), 64, 64, nms_thres=0.2))
            )
        self.assertEqual(counts, [2, 1])

    def test_interleaved_context_and_integer_geometry(self):
        model, _, _, _ = fixture()
        image = np.zeros((37, 59, 3), np.uint8)
        a = model.pre_process(image)
        b = model.pre_process(np.zeros((93, 17, 3), np.uint8))
        result = model.post_process(model.forward(a), transform=a.transform)
        np.testing.assert_allclose(
            result[0]["rrect"][:4],
            [28 * 59 / 64, 24 * 37 / 40, 32 * 59 / 64, 16 * 37 / 40],
            atol=1e-5,
        )
        self.assertEqual(result, model.predict(image))
        with self.assertRaises(ValueError):
            model.post_process(model.forward({}), 59, 37, transform=b.transform)

    def test_invalid_protocol_metadata_and_thresholds(self):
        model, raw, meta, selection = fixture()
        for bad in (
            replace(meta, output_dtypes={**meta.output_dtypes, "angle_8": np.int32}),
            replace(meta, output_quantization={"angle_8": {"quant_type": "SCALE"}}),
        ):
            with self.assertRaises(ValueError):
                bind_model(selection, bad)
        for threshold in (0, 1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.post_process(model.forward({}), 64, 64, score_thres=threshold)
        for threshold in (-1, 2, np.nan):
            with self.assertRaises(ValueError):
                model.post_process(model.forward({}), 64, 64, nms_thres=threshold)
        raw["angle_8"][0, 0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            model.forward({})

    def test_empty_and_angle_override(self):
        model, raw, _, _ = fixture()
        model.cfg.angle_sign = -1
        model.cfg.angle_offset = 90
        r = model.post_process(model.forward({}), 64, 64)[0]
        self.assertAlmostEqual(r["rrect"][4], 0, places=6)
        for stride in (8, 16, 32):
            raw[f"cls_{stride}"].fill(-20)
        self.assertEqual(model.predict(np.zeros((64, 64, 3), np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
