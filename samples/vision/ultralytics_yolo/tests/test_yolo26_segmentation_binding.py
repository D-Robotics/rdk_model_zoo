"""LTRB segmentation retains probability-resize-threshold order and owned ROI masks."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    LTRBSegmentationContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.backend import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.segment import (
    YOLO26Seg,
    YOLO26SegConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.segment import YoloSeg


def fixture(target="s100", chw=False, classes=1, size=64):
    raw = {
        f"{kind}_{stride}": np.full(
            (1, size // stride, size // stride, c),
            -20 if kind == "cls" else 1,
            np.float32,
        )
        for stride in (8, 16, 32)
        for kind, c in [("cls", classes), ("box", 4), ("mces", 32)]
    }
    raw["cls_8"][0, 3, 3, 0] = 6
    proto = np.ones((1, size // 4, size // 4, 32), np.float32)
    proto[:, :, :7, :] = -1
    raw["protos"] = proto.transpose(0, 3, 1, 2).copy() if chw else proto
    ins = (
        {"image": (1, 3, size, size)}
        if target == "x5"
        else {"y": (1, size, size, 1), "uv": (1, size // 2, size // 2, 2)}
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
        "stub",
        target=target,
        task="segment",
        contract=LTRBSegmentationContract(classes=classes),
    )
    runner = ModelRunner(
        SimpleNamespace(run=lambda inputs: {"m": raw}), bind_model(selection, meta)
    )
    return (
        YOLO26Seg(YOLO26SegConfig("stub", classes_num=classes), runner=runner),
        raw,
        meta,
        selection,
    )


class DirectSegmentation(unittest.TestCase):
    def test_raw_identity_layout_and_mask_geometry(self):
        reference = None
        for target in ("x5", "s100", "s100p", "s600"):
            for chw in (False, True):
                model, raw, _, _ = fixture(target, chw)
                output = model.forward({})
                self.assertIs(output["protos"], raw["protos"])
                result = model.post_process(output, 64, 64)
                np.testing.assert_array_equal(result[0], [[20, 20, 36, 36]])
                self.assertEqual(result[3][0].shape, (16, 16))
                self.assertEqual(result[3][0].dtype, np.bool_)
                self.assertTrue(result[3][0].flags.owndata)
                self.assertFalse(result[3][0][:, :6].any())
                self.assertTrue(result[3][0][:, 10:].all())
                if reference is None:
                    reference = result
                else:
                    for a, b in zip(result[:3], reference[:3]):
                        np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(result[3][0], reference[3][0])
                self.assertIs(YOLO26Seg.pre_process, YoloSeg.pre_process)

    def test_empty_border_and_buffer_lifetime(self):
        model, raw, _, _ = fixture()
        raw["protos"].fill(1)
        raw["box_8"].fill(10)
        result = model.post_process(model.forward({}), 64, 64)
        np.testing.assert_array_equal(result[0], [[0, 0, 64, 64]])
        self.assertTrue(result[3][0].all())
        raw["protos"].fill(-1)
        self.assertTrue(result[3][0].all())
        for stride in (8, 16, 32):
            raw[f"cls_{stride}"].fill(-20)
        empty = model.post_process(model.forward({}), 64, 64)
        self.assertEqual(empty[0].shape, (0, 4))
        self.assertEqual(empty[2].dtype, np.int64)
        self.assertEqual(empty[3], [])

    def test_context_pairing_and_mask_coefficients_follow_nms(self):
        model, raw, _, _ = fixture()
        raw["protos"].fill(1)
        raw["cls_8"][0, 3, 4, 0] = 8
        raw["box_8"].fill(3)
        raw["mces_8"][0, 3, 4, :] = -1
        image = np.zeros((37, 59, 3), np.uint8)
        a = model.pre_process(image)
        b = model.pre_process(np.zeros((97, 17, 3), np.uint8))
        result = model.post_process(
            model.forward(a), transform=a.transform, nms_thres=0.3
        )
        self.assertEqual(len(result[3]), 1)
        self.assertFalse(result[3][0].any())
        predicted = model.predict(image, nms_thres=0.3)
        for x, y in zip(result[:3], predicted[:3]):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(result[3][0], predicted[3][0])
        with self.assertRaises(ValueError):
            model.post_process(model.forward({}), 59, 37, transform=b.transform)

    def test_saturated_class_logits_keep_correct_class(self):
        model, raw, _, _ = fixture(classes=2)
        raw["cls_8"][0, 3, 3, :] = [20, 21]
        result = model.post_process(model.forward({}), 64, 64)
        np.testing.assert_array_equal(result[2], [1])

    def test_incompatible_binding_and_metadata_rejected(self):
        model, raw, meta, selection = fixture()
        for bad in (
            replace(meta, output_dtypes={**meta.output_dtypes, "protos": np.int32}),
            replace(meta, output_shapes={**meta.output_shapes, "box_8": (1, 8, 8, 64)}),
            replace(meta, output_quantization={"protos": {"quant_type": "SCALE"}}),
        ):
            with self.assertRaises(ValueError):
                bind_model(selection, bad)
        from test_segmentation_binding import fixture as dfl_fixture

        dfl, _, _ = dfl_fixture()
        with self.assertRaises(ValueError):
            YOLO26Seg(YOLO26SegConfig("stub"), runner=dfl.runner)
        raw["protos"][0, 0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            model.forward({})


if __name__ == "__main__":
    unittest.main()
