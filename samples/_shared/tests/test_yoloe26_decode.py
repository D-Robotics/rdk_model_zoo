"""Pure YOLOE-26 PF policy checks; no model, SDK or board required."""

from pathlib import Path
import importlib.util
import unittest
import numpy as np
from samples._shared.yoloe26_decode import (
    decode_candidates,
    restore_masks,
    validate_shapes,
    OUTPUT_SHAPES,
)
from samples._shared.yoloe26_geometry import letterbox
from samples._shared.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SOURCE = legacy_path("s/samples/vision/yoloe26_seg/runtime/python/yoloe26seg.py")


def source_module():
    spec = importlib.util.spec_from_file_location("yoloe26_reference", SOURCE)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    from unittest.mock import patch
    import hashlib

    expected = "4bc54d43afa217ca6dde23c3eff7f7187eebf514c72506f1860c7767896493e0"
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != expected:
        raise AssertionError("Pinned source reference changed")
    with patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


def fixture():
    data = []
    for g in (80, 40, 20):
        data += [
            np.full((1, g, g, 4585), -30, np.float32),
            np.ones((1, g, g, 4), np.float32),
            np.ones((1, g, g, 32), np.float32),
        ]
    data.append(np.ones((1, 160, 160, 32), np.float32))
    data[0][0, 10, 10, 3] = 6
    data[0][0, 10, 10, 7] = 5
    data[0][0, 10, 11, 4] = 6
    data[3][0, 5, 5, 9] = 5.5
    return data


class YoloE26Kernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = source_module()
        cls.outputs = fixture()

    def test_source_topk_single_and_multiple_labels_no_nms(self):
        for single in (True, False):
            for count in (1, 3, 8):
                actual = decode_candidates(
                    self.outputs, max_det=count, single_label=single
                )
                expected = self.reference.decode_candidates(
                    self.outputs, max_det=count, single_label=single
                )
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
        # Boxes overlap; both remain. Multiple classes may share one anchor.
        boxes, scores, ids, _ = decode_candidates(
            self.outputs, max_det=8, single_label=False
        )
        self.assertEqual(ids.tolist(), [3, 4, 9, 7])
        np.testing.assert_array_equal(boxes[0], boxes[-1])

    def test_threshold_empty_and_strict_options(self):
        result = decode_candidates(self.outputs, score_threshold=0.999999)
        self.assertEqual([a.shape for a in result], [(0, 4), (0,), (0,), (0, 32)])
        for kwargs in (
            {"max_det": True},
            {"max_det": 1.5},
            {"max_det": 0},
            {"max_det": 8401},
            {"single_label": 1},
            {"score_threshold": np.nan},
            {"score_threshold": 0},
            {"score_threshold": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                (TypeError, ValueError)
            ):
                decode_candidates(self.outputs, **kwargs)
        bad = self.outputs.copy()
        bad[1] = np.ones(bad[1].shape, np.int32)
        with self.assertRaises(ValueError):
            decode_candidates(bad)
        with self.assertRaises(ValueError):
            decode_candidates(self.outputs[:-1])

    def test_metadata_dimensions_are_integers(self):
        validate_shapes(OUTPUT_SHAPES)
        for value in (True, 1.0, 1.5):
            shapes = list(OUTPUT_SHAPES)
            shapes[0] = (value, 80, 80, 4585)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_shapes(shapes)

    def test_nonfinite_raw_rejection(self):
        old = self.outputs[2][0, 0, 0, 0]
        try:
            self.outputs[2][0, 0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                decode_candidates(self.outputs)
        finally:
            self.outputs[2][0, 0, 0, 0] = old

    def test_letterbox_source_pixels_and_actual_integer_geometry(self):
        for shape in ((320, 640), (37, 59), (93, 17)):
            image = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(
                *shape, 3
            )
            actual, context = letterbox(image)
            expected, legacy = self.reference.letterbox(image)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(context.resized_size, (legacy[4], legacy[3]))
            self.assertEqual(context.scale_x, legacy[3] / shape[1])
            self.assertEqual(context.scale_y, legacy[4] / shape[0])
        with self.assertRaises(ValueError):
            letterbox(np.zeros((3, 4, 3), np.float32))

    def test_masks_match_source_on_exact_scale_and_own_data(self):
        image = np.zeros((320, 640, 3), np.uint8)
        _, context = letterbox(image)
        boxes = np.array(
            [[100, 200, 200, 300], [-2.5, 161.5, 5.5, 168.5], [700, 1, 710, 9]],
            np.float32,
        )
        coefficients = np.ones((3, 32), np.float32)
        proto = np.ones((160, 160, 32), np.float32)
        actual, masks = restore_masks(boxes, coefficients, proto, context)
        expected, reference_masks = self.reference.restore_masks(
            boxes, coefficients, proto, image.shape
        )
        np.testing.assert_array_equal(actual, expected)
        for a, b in zip(masks, reference_masks):
            np.testing.assert_array_equal(a, b)
        saved = actual.copy()
        boxes.fill(0)
        proto.fill(0)
        coefficients.fill(0)
        np.testing.assert_array_equal(actual, saved)
        self.assertTrue(masks[0].all())

    def test_nontrivial_mask_interpolation_matches_source(self):
        _, context = letterbox(np.zeros((320, 640, 3), np.uint8))
        random = np.random.default_rng(174)
        proto = random.normal(size=(160, 160, 32)).astype(np.float32)
        coefficients = random.normal(size=(2, 32)).astype(np.float32)
        boxes = np.array(
            [[101.25, 200.75, 201.5, 299.5], [0, 150, 90, 220]], np.float32
        )
        actual, masks = restore_masks(boxes, coefficients, proto, context)
        expected, reference = self.reference.restore_masks(
            boxes, coefficients, proto, (320, 640, 3)
        )
        np.testing.assert_array_equal(actual, expected)
        for a, b in zip(masks, reference):
            np.testing.assert_array_equal(a, b)
        self.assertGreater(np.count_nonzero(masks[0]), 0)
        self.assertLess(np.count_nonzero(masks[0]), masks[0].size)

    def test_calibration_matches_runtime_pixels(self):
        from samples._shared.yoloe26_geometry import prepare_rgb

        image = np.arange(37 * 59 * 3, dtype=np.uint8).reshape(37, 59, 3)
        np.testing.assert_array_equal(
            prepare_rgb(image), self.reference.prepare_rgb(image)
        )
        pixels, _ = letterbox(image)
        np.testing.assert_array_equal(
            prepare_rgb(image),
            pixels[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255,
        )

    def test_rounded_inverse_uses_recorded_scales_and_rejects_bad_shapes(self):
        _, context = letterbox(np.zeros((37, 59, 3), np.uint8))
        box = np.array([[100, 200, 200, 300]], np.float32)
        coefficients = np.ones((1, 32), np.float32)
        proto = np.ones((160, 160, 32), np.float32)
        restored, masks = restore_masks(box, coefficients, proto, context)
        left, top, _, _ = context.padding
        np.testing.assert_allclose(
            restored,
            [
                [
                    100 / context.scale_x,
                    (200 - top) / context.scale_y,
                    200 / context.scale_x,
                    (300 - top) / context.scale_y,
                ]
            ],
            rtol=1e-6,
        )
        x1, y1, x2, y2 = restored[0].astype(int)
        self.assertEqual(masks[0].shape, (y2 - y1, x2 - x1))
        with self.assertRaises(ValueError):
            restore_masks(box, coefficients[:0], proto, context)
        with self.assertRaises(ValueError):
            restore_masks(box, coefficients, proto.astype(np.int8), context)


if __name__ == "__main__":
    unittest.main()
