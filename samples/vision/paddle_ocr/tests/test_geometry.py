"""Regression tests for target-local OCR geometry policies."""

from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


class GeometryTests(unittest.TestCase):
    def test_degenerate_crops_preserve_each_original_source_result(self):
        from samples.vision.paddle_ocr.runtime.python.geometry import crop_and_rotate_image

        # A non-extreme textured fixture exercises OpenCV's zero-dimension
        # behavior without relying on a blank image.
        image = (
            np.arange(9 * 13 * 3, dtype=np.uint16).reshape(9, 13, 3) % 251
        ).astype(np.uint8)
        zero = np.zeros((4, 2), dtype=np.int64)
        collinear = np.array([[0, 0], [3, 0], [3, 0], [0, 0]], dtype=np.int64)

        x5_zero = crop_and_rotate_image(image, zero, target="x5")
        self.assertEqual(x5_zero.shape, (9, 13, 3))
        self.assertTrue(np.array_equal(x5_zero, image))
        s100_zero = crop_and_rotate_image(image, zero, target="s100")
        self.assertEqual(s100_zero.shape, (9, 13, 3))
        self.assertEqual(int(s100_zero.sum()), 351)
        s100_collinear = crop_and_rotate_image(image, collinear, target="s100")
        # A collinear (zero-area) box has no designed crop semantics: the
        # S-side source (rdk_s utils/py_utils/postprocess.py
        # crop_and_rotate_image) runs the same minAreaRect -> warp ->
        # rotate chain with no degenerate branch, so the degenerate warp's
        # orientation follows the cv2 build's angle sign for zero-area rects
        # (the Windows build recorded on 2026-09-17 produced the rotated
        # (13, 9, 3); macOS cv2 4.14 produces (9, 13, 3)).  Assert the
        # cross-build invariants instead of one build's accident; the board
        # build's degenerate behavior is covered by board smoke.
        self.assertIn(s100_collinear.shape, ((13, 9, 3), (9, 13, 3)))
        self.assertTrue(
            np.array_equal(
                s100_collinear,
                crop_and_rotate_image(image, collinear, target="s100"),
            )
        )

    def test_x5_and_s100_keep_source_multi_polygon_conversion_order(self):
        from samples.vision.paddle_ocr.runtime.python.geometry import dilate_contours

        class FakeOffset:
            def AddPath(self, *args):
                del args

            def Execute(self, distance):
                del distance
                return [
                    [[0, 0], [2, 0], [2, 2]],
                    [[4, 4], [5, 4], [5, 5], [4, 5]],
                ]

        fake_pyclipper = SimpleNamespace(
            PyclipperOffset=FakeOffset,
            JT_ROUND=1,
            ET_CLOSEDPOLYGON=2,
        )
        contour = np.array(
            [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], dtype=np.int32
        )
        with patch.dict(sys.modules, {"pyclipper": fake_pyclipper}):
            self.assertEqual(
                dilate_contours((contour,), target="s100"), ()
            )
            with self.assertRaises(ValueError):
                dilate_contours((contour,), target="x5")


if __name__ == "__main__":
    unittest.main()
