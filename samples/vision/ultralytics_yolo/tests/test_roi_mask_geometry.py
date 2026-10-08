"""Degenerate boxes retain exact zero-sized ROI dimensions, never fake pixels."""

import unittest
import numpy as np
import cv2
from utils.py_utils.postprocess import (
    resize_masks_to_boxes,
)


class ROIGeometryTests(unittest.TestCase):
    def test_lanczos_overshoot_remains_binary(self):
        signs = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.uint8)
        mask = (signs[:, None] == signs[None, :]).astype(np.uint8)
        raw = cv2.resize(mask, (17, 17), interpolation=cv2.INTER_LANCZOS4)
        actual = resize_masks_to_boxes(
            [mask], [[0, 0, 17, 17]], 20, 20, do_morph=False
        )[0]
        self.assertLessEqual(int(actual.max()), 1)
        np.testing.assert_array_equal(actual, (raw > 0).astype(np.uint8))

    def test_empty_axes_match_clipped_box_extent(self):
        boxes = np.array(
            [[1, 2, 8, 2], [3, 1, 3, 9], [-5, -5, -1, -1], [1, 2, 8, 6]], np.float32
        )
        masks = [np.ones((2, 2), np.uint8) for _ in boxes]
        actual = resize_masks_to_boxes(masks, boxes, 10, 10, do_morph=False)
        self.assertEqual([x.shape for x in actual], [(0, 7), (8, 0), (0, 0), (4, 7)])
        self.assertTrue(np.all(actual[3] == 1))
        actual = resize_masks_to_boxes([np.zeros((0, 0), np.uint8)], boxes[-1:], 10, 10)
        self.assertEqual(actual[0].shape, (4, 7))
        self.assertFalse(actual[0].any())
