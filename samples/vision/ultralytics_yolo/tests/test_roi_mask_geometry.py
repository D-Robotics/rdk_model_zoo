"""ROI mask geometry keeps the source sample's exact resize semantics."""

import unittest
import numpy as np
import cv2
from utils.py_utils.postprocess import (
    resize_masks_to_boxes,
)


class ROIGeometryTests(unittest.TestCase):
    def test_lanczos_overshoot_keeps_source_values(self):
        # The source returns the Lanczos result unchanged: overshoot samples
        # stay 2 instead of being renormalized to a 0/1 representation.
        signs = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.uint8)
        mask = (signs[:, None] == signs[None, :]).astype(np.uint8)
        raw = cv2.resize(mask, (17, 17), interpolation=cv2.INTER_LANCZOS4)
        actual = resize_masks_to_boxes(
            [mask], [[0, 0, 17, 17]], 20, 20, do_morph=False
        )[0]
        np.testing.assert_array_equal(actual, raw)
        if int(raw.max()) > 1:
            self.assertGreater(int(actual.max()), 1)

    def test_source_clamping_and_minimum_axis(self):
        # One-sided clamping with a minimum 1-pixel axis: a degenerate box
        # produces a 1-pixel mask, exactly like the original sample.
        boxes = np.array(
            [[1, 2, 8, 2], [3, 1, 3, 9], [-5, -5, -1, -1], [1, 2, 8, 6]], np.float32
        )
        masks = [np.ones((2, 2), np.uint8) for _ in boxes]
        actual = resize_masks_to_boxes(masks, boxes, 10, 10, do_morph=False)
        self.assertEqual([x.shape for x in actual], [(1, 7), (8, 1), (1, 1), (4, 7)])
        self.assertTrue(np.all(actual[2] == 1))
        self.assertTrue(np.all(actual[3] == 1))
        empty = resize_masks_to_boxes([np.zeros((0, 0), np.uint8)], boxes[-1:], 10, 10)
        self.assertEqual(empty[0].shape, (4, 7))
        self.assertFalse(empty[0].any())
