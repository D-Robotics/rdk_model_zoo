"""Compare the X5 broad phase with exhaustive, classwise rotated NMS."""
import unittest
from unittest.mock import patch

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python import obb


def exhaustive(boxes, scores, ids, threshold):
    keep = []
    for cid in np.unique(ids):
        indices = np.flatnonzero(np.asarray(ids) == cid)
        order = indices[np.argsort(np.asarray(scores, np.float32)[indices])[::-1]]
        while order.size:
            current = order[0]
            keep.append(int(current))
            order = np.asarray([i for i in order[1:]
                                if obb._rotated_iou(boxes[current], boxes[i]) < threshold],
                               dtype=np.int64)
    return keep


class X5RotatedNmsTests(unittest.TestCase):
    def check_case(self, boxes, scores, ids, threshold, expected=None):
        reference = exhaustive(boxes, scores, ids, threshold)
        if expected is not None:
            self.assertEqual(reference, expected)
        self.assertEqual(obb._rotated_nms(boxes, scores, ids, 0.25, threshold, "x5"),
                         reference)

    def test_dense_overlap(self):
        rng = np.random.default_rng(104)
        boxes = [(x, y, w, h, a) for x, y, w, h, a in
                 rng.uniform([-3, -3, 5, 3, -1], [3, 3, 12, 9, 1], size=(60, 5))]
        scores = rng.uniform(0.5, 1, len(boxes))
        ids = np.arange(len(boxes)) % 2
        for threshold in (0.1, 0.45, 0.9):
            with self.subTest(threshold=threshold):
                self.check_case(boxes, scores, ids, threshold)

    def test_edge_contact_gap_and_small_overlap(self):
        boxes = [(0, 0, 8, 4, 0), (8, 0, 8, 4, 0), (0, 4.01, 8, 4, 0),
                 (-7.99, 0, 8, 4, 0)]
        self.check_case(boxes, [1, .9, .8, .7], [0]*4, .0001, [0, 1, 2])

    def test_tied_scores_preserve_original_order_and_classes(self):
        boxes = [(i % 3 * 20, 0, 6, 3, 0.2) for i in range(24)]
        self.check_case(boxes, [.75]*24, [i % 2 for i in range(24)], .3)

    def test_zero_threshold_suppresses_even_disjoint_boxes_within_class(self):
        boxes = [(0, 0, 6, 3, 0), (50, 0, 6, 3, 0), (100, 0, 6, 3, 0)]
        self.check_case(boxes, [.8, .9, .7], [0, 0, 1], 0, [1, 2])

    def test_disjoint_bounds_skip_exact_intersection(self):
        boxes = [(0, 0, 6, 3, .5), (100, 50, 6, 3, -.5)]
        with patch.object(obb, "_rotated_iou", wraps=obb._rotated_iou) as iou:
            self.assertEqual(obb._rotated_nms(boxes, [.9, .8], [0, 0], .25, .3, "x5"),
                             [0, 1])
            iou.assert_not_called()
