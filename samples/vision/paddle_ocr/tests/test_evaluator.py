"""Host tests for the file-based OCR evaluator."""

from __future__ import annotations

import unittest


class EvaluatorTests(unittest.TestCase):
    def test_matching_is_gt_ordered_and_uses_best_unused_prediction(self):
        from samples.vision.paddle_ocr.evaluator.evaluate import evaluate_records

        report = evaluate_records(
            [
                {
                    "image": "second",
                    "boxes": [
                        [[0, 0], [10, 0], [10, 10], [0, 10]],
                        [[20, 0], [30, 0], [30, 10], [20, 10]],
                    ],
                    "texts": ["左", "右"],
                },
                {
                    "image": "first",
                    "boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                    "texts": ["one"],
                },
            ],
            [
                {
                    "image": "first",
                    "boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                    "texts": ["one"],
                },
                {
                    "image": "second",
                    "boxes": [
                        [[20, 0], [30, 0], [30, 10], [20, 10]],
                        [[1, 0], [10, 0], [10, 10], [1, 10]],
                    ],
                    "texts": ["右", "左"],
                },
            ],
        )
        self.assertEqual([item["image"] for item in report["per_image"]], ["second", "first"])
        matches = report["per_image"][0]["matches"]
        self.assertEqual(
            [(item["ground_truth_index"], item["prediction_index"]) for item in matches],
            [(0, 1), (1, 0)],
        )
        self.assertEqual(report["matched_regions"], 3)
        self.assertEqual(report["recognition"]["exact_matches"], 3)

    def test_empty_ground_truth_and_prediction_is_valid_but_recognition_not_run(self):
        from samples.vision.paddle_ocr.evaluator.evaluate import evaluate_records

        report = evaluate_records(
            [{"image": "empty", "boxes": [], "texts": []}],
            [{"image": "empty", "boxes": [], "texts": []}],
        )
        self.assertEqual(report["ground_truth_regions"], 0)
        self.assertEqual(report["predicted_regions"], 0)
        self.assertEqual(report["matched_regions"], 0)
        self.assertEqual(report["recognition"]["status"], "not_run")
        self.assertEqual(report["recognition"]["exact_rate"], 0.0)

    def test_box_text_alignment_is_required(self):
        from samples.vision.paddle_ocr.evaluator.evaluate import (
            EvaluationError,
            evaluate_records,
        )

        with self.assertRaises(EvaluationError):
            evaluate_records(
                [{"image": "bad", "boxes": [[[0, 0], [1, 1]]], "texts": []}],
                [{"image": "bad", "boxes": [], "texts": []}],
            )

    def test_zero_threshold_still_requires_positive_overlap(self):
        from samples.vision.paddle_ocr.evaluator.evaluate import evaluate_records

        report = evaluate_records(
            [{"image": "disjoint", "boxes": [[[0, 0], [1, 1]]], "texts": ["gt"]}],
            [{"image": "disjoint", "boxes": [[[3, 3], [4, 4]]], "texts": ["pred"]}],
            iou_threshold=0.0,
        )
        self.assertEqual(report["matched_regions"], 0)
        self.assertEqual(report["recognition"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
