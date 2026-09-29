"""Strict dataset identity, geometry and real COCO scoring regression fixtures."""

import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256
from samples.vision.yoloe.runtime.python.pipeline_io import Result
from samples.vision.yoloe.evaluator.dataset import load_dataset, load_category_map
from samples.vision.yoloe.evaluator.results import (
    serialize_predictions,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[5]
NAMES = (ROOT / "samples/vision/yoloe/test_data/classes.names").read_text().splitlines()


class EvaluationTests(unittest.TestCase):
    def test_help_without_packages_and_documented_parser_options(self):
        from samples.vision.yoloe.evaluator.evaluate import build_parser

        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "samples/vision/yoloe/evaluator/evaluate.py",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        options = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        for name in ("README.md", "README_cn.md"):
            text = (ROOT / "samples/vision/yoloe/evaluator" / name).read_text()
            for option in options:
                self.assertIn(option, text)

    def test_model_identity_and_board_thread_option_fail_before_sdk(self):
        from samples.vision.yoloe.evaluator.backends import create_predictor
        from samples.vision.yoloe.runtime.python.yoloe import Config
        from hashlib import sha256

        path = self.root / "model.hbm"
        path.write_bytes(b"not-a-real-model")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            create_predictor("board", "s100", "26n", path, "0" * 64, Config())
        with self.assertRaisesRegex(ValueError, "ONNX only"):
            create_predictor(
                "board",
                "s100",
                "26n",
                path,
                sha256(path.read_bytes()).hexdigest(),
                Config(),
                threads=4,
            )

    def test_annotation_size_mismatch_is_not_skipped(self):
        from samples.vision.yoloe.evaluator.engine import evaluate_dataset

        cv2.imwrite(str(self.root / "image.png"), np.zeros((7, 10, 3), np.uint8))
        dataset = load_dataset(self.ann, self.root)
        mapping = load_category_map(self.mapping, dataset.categories, NAMES)

        def forbidden(image):
            self.fail("Predictor must not run on a mismatched image.")

        fixture = SimpleNamespace(
            identity={"backend": "explicit-test-fixture"}, predict=forbidden
        )
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            evaluate_dataset(
                dataset,
                mapping,
                fixture,
                self.root / "bad-dimensions",
                predictions_only=True,
            )
        record = json.loads((self.root / "bad-dimensions/evaluation.json").read_text())
        self.assertEqual(record["processed_images"], 0)
        self.assertEqual(record["status"], "failed")

    def test_execution_preserves_partial_failure_and_rejects_overwrite(self):
        from samples.vision.yoloe.evaluator.engine import evaluate_dataset

        cv2.imwrite(str(self.root / "image.png"), np.zeros((8, 10, 3), np.uint8))
        self.annotation["images"].append(
            {"id": 8, "file_name": "missing.png", "height": 8, "width": 10}
        )
        self.ann.write_text(json.dumps(self.annotation))
        dataset = load_dataset(self.ann, self.root)
        mapping = load_category_map(self.mapping, dataset.categories, NAMES)
        result = Result(
            np.array([[1, 2, 4, 5]], np.float32),
            np.array([0.9], np.float32),
            np.array([2163]),
            [np.ones((3, 3), np.uint8)],
            "roi",
        )
        fixture = SimpleNamespace(
            identity={"backend": "explicit-test-fixture"}, predict=lambda image: result
        )
        out = self.root / "out"
        with self.assertRaises(FileNotFoundError):
            evaluate_dataset(dataset, mapping, fixture, out, predictions_only=True)
        record = json.loads((out / "evaluation.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["processed_images"], 1)
        self.assertIsNone(record["metrics"])
        self.assertEqual(
            len(json.loads((out / "bbox-predictions.partial.json").read_text())), 1
        )
        with self.assertRaises(FileExistsError):
            evaluate_dataset(dataset, mapping, fixture, out, predictions_only=True)

    def test_prediction_only_and_ground_truth_free_scoring_are_distinct(self):
        from samples.vision.yoloe.evaluator.engine import evaluate_dataset

        cv2.imwrite(str(self.root / "image.png"), np.zeros((8, 10, 3), np.uint8))
        dataset = load_dataset(self.ann, self.root)
        mapping = load_category_map(self.mapping, dataset.categories, NAMES)
        empty = Result(
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.int64),
            [],
            "roi",
        )
        fixture = SimpleNamespace(
            identity={"backend": "explicit-test-fixture"}, predict=lambda image: empty
        )
        result = evaluate_dataset(
            dataset, mapping, fixture, self.root / "predictions", predictions_only=True
        )
        self.assertEqual(result["status"], "predictions-only")
        self.assertIsNone(result["metrics"])
        with self.assertRaisesRegex(ValueError, "no non-crowd"):
            evaluate_dataset(dataset, mapping, fixture, self.root / "metrics")
        self.assertEqual(
            json.loads((self.root / "metrics/evaluation.json").read_text())["status"],
            "failed",
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation = {
            "images": [{"id": 7, "file_name": "image.png", "height": 8, "width": 10}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [],
        }
        self.ann = self.root / "annotations.json"
        self.ann.write_text(json.dumps(self.annotation))
        self.mapping = self.root / "mapping.json"
        self.mapping.write_text(
            json.dumps(
                {
                    "vocabulary_sha256": LABELS_SHA256,
                    "mapping": [
                        {
                            "pf_id": 2163,
                            "pf_name": "person",
                            "category_id": 1,
                            "category_name": "person",
                        }
                    ],
                }
            )
        )

    def test_mapping_requires_vocabulary_names_and_complete_dataset_coverage(self):
        data = load_dataset(self.ann, self.root, limit=0)
        mapping = load_category_map(self.mapping, data.categories, NAMES)
        self.assertEqual(mapping.ids, {2163: 1})
        for entry in (
            {
                "pf_id": 2163,
                "pf_name": "chair",
                "category_id": 1,
                "category_name": "person",
            },
            {
                "pf_id": True,
                "pf_name": "person",
                "category_id": 1,
                "category_name": "person",
            },
            {
                "pf_id": 2163,
                "pf_name": "person",
                "category_id": 2,
                "category_name": "person",
            },
        ):
            self.mapping.write_text(
                json.dumps({"vocabulary_sha256": LABELS_SHA256, "mapping": [entry]})
            )
            with self.assertRaises(ValueError):
                load_category_map(self.mapping, data.categories, NAMES)
        self.mapping.write_text(
            json.dumps({"vocabulary_sha256": LABELS_SHA256, "mapping": []})
        )
        with self.assertRaises(ValueError):
            load_category_map(self.mapping, data.categories, NAMES)

    def test_dataset_rejects_duplicate_ids_path_escape_and_invalid_limits(self):
        for changes in (
            {"images": self.annotation["images"] * 2},
            {"images": [dict(self.annotation["images"][0], file_name="../escape.jpg")]},
            {"images": []},
        ):
            self.ann.write_text(json.dumps(self.annotation | changes))
            with self.assertRaises(ValueError):
                load_dataset(self.ann, self.root, limit=0)
        self.ann.write_text(json.dumps(self.annotation))
        with self.assertRaises(ValueError):
            load_dataset(self.ann, self.root, limit=-1)
        self.ann.write_text('{"images": [], "images": []}')
        with self.assertRaises(ValueError):
            load_dataset(self.ann, self.root, limit=0)

    def test_full_and_roi_masks_encode_identical_pixels_without_resizing(self):
        from pycocotools import mask as mask_utils

        full = np.zeros((8, 10), np.uint8)
        full[2:5, 1:4] = 1
        box = np.array([[1.0, 2.0, 4.0, 5.0]], np.float32)
        common = (box, np.array([0.9], np.float32), np.array([2163], np.int64))
        result_full = Result(*common, np.array([full]), "full")
        result_roi = Result(*common, [np.ones((3, 3), np.uint8)], "roi")
        a = serialize_predictions(result_full, 7, (8, 10), {2163: 1})
        b = serialize_predictions(result_roi, 7, (8, 10), {2163: 1})
        self.assertEqual(a, b)
        self.assertEqual(a[0][0]["bbox"], [1.0, 2.0, 3.0, 3.0])
        self.assertNotIn("bbox", a[1][0])
        np.testing.assert_array_equal(mask_utils.decode(a[1][0]["segmentation"]), full)
        malformed = Result(*common, [np.ones((2, 2), np.uint8)], "roi")
        with self.assertRaises(ValueError):
            serialize_predictions(malformed, 7, (8, 10), {2163: 1})

    def test_unmapped_predictions_are_counted_and_invalid_scores_rejected(self):
        result = Result(
            np.array([[1, 2, 4, 5]], np.float32),
            np.array([0.9], np.float32),
            np.array([821]),
            [np.ones((3, 3), np.uint8)],
            "roi",
        )
        boxes, masks, dropped = serialize_predictions(result, 7, (8, 10), {2163: 1})
        self.assertEqual((boxes, masks, dropped), ([], [], 1))
        invalid = Result(
            result.boxes, np.array([np.nan]), result.class_ids, result.masks, "roi"
        )
        with self.assertRaises(ValueError):
            serialize_predictions(invalid, 7, (8, 10), {2163: 1})

    def test_real_coco_scoring_perfect_and_empty_predictions(self):
        from pycocotools import mask as mask_utils

        binary = np.zeros((8, 10), np.uint8)
        binary[2:5, 1:4] = 1
        encoded = mask_utils.encode(np.asfortranarray(binary))
        encoded["counts"] = encoded["counts"].decode()
        self.annotation["annotations"] = [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "bbox": [1, 2, 3, 3],
                "segmentation": encoded,
                "area": 9,
                "iscrowd": 0,
            }
        ]
        self.ann.write_text(json.dumps(self.annotation))
        boxes = [{"image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 3], "score": 0.9}]
        masks = [
            {"image_id": 7, "category_id": 1, "segmentation": encoded, "score": 0.9}
        ]
        scored = score_predictions(self.ann, boxes, masks, [7], [1])
        self.assertAlmostEqual(scored["bbox"]["AP"], 1.0)
        self.assertAlmostEqual(scored["segm"]["AP"], 1.0)
        # Exercise persisted engine metrics, not only the in-memory scorer.
        from samples.vision.yoloe.evaluator.engine import evaluate_dataset

        cv2.imwrite(str(self.root / "image.png"), np.zeros((8, 10, 3), np.uint8))
        dataset = load_dataset(self.ann, self.root)
        mapping = load_category_map(self.mapping, dataset.categories, NAMES)
        fixture = SimpleNamespace(
            identity={"backend": "explicit-test-fixture"},
            predict=lambda image: Result(
                np.array([[1, 2, 4, 5]], np.float32),
                np.array([0.9], np.float32),
                np.array([2163]),
                [np.ones((3, 3), np.uint8)],
                "roi",
            ),
        )
        out = self.root / "scored-engine"
        evaluate_dataset(dataset, mapping, fixture, out)
        persisted = json.loads((out / "evaluation.json").read_text())
        self.assertEqual(persisted["status"], "evaluated")
        self.assertEqual(persisted["metric_scope"], "all-annotation-images")
        self.assertAlmostEqual(persisted["metrics"]["segm"]["AP"], 1.0)
        self.assertEqual(persisted["metrics"]["bbox"]["parameters"]["image_ids"], [7])
        empty = score_predictions(self.ann, [], [], [7], [1])
        self.assertEqual(empty["bbox"]["AP"], 0.0)
        self.assertEqual(empty["segm"]["AP"], 0.0)
