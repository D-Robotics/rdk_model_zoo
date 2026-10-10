"""Contract tests for classification geometry, normalization, and evidence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.py_utils.classification_host import (
    calibration_input, float_input, load_dataset, prepare_rgb, sha256_file,
    top5_ids, validate_contract, write_json,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads((ROOT / "utils/tools/mobilenet/checkpoints.json").read_text())["models"]["v4-small"]["contract"]


class ClassificationHostTests(unittest.TestCase):
    """Exercise failures that would silently corrupt classification accuracy."""

    def test_normalization_and_platform_domains(self):
        rgb = np.empty((224, 224, 3), dtype=np.uint8)
        rgb[:] = [255, 128, 0]
        actual = float_input(rgb, CONTRACT)
        expected = (np.array([1, 128 / 255, 0]) - CONTRACT["mean"]) / CONTRACT["std"]
        np.testing.assert_allclose(actual[0, :, 0, 0], expected, atol=1e-6)
        x5 = calibration_input(rgb, CONTRACT, "x5")
        np.testing.assert_array_equal(x5[0, :, 0, 0], [255, 128, 0])
        for target in ("s100", "s100p", "s600"):
            np.testing.assert_array_equal(calibration_input(rgb, CONTRACT, target), actual)
        with self.assertRaises(ValueError):
            calibration_input(rgb, CONTRACT, "unknown")

    def test_crop_geometry_not_stretch_or_letterbox(self):
        pixels = np.zeros((256, 512, 3), dtype=np.uint8)
        pixels[:, 144:368] = [255, 0, 0]
        actual = prepare_rgb(Image.fromarray(pixels), CONTRACT)
        self.assertEqual(actual.shape, (224, 224, 3))
        np.testing.assert_array_equal(actual, np.broadcast_to([255, 0, 0], actual.shape))

    def test_grayscale_and_rgba_are_rgb(self):
        for mode in ("L", "RGBA"):
            result = prepare_rgb(Image.new(mode, (331, 177)), CONTRACT)
            self.assertEqual(result.shape, (224, 224, 3))
            self.assertEqual(result.dtype, np.uint8)

    def test_reject_invalid_contract(self):
        for key, value in (("size", 225), ("crop_pct", 0), ("crop_pct", float("nan")),
                           ("std", [1, 0, 1]), ("mean", [1, 2]), ("color", "BGR"),
                           ("output", "probabilities"), ("batch", 8)):
            candidate = copy.deepcopy(CONTRACT)
            candidate[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_contract(candidate)

    def test_stable_ranking_and_shape_rejection(self):
        scores = np.zeros((1, 1000), dtype=np.float32)
        scores[0, 999] = 10
        self.assertEqual(top5_ids(scores), [999, 0, 1, 2, 3])
        for invalid in (np.zeros((2, 500)), np.zeros(999), np.full(1000, np.nan)):
            with self.assertRaises(ValueError):
                top5_ids(invalid)

    def test_dataset_numeric_labels_hashes_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "a.png"
            Image.new("RGB", (20, 30)).save(image)
            record = {"path": "a.png", "sha256": sha256_file(image), "label_id": 10}
            value = {"count": 1, "class_count": 1000, "images": [record]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(value))
            _, records = load_dataset(manifest, root, expected_count=1)
            self.assertEqual(records[0][1]["label_id"], 10)
            with self.assertRaises(ValueError):
                load_dataset(manifest, root, expected_count=2)
            for key, invalid in (("label_id", "10"), ("label_id", 1000),
                                 ("sha256", "0" * 64), ("path", "../a.png")):
                changed = copy.deepcopy(value)
                changed["images"][0][key] = invalid
                manifest.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_dataset(manifest, root, expected_count=1)
            value["count"] = 2
            value["images"] = [record, record]
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_dataset(manifest, root, expected_count=2)

    def test_receipt_never_overwrites_or_serializes_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                write_json(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text())["status"], "first")
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "bad.json", {"metric": float("nan")})


if __name__ == "__main__":
    unittest.main()
