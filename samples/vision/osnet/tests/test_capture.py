import json
import tempfile
import unittest
from pathlib import Path

from samples.vision.osnet.evaluator.main import dataset_sha256_manifest, require_matching_capture


class CaptureDatasetTests(unittest.TestCase):
    def _layout(self, root, payload):
        dataset = root / "Market1501"
        query = dataset / "query"
        query.mkdir(parents=True)
        image = query / "0001_c1s1_001.jpg"
        image.write_bytes(payload)
        ordered = [(image, 1, 1)]
        manifest = [{"name": image.name, "pid": 1, "camera": 1}]
        return dataset, image, ordered, manifest

    def test_accepts_matching_image_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, _, ordered, manifest = self._layout(root, b"original-bytes")
            current = dataset_sha256_manifest(dataset, ordered)
            capture = root / "capture"
            capture.mkdir()
            (capture / "manifest.json").write_text(json.dumps(manifest) + "\n")
            (capture / "dataset-sha256.json").write_text(json.dumps(current, indent=2) + "\n")
            require_matching_capture(capture, manifest, current)

    def test_rejects_changed_image_bytes_with_the_same_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, image, ordered, manifest = self._layout(root, b"original-bytes")
            saved = dataset_sha256_manifest(dataset, ordered)
            capture = root / "capture"
            capture.mkdir()
            (capture / "manifest.json").write_text(json.dumps(manifest) + "\n")
            (capture / "dataset-sha256.json").write_text(json.dumps(saved) + "\n")
            image.write_bytes(b"edited-bytes")
            current = dataset_sha256_manifest(dataset, ordered)
            with self.assertRaisesRegex(ValueError, "dataset-sha256 differs"):
                require_matching_capture(capture, manifest, current)

    def test_rejects_a_capture_without_dataset_sha256(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, _, ordered, manifest = self._layout(root, b"original-bytes")
            capture = root / "capture"
            capture.mkdir()
            (capture / "manifest.json").write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "missing dataset-sha256.json"):
                require_matching_capture(capture, manifest, dataset_sha256_manifest(dataset, ordered))


if __name__ == "__main__":
    unittest.main()
