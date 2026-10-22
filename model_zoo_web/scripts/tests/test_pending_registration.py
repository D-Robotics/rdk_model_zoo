import importlib.util
import os
import unittest
from unittest import mock
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("pending_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)

CLIP = catalog.DATA_ROOT / "vision/clip/clip/similarity.yaml"


class PendingRegistrationTest(unittest.TestCase):
    def test_pending_record_uses_its_revision_without_a_live_sample_ref(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        with mock.patch.dict(os.environ, {catalog.SAMPLE_REF_ENV: "missing-live-branch"}):
            normalized = catalog.validate_record(record, CLIP)
        self.assertEqual(normalized["sample_revision"], record["sample_revision"])

    def test_upstream_repositories_reject_weight_downloads_and_duplicate_urls(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        source = "samples/vision/clip/README.md"
        for url in ["https://github.com/example/model/releases/download/v1/model.pt",
                    "https://github.com/example/model/model.pth"]:
            with self.subTest(url=url):
                record["upstream"] = [{"kind": "repository", "name": "example/model", "url": url, "source": source}]
                with self.assertRaisesRegex(catalog.CatalogError, "repository.*download"):
                    catalog.validate_record(record, CLIP)
        yolov5 = yaml.safe_load((catalog.DATA_ROOT / "vision/yolov5/yolov5/detect.yaml").read_text())
        record["upstream"] = [yolov5["upstream"][0]] * 2
        with self.assertRaisesRegex(catalog.CatalogError, "duplicate upstream"):
            catalog.validate_record(record, CLIP)

    def test_verified_identity_record_has_no_measurements(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        self.assertEqual(record["registration"], "pending")
        self.assertNotIn("variants", record)
        normalized = catalog.validate_record(record, CLIP)
        self.assertEqual(normalized["variants"], [])
        self.assertTrue(normalized["supported_platforms"])
        self.assertEqual(normalized["sample_revision"], "7a97c1780b0cfa131ff0007d662e255e5e41c12e")

    def test_sourced_benchmark_must_appear_in_the_cited_file(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        record["sourced_benchmarks"] = [{
            "id": "clip",
            "platform": "x5",
            "metric": "latency",
            "value": 12345.67,
            "unit": "ms",
            "source": "docs/release/x5/models.yaml",
        }]
        with self.assertRaisesRegex(catalog.CatalogError, "is not in the source file"):
            catalog.validate_record(record, CLIP)

    def test_pending_record_rejects_variants(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        record["variants"] = [{"size": "base"}]
        with self.assertRaisesRegex(catalog.CatalogError, "must not include variants"):
            catalog.validate_record(record, CLIP)

    def test_catalog_keeps_released_rows_and_counts_pending_rows(self):
        with mock.patch.dict(os.environ, {catalog.SAMPLE_REF_ENV: "7a97c1780b0cfa131ff0007d662e255e5e41c12e"}):
            payload, _ = catalog.build_catalog()
        built = yaml.safe_load(payload)
        pending = [model for model in built["models"] if model.get("registration") == "pending"]
        released = [model for model in built["models"] if model.get("registration") != "pending"]
        self.assertGreaterEqual(len(pending), 53)
        self.assertGreaterEqual(len(released), 18)
        self.assertEqual(built["pending_count"], len(pending))
        self.assertTrue(all(
            platform["status"] == "released"
            for model in released
            for variant in model["variants"]
            for platform in variant["platforms"]
        ))
        self.assertTrue(all(not model["variants"] for model in pending))


if __name__ == "__main__":
    unittest.main()
