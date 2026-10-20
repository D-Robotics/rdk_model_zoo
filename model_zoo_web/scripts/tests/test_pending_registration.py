import importlib.util
import os
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("pending_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)

os.environ[catalog.SAMPLE_REF_ENV] = "refs/remotes/origin/develop"
CLIP = catalog.DATA_ROOT / "vision/clip/clip/similarity.yaml"


class PendingRegistrationTest(unittest.TestCase):
    def test_verified_identity_record_has_no_measurements(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        self.assertEqual(record["registration"], "pending")
        self.assertNotIn("variants", record)
        normalized = catalog.validate_record(record, CLIP)
        self.assertEqual(normalized["variants"], [])
        self.assertTrue(normalized["supported_platforms"])
        self.assertEqual(normalized["sample_revision"], "7a97c1780b0cfa131ff0007d662e255e5e41c12e")

    def test_pending_record_rejects_variants(self):
        record = yaml.safe_load(CLIP.read_text(encoding="utf-8"))
        record["variants"] = [{"size": "base"}]
        with self.assertRaisesRegex(catalog.CatalogError, "must not include variants"):
            catalog.validate_record(record, CLIP)

    def test_catalog_keeps_released_rows_and_counts_pending_rows(self):
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
