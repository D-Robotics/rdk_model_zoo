import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


WEB_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_catalog_candidate_guard", SCRIPT)
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)


class CandidateReleaseGuardTest(unittest.TestCase):
    def test_candidate_batch_is_outside_active_catalog_scan(self):
        catalog, _ = build_catalog.build_catalog()
        parsed = yaml.safe_load(catalog)
        self.assertEqual(parsed["source"], "model_zoo_web/data")
        self.assertFalse(any("candidate-staging" in model["source_file"] for model in parsed["models"]))
        self.assertTrue(all(model["source_file"].startswith("model_zoo_web/data/")
                            for model in parsed["models"]))
        self.assertTrue(all(platform["status"] == "released"
                            for model in parsed["models"]
                            for variant in model["variants"]
                            for platform in variant["platforms"]))

    def test_candidate_status_is_rejected_by_the_active_release_validator(self):
        source_path = WEB_ROOT / "data" / "vision" / "ultralytics_yolo" / "yolo26" / "detect.yaml"
        record = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        candidate_record = copy.deepcopy(record)
        candidate_record["variants"][0]["platforms"][0]["status"] = "candidate"
        with self.assertRaisesRegex(build_catalog.CatalogError, r"status must be released"):
            build_catalog.validate_record(candidate_record, source_path)


if __name__ == "__main__":
    unittest.main()
