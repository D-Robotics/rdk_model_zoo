import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


WEB_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_catalog_candidate_guard", SCRIPT)
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)


class CandidateReleaseGuardTest(unittest.TestCase):
    def test_candidate_batch_is_outside_active_catalog_scan(self):
        # This test is about which YAML files are scanned, not about sample sources, which the Web
        # branch does not have to carry (see MODEL_ZOO_SAMPLE_REF in build_catalog.py).
        with mock.patch.object(build_catalog, "validate_sample_path", lambda value, label, revision=None: value):
            catalog, _ = build_catalog.build_catalog()
        parsed = yaml.safe_load(catalog)
        self.assertEqual(parsed["source"], "model_zoo_web/data")
        self.assertFalse(any("candidate-staging" in model["source_file"] for model in parsed["models"]))
        self.assertTrue(all(model["source_file"].startswith("model_zoo_web/data/")
                            for model in parsed["models"]))
        self.assertTrue(all(platform["status"] in {"released", "converted", "demo", "official"}
                            for model in parsed["models"]
                            for variant in model["variants"]
                            for platform in variant["platforms"]))
        for model in parsed["models"]:
            for variant in model["variants"]:
                for platform in variant["platforms"]:
                    if platform["status"] in {"demo", "official"}:
                        if model["sample_path"] == "samples/llm/oellm":
                            self.assertEqual(platform["status"], "official")
                            self.assertIn((model["domain"], model["task"]),
                                          {("llm", "text-gen"), ("llm", "vlm"), ("speech", "asr")})
                            self.assertEqual(platform["platform"], "s600")
                        else:
                            self.assertEqual((model["domain"], model["task"], platform["platform"]),
                                             ("llm", "text-generation", "s600"))
                        for field in ("artifact", "accuracy", "performance", "reports"):
                            self.assertIsNone(platform[field])

    def test_candidate_status_is_rejected_by_the_active_release_validator(self):
        source_path = WEB_ROOT / "data" / "vision" / "ultralytics_yolo" / "yolo26" / "detect.yaml"
        record = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        candidate_record = copy.deepcopy(record)
        candidate_record["variants"][0]["platforms"][0]["status"] = "candidate"
        with self.assertRaisesRegex(build_catalog.CatalogError, r"status must be released"):
            build_catalog.validate_record(candidate_record, source_path)

    def test_public_catalog_rejects_oss_downloads_for_released_and_converted_models(self):
        source_path = WEB_ROOT / "data/vision/ultralytics_yolo/yolo26/detect.yaml"
        record = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        record["variants"] = [record["variants"][0]]
        variant = record["variants"][0]
        variant["platforms"] = [variant["platforms"][0]]
        platform = variant["platforms"][0]
        filename = platform["artifact"]["url"].rsplit("/", 1)[-1]
        platform["artifact"]["url"] = (
            f"https://{build_catalog.OSS_HOST}/models/{record['id']}/"
            f"{variant['size']}/{platform['platform']}/{filename}"
        )
        # Exercise public publication after normalization, independently of the
        # legacy package validator that also reads historical OSS receipts.
        with tempfile.TemporaryDirectory(dir=WEB_ROOT) as directory:
            Path(directory, "model.yaml").write_text("{}", encoding="utf-8")
            with mock.patch.object(build_catalog, "DATA_ROOT", Path(directory)), \
                    mock.patch.object(build_catalog, "validate_record", return_value=record):
                for status in ("released", "converted"):
                    platform["status"] = status
                    with self.subTest(status=status), self.assertRaisesRegex(
                        build_catalog.CatalogError, "public model downloads must use archive"
                    ):
                        build_catalog.build_catalog()


if __name__ == "__main__":
    unittest.main()
