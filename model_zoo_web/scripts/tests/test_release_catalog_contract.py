import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("release_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class ReleaseCatalogContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = catalog.DATA_ROOT / "vision/ultralytics_yolo/yolo26/detect.yaml"
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        cls.variant = record["variants"][0]
        cls.platform = cls.variant["platforms"][0]

    def test_measured_frequency_accepts_float_without_changing_the_value(self):
        evidence = copy.deepcopy(self.platform["performance"]["end_to_end"])
        if isinstance(evidence, list):
            evidence = evidence[0]
        evidence["cpu_frequency_mhz"] = 2100.0
        evidence["bpu_frequency_mhz"] = 996.5
        before = copy.deepcopy(evidence)
        catalog.validate_end_to_end_record(evidence, "fixture.end_to_end")
        self.assertEqual(evidence, before)
        for field in ("cpu_frequency_mhz", "bpu_frequency_mhz"):
            for invalid in (0, -1, True, "2100", float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    bad = copy.deepcopy(evidence)
                    bad[field] = invalid
                    with self.assertRaises(catalog.CatalogError):
                        catalog.validate_end_to_end_record(bad, "fixture.end_to_end")

    def validate_artifact(self, artifact, build_id=None):
        catalog.validate_artifact(
            artifact, source="ultralytics_yolo", family="yolo26", task="detect",
            size=self.variant["size"], platform=self.platform["platform"],
            label="fixture.artifact", build_id=build_id,
        )

    def test_rebuild_url_must_match_the_recorded_build_id(self):
        artifact = copy.deepcopy(self.platform["artifact"])
        self.validate_artifact(artifact)
        base, filename = artifact["url"].rsplit("/", 1)
        build_id = "20261001T0000Z-regression-r01"
        artifact["url"] = f"{base}/rebuilds/{build_id}/{filename}"
        artifact["release_manifest_url"] = f"{base}/rebuilds/{build_id}/release.json"
        artifact["checksums_url"] = f"{base}/rebuilds/{build_id}/SHA256SUMS"
        self.validate_artifact(artifact, build_id)
        for mismatched_id in (None, "another-build", "../unsafe", "bad/id"):
            with self.subTest(build_id=mismatched_id):
                with self.assertRaises(catalog.CatalogError):
                    self.validate_artifact(artifact, mismatched_id)


if __name__ == "__main__":
    unittest.main()
