import copy
import importlib.util
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

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

    def test_native_workers_have_no_opencv_threads_and_match_stream_count(self):
        evidence = copy.deepcopy(self.platform["performance"]["end_to_end"])
        if isinstance(evidence, list):
            evidence = evidence[0]
        evidence.pop("opencv_threads", None)
        evidence.update(cpu_backend="native", cpu_thread_policy="fixed",
                        cpu_worker_threads=evidence["pipeline_streams"])
        catalog.validate_end_to_end_record(evidence, "native")
        for change in ({"cpu_worker_threads": evidence["pipeline_streams"] + 1},
                       {"opencv_threads": 1}, {"cpu_backend": "unknown"},
                       {"cpu_thread_policy": "all_online"}):
            with self.subTest(change=change), self.assertRaises(catalog.CatalogError):
                catalog.validate_end_to_end_record({**evidence, **change}, "native")

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

    def test_archive_uses_the_same_model_identity_beneath_its_public_root(self):
        artifact = copy.deepcopy(self.platform["artifact"])
        old_base = artifact["url"].rsplit("/", 1)[0]
        new_base = (
            "https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/"
            f"rdk_{self.platform['platform']}/ultralytics_yolo"
        )
        for field in ("url", "release_manifest_url", "checksums_url"):
            artifact[field] = artifact[field].replace(old_base, new_base)
        self.validate_artifact(artifact)
        for bad_url in (
            artifact["url"].replace("https:", "http:"),
            artifact["url"] + "?signature=temporary",
            artifact["url"] + "#fragment",
            artifact["url"].replace("archive.d-robotics.cc", "archive.d-robotics.cc.example.org"),
            artifact["url"].replace("/downloads/rdk_model_zoo_web/", "/downloads/rdk_model_zoo/"),
            artifact["url"].replace("/ultralytics_yolo/", "/another-sample/"),
            artifact["url"].replace(f"/rdk_{self.platform['platform']}/", "/rdk_unknown/"),
            artifact["url"].replace(new_base, "https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/models/ultralytics_yolo/yolo26/detect/n/x5"),
        ):
            with self.subTest(url=bad_url), self.assertRaises(catalog.CatalogError):
                self.validate_artifact({**artifact, "url": bad_url})
        with self.assertRaises(catalog.CatalogError):
            self.validate_artifact({**artifact, "checksums_url": new_base + "/SHA256SUMS"})

    def test_companions_use_the_platform_sample_and_model_specific_assets_directory(self):
        artifact = copy.deepcopy(self.platform["artifact"])
        original_model_url = artifact["url"]
        filename = original_model_url.rsplit("/", 1)[1]
        base = (
            "https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/assets/"
            f"rdk_{self.platform['platform']}/ultralytics_yolo/{Path(filename).stem}"
        )
        artifact.update(release_manifest_url=base+"/release.json", checksums_url=base+"/SHA256SUMS")
        self.validate_artifact(artifact)
        self.assertEqual(artifact["url"], original_model_url)
        for invalid_base in (
            base.replace("/assets/", "/"),
            base.replace("/ultralytics_yolo/", "/resnet/"),
            base.replace(f"/rdk_{self.platform['platform']}/", "/rdk_unknown/"),
            base + "-other-model",
        ):
            with self.subTest(base=invalid_base), self.assertRaises(catalog.CatalogError):
                self.validate_artifact({**artifact, "release_manifest_url": invalid_base+"/release.json",
                                       "checksums_url": invalid_base+"/SHA256SUMS"})
        with self.assertRaises(catalog.CatalogError):
            self.validate_artifact({**artifact, "checksums_url": base+"-other/SHA256SUMS"})

    def test_reports_cannot_reference_another_models_companions(self):
        path = catalog.DATA_ROOT / "vision/osnet/osnet/reid.yaml"
        for field in ("oe_conversion_url", "oe_data_url"):
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            reports = record["variants"][0]["platforms"][0]["reports"]
            reports[field] = reports[field].replace("/osnet/", "/another-sample/")
            with self.subTest(field=field), self.assertRaises(catalog.CatalogError):
                catalog.validate_record(record, path)

    def test_osnet_release_matches_its_pinned_native_models(self):
        path = catalog.DATA_ROOT / "vision/osnet/osnet/reid.yaml"
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        with mock.patch.dict(os.environ, {catalog.SAMPLE_REF_ENV: "missing-live-branch"}):
            normalized = catalog.validate_record(record, path)
        revision = normalized["sample_revision"]
        native = json.loads(subprocess.check_output([
            "git", "-C", str(catalog.ROOT), "show",
            f"{revision}:samples/vision/osnet/model/artifacts.json",
        ], text=True))
        platforms = normalized["variants"][0]["platforms"]
        self.assertEqual({row["platform"] for row in platforms}, {"x5", "s100", "s100p", "s600"})
        self.assertEqual(normalized["license"]["name"], "Research use only")
        for row in platforms:
            artifact = native["artifacts"][row["provenance"]["runtime_variant"]]
            self.assertEqual(row["artifact"]["sha256"], artifact["sha256"])
            self.assertEqual(row["artifact"]["size_bytes"], artifact["size_bytes"])
            self.assertTrue(row["artifact"]["url"].startswith(
                "https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/"
                f"rdk_{row['platform']}/osnet/"
            ))


if __name__ == "__main__":
    unittest.main()
