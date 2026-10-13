import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_catalog_sample_profile", SCRIPT)
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)

LABEL = "fixture.sample_profile"
ENV = build_catalog.SAMPLE_REF_ENV


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True, text=True)


def sample_record(commit, **overrides):
    record = {
        "schema_version": 1,
        "profile": "sample",
        "id": "demo/demo/tracking",
        "domain": "vision",
        "source": "demo",
        "provider": "demo-upstream",
        "family": "demo",
        "task": "tracking",
        "name": "Demo Tracker",
        "description": {"zh": "演示跟踪器。", "en": "Demo tracker."},
        "license": {"name": "MIT", "url": "https://example.invalid/license"},
        "sample_path": "samples/vision/demo",
        "sample_commit": commit,
        "pipeline": "cpu_tracker",
        "pipeline_note": {"zh": "纯 CPU。", "en": "CPU only."},
        "variants": [{
            "size": "default",
            "platforms": [{
                "platform": "x5",
                "status": "available",
                "runtimes": ["python"],
                "model_availability": {
                    "status": "no-neural-artifact",
                    "note": {"zh": "无神经网络模型。", "en": "No neural model."},
                },
                "verification": {"zh": "板端已验证。", "en": "Verified on board."},
                "evidence": [{
                    "label": {"zh": "证据摘要", "en": "Evidence summary"},
                    "url": "sample-evidence/demo.json",
                }],
                "quickstart": [{
                    "title": {"zh": "获取源码", "en": "Get the source"},
                    "command": "git clone https://example.invalid/demo.git",
                }],
            }],
        }],
        "missing": [{
            "field": "mot-accuracy",
            "status": "missing",
            "title": {"zh": "跟踪精度", "en": "Tracking accuracy"},
            "note": {"zh": "待测量。", "en": "Pending measurement."},
        }],
    }
    record.update(copy.deepcopy(overrides))
    return record


class SampleProfileTest(unittest.TestCase):
    """Sample profiles render beside released models but never pass as one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        git(self.root, "init", "-q", "-b", "develop")
        sample = self.root / "samples/vision/demo"
        sample.mkdir(parents=True)
        (sample / "README.md").write_text("demo\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "develop sample")
        self.commit = git_output(self.root, "rev-parse", "HEAD").strip()
        self.data_root = self.root / "model_zoo_web/data"
        self.source_path = self.data_root / "vision/demo/demo/tracking.yaml"
        self.source_path.parent.mkdir(parents=True)
        self.web_root = self.root / "model_zoo_web"
        evidence_dir = self.web_root / "release/sample-evidence"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "demo.json").write_text("{}\n", encoding="utf-8")
        # ROOT, DATA_ROOT and WEB_ROOT are patched together so the validator
        # resolves the record path and the shipped evidence files against the
        # fixture repository.
        for name, value in (
            ("ROOT", self.root), ("DATA_ROOT", self.data_root), ("WEB_ROOT", self.web_root),
        ):
            patcher = mock.patch.object(build_catalog, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(ENV, None)

    def validate(self, record):
        return build_catalog.validate_sample_record(record, self.source_path)

    def platform(self, record):
        return record["variants"][0]["platforms"][0]

    def pair_artifacts(self, parta_format="bin", partb_format="bin"):
        return [
            {"role": "parta", "filename": "demo_parta.bin", "format": parta_format,
             "url": "https://archive.d-robotics.cc/demo_parta.bin",
             "sha256_provenance": "publisher", "sha256": "a" * 64, "size_bytes": 16},
            {"role": "partb", "filename": "demo_partb.bin", "format": partb_format,
             "url": "https://archive.d-robotics.cc/demo_partb.bin",
             "sha256_provenance": "publisher", "sha256": "b" * 64, "size_bytes": 16},
        ]

    def publish(self, record, artifacts):
        platform = self.platform(record)
        platform["model_availability"]["status"] = "published-download"
        platform["artifacts"] = artifacts
        return platform

    def commit_descriptor(self, descriptor):
        path = self.root / "samples/vision/demo/model/demo.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(descriptor), encoding="utf-8")
        # The release manifests the sample downloader fetches from; the URL
        # binding reads them from the same commit.
        entry = {
            "id": "demo",
            "sample_path": "samples/vision/demo",
            "assets": [
                {"filename": "demo_parta.bin", "format": "bin",
                 "url": "https://archive.d-robotics.cc/demo_parta.bin", "sha256": "a" * 64},
                {"filename": "demo_partb.bin", "format": "bin",
                 "url": "https://archive.d-robotics.cc/demo_partb.bin", "sha256": "b" * 64},
            ],
        }
        document = {"schema_version": 1, "models": [entry]}
        for manifest_rel in build_catalog.SAMPLE_RELEASE_MANIFESTS.values():
            manifest_path = self.root / manifest_rel
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "artifact descriptor")
        return git_output(self.root, "rev-parse", "HEAD").strip()

    def test_valid_record_normalizes_with_source_file(self):
        normalized = self.validate(sample_record(self.commit))
        self.assertEqual(normalized["source_file"], "model_zoo_web/data/vision/demo/demo/tracking.yaml")

    def test_sample_commit_must_resolve_to_a_real_tree(self):
        nonsense = "0" * 40  # matches the shape regex; resolves to nothing
        with self.assertRaisesRegex(build_catalog.CatalogError, "is not a tree"):
            self.validate(sample_record(nonsense))
        with self.assertRaisesRegex(build_catalog.CatalogError, "is not a tree"):
            # A real commit still fails when the sample path is absent from it.
            build_catalog.verify_commit_tree(self.commit, "samples/vision/absent", LABEL)
        build_catalog.verify_commit_tree(self.commit, "samples/vision/demo", LABEL)

    def test_released_model_fields_are_rejected(self):
        variant_bad = sample_record(self.commit)
        variant_bad["variants"][0]["performance"] = []
        with self.assertRaisesRegex(build_catalog.CatalogError, "must not carry released-model fields: performance"):
            self.validate(variant_bad)
        platform_bad = sample_record(self.commit)
        self.platform(platform_bad)["accuracy"] = []
        with self.assertRaisesRegex(build_catalog.CatalogError, "must not carry released-model fields: accuracy"):
            self.validate(platform_bad)

    def test_platform_target_and_status(self):
        x3 = sample_record(self.commit)
        self.platform(x3)["platform"] = "x3"
        with self.assertRaisesRegex(build_catalog.CatalogError, "x3 is archive-only"):
            self.validate(x3)
        released = sample_record(self.commit)
        self.platform(released)["status"] = "released"
        with self.assertRaisesRegex(build_catalog.CatalogError, "status must be available"):
            self.validate(released)

    def test_published_artifact_host_allowlist(self):
        record = sample_record(self.commit)
        self.platform(record)["model_availability"]["status"] = "published-download"
        self.platform(record)["artifacts"] = [{
            "role": "detector", "filename": "demo.bin", "format": "bin",
            "url": "https://example.invalid/demo.bin", "sha256_provenance": "publisher",
            "sha256": "a" * 64, "size_bytes": 16,
        }]
        with self.assertRaisesRegex(build_catalog.CatalogError, "direct stable HTTPS URL"):
            self.validate(record)
        self.platform(record)["artifacts"][0]["url"] = "http://archive.d-robotics.cc/demo.bin"
        with self.assertRaisesRegex(build_catalog.CatalogError, "direct stable HTTPS URL"):
            self.validate(record)
        self.platform(record)["artifacts"][0]["url"] = "https://archive.d-robotics.cc/demo.bin"
        self.validate(record)

    def test_published_split_detector_needs_both_parts(self):
        record = sample_record(self.commit)
        artifacts = self.pair_artifacts()
        self.publish(record, artifacts)
        self.validate(record)
        del artifacts[1]  # one half of the pair alone is unusable on the board
        with self.assertRaisesRegex(build_catalog.CatalogError, "must carry both parta and partb"):
            self.validate(record)

    def test_neural_artifact_format_follows_the_delivery_line(self):
        record = sample_record(self.commit)
        self.publish(record, self.pair_artifacts(parta_format="hbm", partb_format="hbm"))
        with self.assertRaisesRegex(build_catalog.CatalogError, "must be a bin file on x5"):
            self.validate(record)
        s_record = sample_record(self.commit)
        self.platform(s_record)["platform"] = "s100"
        self.publish(s_record, self.pair_artifacts())
        with self.assertRaisesRegex(build_catalog.CatalogError, "must be a hbm file on s100"):
            self.validate(s_record)

    def test_artifact_descriptor_binds_published_artifacts(self):
        descriptor = {"targets": {"x5": {
            "parta": {"file": "demo_parta.bin", "sha256": "a" * 64},
            "partb": {"file": "demo_partb.bin", "sha256": "b" * 64},
        }}}
        commit = self.commit_descriptor(descriptor)
        record = sample_record(
            commit, artifact_descriptor="samples/vision/demo/model/demo.json"
        )
        self.publish(record, self.pair_artifacts())
        self.validate(record)
        self.platform(record)["artifacts"][0]["sha256"] = "c" * 64
        with self.assertRaisesRegex(build_catalog.CatalogError, "does not match the pinned descriptor"):
            self.validate(record)

    def test_release_manifest_binds_download_urls(self):
        descriptor = {"targets": {"x5": {
            "parta": {"file": "demo_parta.bin", "sha256": "a" * 64},
            "partb": {"file": "demo_partb.bin", "sha256": "b" * 64},
        }}}
        commit = self.commit_descriptor(descriptor)
        record = sample_record(
            commit, artifact_descriptor="samples/vision/demo/model/demo.json"
        )
        self.publish(record, self.pair_artifacts())
        self.validate(record)
        # Filename and digest still match, but the URL points at another
        # file's location (the S600 part offered under S100): rejected.
        self.platform(record)["artifacts"][0]["url"] = \
            "https://archive.d-robotics.cc/s600/demo_parta.bin"
        with self.assertRaisesRegex(
            build_catalog.CatalogError, "does not match the pinned release manifest"
        ):
            self.validate(record)

    def test_artifact_descriptor_must_live_inside_the_sample(self):
        record = sample_record(
            self.commit, artifact_descriptor="samples/vision/other/model/demo.json"
        )
        self.publish(record, self.pair_artifacts())
        with self.assertRaisesRegex(build_catalog.CatalogError, "must be a path below sample_path"):
            self.validate(record)
        record["artifact_descriptor"] = "samples/vision/demo/model/absent.json"
        with self.assertRaisesRegex(build_catalog.CatalogError, "cannot be read from this repository"):
            self.validate(record)

    def test_sha256_omitted_when_provenance_is_not_provided(self):
        record = sample_record(self.commit)
        self.platform(record)["model_availability"]["status"] = "published-download"
        self.platform(record)["artifacts"] = [{
            "role": "detector", "filename": "demo.bin", "format": "bin",
            "url": "https://archive.d-robotics.cc/demo.bin",
            "sha256_provenance": "not-provided", "sha256": "a" * 64,
        }]
        with self.assertRaisesRegex(build_catalog.CatalogError, "sha256 must be omitted"):
            self.validate(record)
        del self.platform(record)["artifacts"][0]["sha256"]
        self.validate(record)

    def test_manual_compile_needs_https_guide_and_no_artifacts(self):
        record = sample_record(self.commit)
        platform = self.platform(record)
        platform["model_availability"]["status"] = "manual-compile"
        for guide in (None, "", "   ", "http://example.invalid/guide"):
            platform["model_availability"]["guide_url"] = guide
            with self.subTest(guide_url=guide):
                with self.assertRaisesRegex(build_catalog.CatalogError, "guide_url"):
                    self.validate(record)
        platform["model_availability"]["guide_url"] = "https://example.invalid/guide"
        self.validate(record)
        platform["artifacts"] = [{
            "role": "detector", "filename": "demo.hbm", "format": "hbm",
            "url": "https://archive.d-robotics.cc/demo.hbm",
            "sha256_provenance": "publisher", "sha256": "a" * 64,
        }]
        with self.assertRaisesRegex(build_catalog.CatalogError, "artifacts must be omitted"):
            self.validate(record)

    def test_no_neural_artifact_platform_ships_only_upstream_source(self):
        record = sample_record(self.commit)
        platform = self.platform(record)
        platform["artifacts"] = [{
            "role": "detector", "filename": "demo.hbm", "format": "hbm",
            "url": "https://archive.d-robotics.cc/demo.hbm",
            "sha256_provenance": "publisher", "sha256": "a" * 64,
        }]
        with self.assertRaisesRegex(build_catalog.CatalogError, "role must be upstream-source"):
            self.validate(record)
        platform["artifacts"][0] = {
            "role": "upstream-source", "filename": "demo.tar.gz", "format": "tar.gz",
            "url": "https://codeload.github.com/D-Robotics/demo/tar.gz/" + self.commit[:7],
            "sha256_provenance": "publisher", "sha256": "a" * 64,
        }
        self.validate(record)
        platform["artifacts"][0]["format"] = "pt"
        with self.assertRaisesRegex(build_catalog.CatalogError, "must not look like a deployable model"):
            self.validate(record)

    def test_missing_items_carry_localized_titles(self):
        record = sample_record(self.commit)
        del record["missing"][0]["title"]
        with self.assertRaisesRegex(build_catalog.CatalogError, r"missing\[0\].title is required"):
            self.validate(record)
        record["missing"][0]["title"] = {"zh": "跟踪精度", "en": "Tracking accuracy"}
        record["missing"][0]["status"] = "unknown"
        with self.assertRaisesRegex(build_catalog.CatalogError, "status must be one of"):
            self.validate(record)

    def test_quickstart_on_the_platform_or_the_record(self):
        record = sample_record(self.commit)
        del self.platform(record)["quickstart"]
        with self.assertRaisesRegex(build_catalog.CatalogError, "no quickstart"):
            self.validate(record)
        record["quickstart"] = [{
            "title": {"zh": "运行", "en": "Run"},
            "command": "python3 samples/vision/demo/runtime/python/main.py",
        }]
        self.validate(record)  # the record-level steps cover every platform

    def test_platform_quickstart_must_not_be_empty(self):
        record = sample_record(self.commit)
        self.platform(record)["quickstart"] = []
        with self.assertRaisesRegex(build_catalog.CatalogError, "quickstart must be a non-empty list"):
            self.validate(record)

    def test_evidence_links_stay_summary_sized(self):
        record = sample_record(self.commit)
        self.platform(record)["evidence"][0]["url"] = "sample-evidence/../secrets.txt"
        with self.assertRaisesRegex(build_catalog.CatalogError, "site-relative path"):
            self.validate(record)
        self.platform(record)["evidence"][0]["url"] = "sample-evidence/report.pdf"
        with self.assertRaisesRegex(build_catalog.CatalogError, "sample-evidence/<file>"):
            self.validate(record)

    def test_evidence_file_must_ship_with_the_site(self):
        record = sample_record(self.commit)
        self.platform(record)["evidence"][0]["url"] = "sample-evidence/absent.json"
        with self.assertRaisesRegex(
            build_catalog.CatalogError, "does not exist in model_zoo_web/release"
        ):
            self.validate(record)


class CatalogGateTest(unittest.TestCase):
    """The full build keeps the eleven released families untouched and adds four samples."""

    def test_build_counts_and_sample_ids(self):
        with mock.patch.dict(os.environ, {ENV: "refs/remotes/origin/develop"}):
            catalog_bytes, _metadata = build_catalog.build_catalog()
        catalog = json.loads(catalog_bytes)
        self.assertEqual(catalog["model_count"], 11)
        self.assertEqual(catalog["release_count"], 174)
        self.assertEqual(catalog["sample_count"], 4)
        samples = {record["id"] for record in catalog["models"] if record.get("profile") == "sample"}
        self.assertEqual(samples, {
            "bytetrack/bytetrack/tracking",
            "fasterrcnn/fasterrcnn/detect",
            "hobot_mot/hobot_mot/tracking",
            "lite_avatar/lite_avatar/avatar",
        })
        for record in catalog["models"]:
            if record.get("profile") != "sample":
                continue
            for variant in record["variants"]:
                for platform in variant["platforms"]:
                    self.assertEqual(platform["status"], "available")
                    self.assertIn(
                        platform["model_availability"]["status"],
                        ("published-download", "manual-compile", "no-neural-artifact"),
                    )


class LiteAvatarRegistrationTest(unittest.TestCase):
    """The lite_avatar entry is the only multi-model pipeline sample; it is
    pinned to its develop commit, supported on the three S-series targets as
    Python-only manual-compile runs, and every evidence link ships with the site."""

    ID = "lite_avatar/lite_avatar/avatar"

    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(os.environ, {ENV: "refs/remotes/origin/develop"}):
            catalog_bytes, _metadata = build_catalog.build_catalog()
        catalog = json.loads(catalog_bytes)
        cls.record = next(r for r in catalog["models"] if r.get("id") == cls.ID)

    def test_identity_and_pinning(self):
        record = self.record
        self.assertEqual(record["profile"], "sample")
        self.assertEqual(record["pipeline"], "avatar_pipeline")
        self.assertEqual(record["task_ids"], ["audio-driven-talking-avatar"])
        self.assertEqual(record["sample_commit"], "f77bcea4c203f935efbafe0374cb21c9f3d4cd28")
        self.assertEqual(record["sample_path"], "samples/vision/lite_avatar")

    def test_three_s_targets_python_only_manual_compile(self):
        platforms = self.record["variants"][0]["platforms"]
        self.assertEqual({p["platform"] for p in platforms}, {"s100", "s100p", "s600"})
        for platform in platforms:
            self.assertEqual(platform["runtimes"], ["python"])
            availability = platform["model_availability"]
            self.assertEqual(availability["status"], "manual-compile")
            self.assertNotIn("artifacts", platform)
            self.assertTrue(
                availability["guide_url"].startswith(
                    "https://github.com/D-Robotics/rdk_model_zoo/tree/f77bcea4c203f935efbafe0374cb21c9f3d4cd28/"
                ),
                "the guide must point into the pinned sample commit",
            )
            evidence_files = [
                link["url"] for link in platform["evidence"]
                if not link["url"].startswith("https://")
            ]
            self.assertTrue(all(
                (SCRIPT.parents[1] / "release" / url).is_file() for url in evidence_files
            ), "evidence summaries must ship with the site")

    def test_missing_gaps_are_declared_not_faked(self):
        missing = {item["field"]: item["status"] for item in self.record["missing"]}
        self.assertEqual(missing, {
            "warmed-render-benchmark": "missing",
            "float-vs-board-quality": "missing",
            "model-size-and-compute": "missing",
            "oe-conversion-reports": "missing",
            "published-model-downloads": "missing",
            "cpp-runtime": "not-applicable",
        })


def git_output(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


if __name__ == "__main__":
    unittest.main()
