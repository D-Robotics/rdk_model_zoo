import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import promote_yolo26_task_batch as promotion


class PromotionValidationTests(unittest.TestCase):
    @staticmethod
    def comparison_row(float_value, board_value):
        return {
            "float": float_value,
            "board": board_value,
            "board_minus_float": board_value - float_value,
            "retention_ratio": board_value / float_value,
        }

    def test_board_receipt_metric_fields_map_to_web_task_fields(self):
        fixtures = {
            "cls": {
                "board": {"top1": 0.72, "top5": 0.91},
                "comparison": {
                    "top1": self.comparison_row(0.75, 0.72),
                    "top5": self.comparison_row(0.93, 0.91),
                },
                "expected_float": {"top1": 0.75, "top5": 0.93},
                "expected_runtime": {"top1": 0.72, "top5": 0.91},
            },
            "seg": {
                "board": {"bbox/AP": 0.42, "segm/AP": 0.37},
                "comparison": {
                    "bbox/AP": self.comparison_row(0.44, 0.42),
                    "segm/AP": self.comparison_row(0.39, 0.37),
                },
                "expected_float": {"box_ap": 0.44, "mask_ap": 0.39},
                "expected_runtime": {"box_ap": 0.42, "mask_ap": 0.37},
            },
            "pose": {
                "board": {"keypoints/AP": 0.51},
                "comparison": {
                    "keypoints/AP": self.comparison_row(0.53, 0.51),
                },
                "expected_float": {"keypoints_ap": 0.53},
                "expected_runtime": {"keypoints_ap": 0.51},
            },
            "obb": {
                "board": {"mAP50": 0.68},
                "comparison": {
                    "mAP50": self.comparison_row(0.70, 0.68),
                },
                "expected_float": {"map_50": 0.70},
                "expected_runtime": {"map_50": 0.68},
            },
        }
        for task, fixture in fixtures.items():
            with self.subTest(task=task):
                float_metrics, runtime_metrics = promotion._metric_pair(task, fixture)
                self.assertEqual(float_metrics, fixture["expected_float"])
                self.assertEqual(runtime_metrics, fixture["expected_runtime"])

    def test_inconsistent_board_metric_or_delta_blocks_promotion(self):
        accuracy = {
            "board": {"bbox/AP": 0.42, "segm/AP": 0.37},
            "comparison": {
                "bbox/AP": self.comparison_row(0.44, 0.41),
                "segm/AP": self.comparison_row(0.39, 0.37),
            },
        }
        with self.assertRaisesRegex(promotion.PromotionError, "board metric and comparator"):
            promotion._metric_pair("seg", accuracy)

        accuracy["comparison"]["bbox/AP"]["board"] = 0.42
        with self.assertRaisesRegex(promotion.PromotionError, "delta is inconsistent"):
            promotion._metric_pair("seg", accuracy)

    def test_obb_accuracy_scope_is_local_dota_val_not_official_test(self):
        comparison = {"mAP50": self.comparison_row(0.70, 0.68)}
        release = {
            "accuracy": {
                "dataset": "DOTA-v1.0 labeled val (single-scale offline metric)",
                "expected_images": 458,
                "comparison_status": "valid",
                "board": {"mAP50": 0.68},
                "comparison": comparison,
                "human_review": {
                    "decision": "accepted",
                    "reviewed_by": "reviewer",
                    "notes": "Reviewed the local validation scope and comparator values.",
                    "automatic_threshold_applied": False,
                    "reviewed_metrics": comparison,
                },
            },
        }
        campaign = {
            "dataset": {
                "dataset": "DOTA-v1.0 labeled val (single-scale offline metric)",
                "images": 458,
            },
        }
        accuracy = promotion._verify_accuracy("obb", release, campaign)
        self.assertEqual(accuracy["dataset"], "DOTA val")
        self.assertEqual(accuracy["evaluation_scope"], "local_dota_val_single_scale")
        self.assertIn("not an official DOTA test-set score", accuracy["scope_note"])
        self.assertEqual(accuracy["runtime"], {"map_50": 0.68})

    def test_checksum_manifest_requires_all_four_release_bundle_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            files = {
                "model.hbm": b"model",
                "oe_report.html": b"<html>report</html>",
                "oe_report_data.json": b'{"schema_version":1}\n',
                "release.json": b'{"status":"staged-local"}\n',
            }
            import hashlib
            for name, content in files.items():
                (bundle / name).write_bytes(content)
            lines = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n"
                            for name, content in files.items())
            (bundle / "SHA256SUMS").write_text(lines, encoding="utf-8")
            self.assertEqual(set(promotion._verify_sums(bundle, "model.hbm")), set(files))
            (bundle / "release.json").write_text('{"status":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(promotion.PromotionError, "SHA256SUMS mismatch"):
                promotion._verify_sums(bundle, "model.hbm")

    def test_only_finalized_released_and_published_manifest_is_accepted(self):
        manifest_path = Path("release.json")
        published = {
            "schema_version": 2,
            "status": "released",
            "publication_status": "published",
            "publication": {"schema_version": 1, "status": "published"},
        }
        promotion._verify_published_manifest(published, manifest_path)

        staged = {
            "schema_version": 2,
            "status": "staged-local",
            "publication_status": "not-uploaded",
            "publication": {"schema_version": 1, "status": "published"},
        }
        with self.assertRaisesRegex(promotion.PromotionError, "not finalized as released/published"):
            promotion._verify_published_manifest(staged, manifest_path)

        unconfirmed = {
            "schema_version": 2,
            "status": "released",
            "publication_status": "uploaded",
            "publication": {"schema_version": 1, "status": "published"},
        }
        with self.assertRaisesRegex(promotion.PromotionError, "not finalized as released/published"):
            promotion._verify_published_manifest(unconfirmed, manifest_path)

        stale_publication = {
            "schema_version": 2,
            "status": "released",
            "publication_status": "published",
            "publication": {
                "schema_version": 1,
                "status": "published",
                "metadata_objects_pending_upload": True,
            },
        }
        with self.assertRaisesRegex(promotion.PromotionError, "still claims pending uploads"):
            promotion._verify_published_manifest(stale_publication, manifest_path)

    def test_bundle_verification_reads_the_separate_finalized_tree(self):
        workbench = Path("/workbench")
        self.assertEqual(
            promotion._finalized_bundle_path(workbench, "seg", "s", "s100p"),
            workbench / "releases/finalized/models/ultralytics_yolo/yolo26/seg/s/s100p",
        )

    def test_finalization_receipt_binds_v2_bundle_and_five_verified_uploads(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            workbench = Path(directory) / "workbench"
            task, size, platform = "cls", "n", "s600"
            bundle = promotion._finalized_bundle_path(workbench, task, size, platform)
            bundle.mkdir(parents=True)
            stage = (workbench / "releases/models/ultralytics_yolo/yolo26" /
                     task / size / platform)
            stage.mkdir(parents=True)
            artifact_name = "model.hbm"
            file_bytes = {
                artifact_name: b"compiled-model",
                "oe_report.html": b"<html>report</html>",
                "oe_report_data.json": b'{"artifact_sha256":"bound"}\n',
            }
            for name, content in file_bytes.items():
                (bundle / name).write_bytes(content)
                (stage / name).write_bytes(content)

            stage_manifest = b'{"schema_version":1,"status":"staged-local"}\n'
            (stage / "release.json").write_bytes(stage_manifest)
            stage_names = (*file_bytes, "release.json")
            (stage / "SHA256SUMS").write_text(
                "".join(f"{hashlib.sha256((stage / name).read_bytes()).hexdigest()}  {name}\n"
                        for name in stage_names), encoding="utf-8")
            stage_release_sha = promotion.sha256(stage / "release.json")
            stage_sums_sha = promotion.sha256(stage / "SHA256SUMS")

            upload_dir = workbench / "outputs/receipts/uploads"
            upload_dir.mkdir(parents=True)
            names = {
                "artifact": artifact_name,
                "oe_report_html": "oe_report.html",
                "oe_report_data": "oe_report_data.json",
            }
            uploaded_objects = {}
            verified_upload_receipts = {}
            for role, name in names.items():
                key = promotion._expected_key(task, size, platform, name)
                receipt_path = promotion._upload_receipt_path(workbench, key)
                receipt_path.write_text(json.dumps({
                    "status": "verified", "key": key,
                    "sha256": promotion.sha256(bundle / name),
                    "size_bytes": (bundle / name).stat().st_size,
                    "acl": "public-read",
                    "url": f"{promotion.OSS_BASE}/{key}",
                }), encoding="utf-8")
                uploaded_objects[role] = {
                    "key": key,
                    "sha256": promotion.sha256(bundle / name),
                    "size_bytes": (bundle / name).stat().st_size,
                    "acl": "public-read",
                    "url": f"{promotion.OSS_BASE}/{key}",
                    "upload_receipt_sha256": promotion.sha256(receipt_path),
                }
                verified_upload_receipts[role] = {
                    "path": str(receipt_path), "sha256": promotion.sha256(receipt_path),
                }

            metadata_objects = [
                {"key": promotion._expected_key(task, size, platform, "release.json"),
                 "name": "release.json", "acl": "public-read"},
                {"key": promotion._expected_key(task, size, platform, "SHA256SUMS"),
                 "name": "SHA256SUMS", "acl": "public-read"},
            ]
            release = {
                "schema_version": 2, "status": "released", "publication_status": "published",
                "model": {"source": "ultralytics_yolo", "provider": "ultralytics",
                          "family": "yolo26", "task": task, "size": size},
                "target": {"platform": platform, "march": "nash-p", "format": "hbm"},
                "artifact": {"name": artifact_name,
                             "url": uploaded_objects["artifact"]["url"]},
                "reports": {
                    "oe_report_html_url": uploaded_objects["oe_report_html"]["url"],
                    "oe_report_data_url": uploaded_objects["oe_report_data"]["url"],
                },
                "provenance": {
                    "staged_release_manifest_sha256": stage_release_sha,
                    "staged_sha256sums_sha256": stage_sums_sha,
                },
                "release_approval": "reviewer",
                "publication": {
                    "schema_version": 1, "status": "published",
                    "oss_bucket": "rdk-model-zoo",
                    "object_prefix": f"models/ultralytics_yolo/yolo26/{task}/{size}/{platform}",
                    "uploaded_objects": uploaded_objects,
                    "metadata_objects": metadata_objects,
                },
            }
            release["artifact"]["release_manifest_url"] = (
                f"{promotion.OSS_BASE}/{metadata_objects[0]['key']}")
            release["artifact"]["checksums_url"] = (
                f"{promotion.OSS_BASE}/{metadata_objects[1]['key']}")
            (bundle / "release.json").write_text(
                json.dumps(release, indent=2) + "\n", encoding="utf-8")
            bundle_names = (*file_bytes, "release.json")
            (bundle / "SHA256SUMS").write_text(
                "".join(f"{promotion.sha256(bundle / name)}  {name}\n"
                        for name in bundle_names), encoding="utf-8")

            for name in ("release.json", "SHA256SUMS"):
                key = promotion._expected_key(task, size, platform, name)
                (promotion._upload_receipt_path(workbench, key)).write_text(json.dumps({
                    "status": "verified", "key": key,
                    "sha256": promotion.sha256(bundle / name),
                    "size_bytes": (bundle / name).stat().st_size,
                    "acl": "public-read", "url": f"{promotion.OSS_BASE}/{key}",
                }), encoding="utf-8")

            final_names = (*bundle_names, "SHA256SUMS")
            file_records = {
                name: {"sha256": promotion.sha256(bundle / name),
                       "size_bytes": (bundle / name).stat().st_size}
                for name in final_names
            }
            finalization = {
                "schema_version": 1, "kind": "yolo26-task-release-finalization",
                "status": "prepared-for-metadata-upload",
                "model": release["model"], "target": release["target"],
                "stage": {"path": str(stage),
                          "release_manifest_sha256": stage_release_sha,
                          "sha256sums_sha256": stage_sums_sha},
                "finalized_bundle": {
                    "path": str(bundle),
                    "release_manifest_sha256": promotion.sha256(bundle / "release.json"),
                    "sha256sums_sha256": promotion.sha256(bundle / "SHA256SUMS"),
                    "files": file_records,
                },
                "uploaded_objects": uploaded_objects,
                "verified_upload_receipts": verified_upload_receipts,
                "metadata_objects_pending_upload": True,
                "pending_metadata_objects": metadata_objects,
                "release_approval": release["release_approval"],
            }
            receipt_path = (workbench / "outputs/receipts/finalizations" /
                            f"{task}-{size}-{platform}.json")
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(json.dumps(finalization, indent=2) + "\n", encoding="utf-8")

            promotion._verify_published_manifest(release, bundle / "release.json")
            urls = promotion._verify_bundle_upload_receipts(
                workbench, task, size, platform, bundle, artifact_name)
            self.assertEqual(len(urls), 5)
            promotion._verify_finalization_receipt(
                workbench, task, size, platform, bundle, release, urls)

            good_receipt = receipt_path.read_bytes()
            tamper_cases = [
                (lambda row: row["finalized_bundle"].update(path=str(bundle.parent)),
                 "path/manifest/checksum hash mismatch"),
                (lambda row: row["finalized_bundle"].update(release_manifest_sha256="0" * 64),
                 "path/manifest/checksum hash mismatch"),
                (lambda row: row["finalized_bundle"]["files"]["oe_report.html"].update(
                    sha256="0" * 64), "file hash/size mismatch"),
                (lambda row: row["verified_upload_receipts"]["artifact"].update(
                    path="/wrong/receipt.json"), "upload-receipt path/hash mismatch"),
            ]
            for mutate, error in tamper_cases:
                tampered = json.loads(good_receipt)
                mutate(tampered)
                receipt_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        promotion.PromotionError, error):
                    promotion._verify_finalization_receipt(
                        workbench, task, size, platform, bundle, release, urls)

            receipt_path.write_bytes(good_receipt)
            sums_key = promotion._expected_key(task, size, platform, "SHA256SUMS")
            sums_upload_receipt = promotion._upload_receipt_path(workbench, sums_key)
            bad_receipt = json.loads(sums_upload_receipt.read_text(encoding="utf-8"))
            bad_receipt["sha256"] = "0" * 64
            sums_upload_receipt.write_text(json.dumps(bad_receipt), encoding="utf-8")
            with self.assertRaisesRegex(promotion.PromotionError, "OSS receipt SHA mismatch"):
                promotion._verify_bundle_upload_receipts(
                    workbench, task, size, platform, bundle, artifact_name)

    def test_oss_receipt_is_bound_to_exact_key_bytes_acl_and_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench = root / "workbench"
            receipts = workbench / "outputs/receipts/uploads"
            receipts.mkdir(parents=True)
            payload = root / "model.hbm"
            payload.write_bytes(b"binary")
            key = "models/ultralytics_yolo/yolo26/cls/n/s600/model.hbm"
            url = f"{promotion.OSS_BASE}/{key}"
            import hashlib
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            receipt_path = receipts / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
            receipt = {
                "status": "verified",
                "key": key,
                "sha256": digest,
                "size_bytes": payload.stat().st_size,
                "acl": "public-read",
                "url": url,
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(promotion._verify_upload_receipt(workbench, key, payload), url)
            receipt["acl"] = "private"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(promotion.PromotionError, "must be public-read"):
                promotion._verify_upload_receipt(workbench, key, payload)

    def test_missing_gate_evidence_produces_dry_run_blockers_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            workbench = Path(directory)
            configs = workbench / "configs"
            configs.mkdir()
            geometry = {"cls": 224, "seg": 640, "pose": 640, "obb": 1024}
            for task in promotion.TASKS:
                for size in promotion.SIZES:
                    config = {
                        "schema_version": 4,
                        "family": "yolo26",
                        "source": "ultralytics_yolo",
                        "task": task,
                        "size": size,
                        "source_sha256": "a" * 64,
                        "source_commit": "b" * 40,
                        "geometry": {"size": [geometry[task], geometry[task]]},
                        "dataset": {"dataset": promotion.DATASET_LABELS[task], "images": 10},
                        "model": {
                            "parameter_count": 1000,
                            "gflops": 1.0,
                            "measurement_method": "test fixture",
                        },
                        "license": {"name": "AGPL-3.0", "url": "https://example.invalid/license"},
                        "platforms": {
                            platform: {
                                "march": "nash-p",
                                "format": "hbm",
                                "image_id": "sha256:" + "c" * 64,
                            }
                            for platform in promotion.PLATFORMS
                        },
                    }
                    name = f"campaign-yolo26-{task}-{size}-v2.json"
                    (configs / name).write_text(json.dumps(config), encoding="utf-8")

            inputs_before = (promotion.WEB_ROOT / "release/inputs.json").read_bytes()
            task_paths = [promotion.WEB_ROOT / "data/vision/ultralytics_yolo/yolo26" /
                          f"{task}.yaml" for task in promotion.TASKS]
            task_contents_before = {path: path.read_bytes() if path.exists() else None
                                    for path in task_paths}
            no_gate = lambda task, size, platform: (False, "fixture gate pending")
            writes, blockers, summary = promotion.make_plan(
                workbench, "2026-09-30", gate_auditor=no_gate)
            self.assertEqual(summary["entry_count"], 60)
            self.assertEqual(summary["release_receipts_expected"], 300)
            self.assertEqual(summary["detect_release_mappings_preserved"], 40)
            self.assertTrue(any("task_release_gate failed" in row for row in blockers))
            self.assertEqual((promotion.WEB_ROOT / "release/inputs.json").read_bytes(), inputs_before)
            inputs_path = promotion.WEB_ROOT / "release/inputs.json"
            contents_before = {inputs_path: inputs_before, **task_contents_before}
            for path in writes:
                self.assertEqual(path.read_bytes() if path.exists() else None,
                                 contents_before.get(path))
            for path, contents in task_contents_before.items():
                self.assertEqual(path.read_bytes() if path.exists() else None, contents)

    def test_apply_requires_an_explicit_release_date(self):
        with self.assertRaisesRegex(SystemExit, "--apply requires an explicit"):
            promotion.main(["--apply"])

    def test_rebuild_object_keys_and_legacy_pose_source_stage_are_bounded(self):
        workbench = Path("/workbench")
        build_id = "20260930T1402Z-becb0688-yolo26s-pose-s100p-fixed-max-asym-b8-r01"
        canonical = workbench / "releases/models/ultralytics_yolo/yolo26/pose/s/s100p"
        versioned = canonical / "rebuilds" / build_id
        for path in (canonical, versioned):
            self.assertEqual(promotion._source_stage_path(
                workbench, "pose", "s", "s100p", {"path": str(path)}, build_id), path)
        for task, size, platform, path in (
                ("pose", "s", "s100p", canonical.parent),
                ("pose", "s", "s100", canonical),
                ("obb", "x", "s100p", canonical)):
            with self.subTest(path=path), self.assertRaises(promotion.PromotionError):
                promotion._source_stage_path(
                    workbench, task, size, platform, {"path": str(path)}, build_id)
        key = promotion._expected_key("pose", "s", "s100p", "release.json", build_id)
        self.assertIn(f"/rebuilds/{build_id}/", key)
        with self.assertRaises(promotion.PromotionError):
            promotion._expected_key("pose", "s", "s100p", "release.json", "../../old")

    def test_historical_float_reference_binds_both_files_and_stays_in_workbench(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = root / "runs/benchmark/rebuild"
            (bench / "output").mkdir(parents=True)
            baseline = root / "runs/benchmark/original/output/float_result.json"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps({"model": {"task": "pose", "size": "s"}}))
            reference_path = bench / "output/float_result_reference.json"
            reference = {"schema_version": 1, "kind": "s-family-float-result-reference",
                         "float_result": {"path": str(baseline), "sha256": promotion.sha256(baseline)}}
            reference_path.write_text(json.dumps(reference))
            release = {"provenance": {
                "float_result_sha256": promotion.sha256(baseline),
                "float_result_reference_sha256": promotion.sha256(reference_path)}}
            self.assertEqual(promotion._find_float_result(bench, release, root)["model"]["task"], "pose")
            baseline.write_text('{"tampered":true}')
            with self.assertRaisesRegex(promotion.PromotionError, "baseline changed"):
                promotion._find_float_result(bench, release, root)
            reference["float_result"]["path"] = str(root.parent / "outside.json")
            reference_path.write_text(json.dumps(reference))
            release["provenance"]["float_result_reference_sha256"] = promotion.sha256(reference_path)
            with self.assertRaisesRegex(promotion.PromotionError, "safe historical baseline"):
                promotion._find_float_result(bench, release, root)

    def test_agent_accuracy_review_requires_explicit_user_publication_authorization(self):
        comparison = {"keypoints/AP": self.comparison_row(0.56, 0.467)}
        release = {"accuracy": {
            "dataset": "COCO2017 val", "expected_images": 5000,
            "comparison_status": "valid", "board": {"keypoints/AP": 0.467},
            "comparison": comparison, "review_record": {
                "reviewer_kind": "agent", "reviewed_by": "Codex agent",
                "decision": "accepted", "notes": "Reviewed and disclosed the measured accuracy tradeoff.",
                "reviewed_metrics": comparison, "automatic_threshold_applied": False}}}
        campaign = {"dataset": {"dataset": "COCO2017 val", "images": 5000}}
        with self.assertRaisesRegex(promotion.PromotionError, "explicit user publication"):
            promotion._verify_accuracy("pose", release, campaign)
        release["release_approval"] = {"authorized": True,
            "kind": "explicit-user-web-publication-authorization"}
        self.assertEqual(promotion._verify_accuracy("pose", release, campaign)["runtime"],
                         {"keypoints_ap": 0.467})
        release["accuracy"]["review_record"]["automatic_threshold_applied"] = True
        with self.assertRaisesRegex(promotion.PromotionError, "automatic accuracy threshold"):
            promotion._verify_accuracy("pose", release, campaign)

    def test_rebuild_manifest_must_cover_exactly_the_four_repaired_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "selection.json"
            manifest.write_text(json.dumps({"schema_version": 1, "records": []}))
            with self.assertRaises(promotion.PromotionError):
                promotion._load_rebuild_overrides(manifest, root)


if __name__ == "__main__":
    unittest.main()
