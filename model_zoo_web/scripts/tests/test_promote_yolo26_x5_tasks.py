import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import promote_yolo26_x5_tasks as promotion


class X5PromotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.web = self.root / "model_zoo_web"
        (self.web / "release/reports").mkdir(parents=True)
        (self.root / "samples/vision/ultralytics_yolo").mkdir(parents=True)
        self.before = {}
        for task in promotion.TASKS:
            relative = Path("data/vision/ultralytics_yolo/yolo26") / f"{task}.yaml"
            target = self.web / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            original = yaml.safe_load((promotion.WEB_ROOT / relative).read_text())
            # Reconstruct the deployed 100-entry base even when this suite is
            # run from the later 120-entry release commit.
            for variant in original["variants"]:
                variant["platforms"] = [p for p in variant["platforms"] if p["platform"] != "x5"]
            target.write_text(yaml.safe_dump(original, sort_keys=False, allow_unicode=True))
            self.before[task] = original
        self.inputs = json.loads((promotion.WEB_ROOT / "release/inputs.json").read_text())
        self.inputs["releases"] = {key: value for key, value in self.inputs["releases"].items()
                                   if not (key.endswith("/x5") and key.split("/")[2] in promotion.TASKS)}
        (self.web / "release/inputs.json").write_text(json.dumps(self.inputs))
        self.builder = promotion._load_catalog_builder()

    def provider(self, workbench, task, size, release_date):
        variant = next(v for v in self.before[task]["variants"] if v["size"] == size)
        entry = copy.deepcopy(variant["platforms"][0])
        entry["platform"] = "x5"
        artifact = entry["artifact"]
        artifact.update(format="bin", march="bayes-e")
        for field in ("url", "release_manifest_url", "checksums_url"):
            artifact[field] = artifact[field].replace("/s600/", "/x5/").replace(".hbm", ".bin")
        entry["reports"]["oe_conversion_url"] = artifact["url"].rsplit("/", 1)[0] + "/oe_report.html"
        report = {"target": {"platform": "x5"}, "provenance": {"artifact_sha256": artifact["sha256"]}}
        return entry, json.dumps(report).encode()

    def plan(self, provider=None, models=None, originals=None):
        with patch.object(promotion, "WEB_ROOT", self.web), \
             patch.object(promotion, "_load_catalog_builder", return_value=self.builder), \
             patch.object(self.builder, "ROOT", self.root), \
             patch.object(self.builder, "DATA_ROOT", self.web / "data"):
            return promotion.make_plan(self.root, "2026-10-01", provider or self.provider,
                                       models=models, originals=originals)

    def test_apply_replaces_reviewed_catalog_and_rolls_back_failed_increment(self):
        originals = {}
        writes, _ = self.plan(models=(("cls", "n"),), originals=originals)
        with patch.object(promotion, "WEB_ROOT", self.web):
            promotion.commit_incremental_plan(writes, originals)
        inputs_path = self.web / "release/inputs.json"
        self.assertEqual(len(json.loads(inputs_path.read_bytes())["releases"]), 101)
        for path, payload in writes.items():
            self.assertEqual(path.read_bytes(), payload)

        originals = {}
        writes, _ = self.plan(models=(("seg", "n"), ("seg", "s")), originals=originals)
        replace = promotion.os.replace
        calls = 0

        def fail_second_replace(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated commit failure")
            return replace(source, target)

        with patch.object(promotion, "WEB_ROOT", self.web), \
             patch.object(promotion.os, "replace", side_effect=fail_second_replace):
            with self.assertRaisesRegex(OSError, "simulated commit failure"):
                promotion.commit_incremental_plan(writes, originals)
        for path, payload in originals.items():
            self.assertEqual(path.read_bytes(), payload)
        for path in set(writes) - set(originals):
            self.assertFalse(path.exists())
        self.assertEqual(len(json.loads(inputs_path.read_bytes())["releases"]), 101)

    def test_apply_refuses_catalog_changed_after_review(self):
        originals = {}
        writes, _ = self.plan(models=(("cls", "n"),), originals=originals)
        path = self.web / "data/vision/ultralytics_yolo/yolo26/cls.yaml"
        changed = path.read_bytes() + b"\n# concurrent change\n"
        path.write_bytes(changed)
        with patch.object(promotion, "WEB_ROOT", self.web):
            with self.assertRaisesRegex(ValueError, "Catalog changed after planning"):
                promotion.commit_incremental_plan(writes, originals)
        self.assertEqual(path.read_bytes(), changed)
        for report in set(writes) - set(originals):
            self.assertFalse(report.exists())

    @staticmethod
    def install_plan(writes):
        for path, content in writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def test_incremental_plan_preserves_all_existing_platforms_and_covers(self):
        writes, summary = self.plan()
        self.assertEqual(summary["total_entries"], 120)
        self.assertEqual(summary["new_entry_count"], 20)
        self.assertEqual(summary["public_objects_verified"], 100)
        after_inputs = json.loads(writes[self.web / "release/inputs.json"])
        self.assertEqual(after_inputs["models"], self.inputs["models"])
        for key, value in self.inputs["releases"].items():
            self.assertEqual(after_inputs["releases"][key], value)
        for task, original in self.before.items():
            path = self.web / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml"
            after = yaml.safe_load(writes[path])
            self.assertEqual(yaml.safe_load(path.read_text()), original)  # dry run is read-only
            for variant in after["variants"]:
                self.assertEqual({p["platform"] for p in variant["platforms"]}, {"s600", "s100p", "s100", "x5"})
                variant["platforms"] = [p for p in variant["platforms"] if p["platform"] != "x5"]
            self.assertEqual(after, original)

    def test_subset_then_remaining_models_preserve_and_extend_catalog(self):
        first = tuple(
            [("cls", size) for size in promotion.SIZES]
            + [("seg", size) for size in promotion.SIZES[:4]]
            + [("obb", size) for size in promotion.SIZES]
        )
        remaining = tuple(model for model in promotion.ALL_MODELS if model not in set(first))

        first_writes, first_summary = self.plan(models=first)
        self.assertEqual(first_summary["existing_entries_preserved"], 100)
        self.assertEqual(first_summary["new_entry_count"], 14)
        self.assertEqual(first_summary["public_objects_verified"], 70)
        self.assertEqual(first_summary["total_entries"], 114)
        first_inputs = json.loads(first_writes[self.web / "release/inputs.json"])
        self.assertEqual(len(first_inputs["releases"]), 114)
        self.assertTrue(all(first_inputs["releases"][key] == value
                            for key, value in self.inputs["releases"].items()))
        self.install_plan(first_writes)

        second_writes, second_summary = self.plan(models=remaining)
        self.assertEqual(second_summary["existing_entries_preserved"], 114)
        self.assertEqual(second_summary["new_entry_count"], 6)
        self.assertEqual(second_summary["public_objects_verified"], 30)
        self.assertEqual(second_summary["total_entries"], 120)
        second_inputs = json.loads(second_writes[self.web / "release/inputs.json"])
        self.assertEqual(len(second_inputs["releases"]), 120)
        self.assertTrue(all(second_inputs["releases"][key] == value
                            for key, value in first_inputs["releases"].items()))
        added_keys = set(second_inputs["releases"]) - set(self.inputs["releases"])
        self.assertEqual(
            added_keys,
            {f"ultralytics_yolo/yolo26/{task}/{size}/x5" for task, size in promotion.ALL_MODELS},
        )
        for task in promotion.TASKS:
            task_write = second_writes.get(
                self.web / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml")
            if task_write is not None:
                after = yaml.safe_load(task_write)
                for variant in after["variants"]:
                    platforms = [record["platform"] for record in variant["platforms"]]
                    self.assertEqual(len(platforms), len(set(platforms)))

    def test_invalid_and_duplicate_model_selections_are_rejected(self):
        with self.assertRaisesRegex(promotion.argparse.ArgumentTypeError, "Invalid model"):
            promotion.parse_models("cls-q")
        with self.assertRaisesRegex(promotion.argparse.ArgumentTypeError, "Duplicate model"):
            promotion.parse_models("cls-n,cls-n")

        first_writes, _ = self.plan(models=(("cls", "n"),))
        self.install_plan(first_writes)
        with self.assertRaisesRegex(ValueError, "entry/report already exists"):
            self.plan(models=(("cls", "n"),))

    def test_performance_frame_count_is_derived_from_valid_one_and_two_stream_receipts(self):
        performance = {
            "end_to_end": [
                {"pipeline_streams": 1, "runs_per_round": 200, "rounds": 3,
                 "timed_frames": 600, "frames_per_round": None},
                {"pipeline_streams": 2, "runs_per_round": 200, "rounds": 3,
                 "timed_frames": 1200, "frames_per_round": None},
            ],
        }
        normalized = promotion.normalize_performance_for_catalog(performance)
        self.assertEqual([row["frames_per_round"] for row in normalized["end_to_end"]], [200, 200])
        self.assertEqual([row["frames_per_round"] for row in performance["end_to_end"]], [None, None])

    def test_performance_frame_count_rejects_inconsistent_timed_frames(self):
        performance = {
            "end_to_end": [
                {"pipeline_streams": 1, "runs_per_round": 200, "rounds": 3,
                 "timed_frames": 599, "frames_per_round": None},
            ],
        }
        with self.assertRaisesRegex(ValueError, "timed_frames does not match"):
            promotion.normalize_performance_for_catalog(performance)

    def test_report_for_another_binary_is_rejected(self):
        def changed(*args):
            entry, raw = self.provider(*args)
            report = json.loads(raw)
            report["provenance"]["artifact_sha256"] = "f" * 64
            return entry, json.dumps(report).encode()
        with self.assertRaisesRegex(ValueError, "OE data does not bind"):
            self.plan(changed)

    def test_pending_gate_cannot_become_an_official_entry(self):
        gate = types.SimpleNamespace(audit_one=lambda task, size: {
            "name": f"{task}-{size}-x5", "ready": False, "checks": []})
        with patch.object(promotion, "load_workbench", return_value=(gate, None)):
            with self.assertRaisesRegex(ValueError, "gate is not ready"):
                promotion.verified_entry(self.root, "cls", "n", "2026-10-01")

    def test_an_existing_x5_report_is_not_overwritten(self):
        (self.web / "release/reports/yolo26-cls-n-x5-oe-data.json").write_text("original")
        with self.assertRaisesRegex(ValueError, "entry/report already exists"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
