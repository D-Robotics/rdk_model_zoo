import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_catalog", SCRIPT)
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)


class AccuracySchemaTest(unittest.TestCase):
    def test_all_supported_tasks_validate_both_stages(self):
        profiles = build_catalog.ACCURACY_SCHEMA["tasks"]
        for task, profile in profiles.items():
            stage = {field: 0.5 for field in profile["required"]}
            accuracy = {
                "dataset": profile.get("dataset", "test dataset"),
                "task": task,
                "images": 100,
                "float_onnx": stage,
                "runtime": stage,
            }
            if profile.get("evaluation_scope"):
                accuracy["evaluation_scope"] = profile["evaluation_scope"]["value"]
            with self.subTest(task=task):
                build_catalog.validate_accuracy(accuracy, task, "fixture.accuracy")

    def test_aliases_use_the_canonical_task_profile(self):
        for alias, canonical in build_catalog.ACCURACY_SCHEMA["task_aliases"].items():
            profile = build_catalog.ACCURACY_SCHEMA["tasks"][canonical]
            stage = {field: 0.5 for field in profile["required"]}
            accuracy = {
                "dataset": "test dataset",
                "task": canonical,
                "images": 100,
                "float_onnx": stage,
                "runtime": stage,
            }
            with self.subTest(alias=alias):
                build_catalog.validate_accuracy(accuracy, alias, "fixture.accuracy")

    def test_missing_metrics_and_non_ratio_values_are_rejected(self):
        accuracy = {
            "dataset": "test dataset",
            "task": "classification",
            "images": 100,
            "float_onnx": {"top1": 0.8, "top5": 0.9},
            "runtime": {"top1": 1.2, "top5": 0.9},
        }
        with self.assertRaisesRegex(build_catalog.CatalogError, r"runtime\.top1.*between 0 and 1"):
            build_catalog.validate_accuracy(accuracy, "cls", "fixture.accuracy")

        accuracy["runtime"] = {"top1": 0.79}
        with self.assertRaisesRegex(build_catalog.CatalogError, r"runtime\.top5.*number"):
            build_catalog.validate_accuracy(accuracy, "cls", "fixture.accuracy")

    def test_dataset_task_and_image_count_are_required(self):
        with self.assertRaisesRegex(build_catalog.CatalogError, r"dataset.*non-empty string"):
            build_catalog.validate_accuracy({}, "detect", "fixture.accuracy")

    def test_obb_requires_local_dota_validation_scope(self):
        profile = build_catalog.ACCURACY_SCHEMA["tasks"]["obb"]
        accuracy = {
            "dataset": profile["dataset"],
            "task": "obb",
            "images": 100,
            "float_onnx": {"map_50": 0.8},
            "runtime": {"map_50": 0.79},
            "evaluation_scope": profile["evaluation_scope"]["value"],
        }
        build_catalog.validate_accuracy(accuracy, "obb", "fixture.accuracy")

        accuracy["evaluation_scope"] = "official_test"
        with self.assertRaisesRegex(build_catalog.CatalogError, "local_dota_val_single_scale"):
            build_catalog.validate_accuracy(accuracy, "obb", "fixture.accuracy")

        accuracy["evaluation_scope"] = profile["evaluation_scope"]["value"]
        accuracy["dataset"] = "DOTA test"
        with self.assertRaisesRegex(build_catalog.CatalogError, r"dataset must be DOTA val"):
            build_catalog.validate_accuracy(accuracy, "obb", "fixture.accuracy")


if __name__ == "__main__":
    unittest.main()
