"""Boundary checks for the simplified FCOS CLI/task split.

Pins the ordinary-sample shape after the Runtime simplification: the option/
listing/dry-run/presentation surface lives in ``cli.py``; the model-free
modes import neither NumPy, OpenCV, nor the board SDK; the real preprocess
implementation (prepare/ImageContext) lives in the task file ``fcos.py``;
``main.py`` keeps the visible construct → predict sequence. No board SDK is
loaded and no board inference is claimed.
"""

from pathlib import Path
import ast
import subprocess
import sys
import unittest

import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
MAIN = SAMPLE / "runtime/python/main.py"


def _imported_packages(args):
    result = subprocess.run([sys.executable, "-X", "importtime", str(MAIN), *args],
                            cwd=SAMPLE.parents[2], text=True, capture_output=True, timeout=60)
    packages = {line.rsplit("|", 1)[-1].strip()
                for line in result.stderr.splitlines() if line.startswith("import time:")}
    return result, packages


class CliSurfaceTests(unittest.TestCase):
    def test_model_free_modes_import_no_numpy_opencv_or_sdk(self):
        for args in (["--help"], ["--list-models"], ["--dry-run", "--target", "x5"]):
            with self.subTest(args=args):
                result, packages = _imported_packages(args)
                self.assertEqual(result.returncode, 0, result.stderr)
                for banned in ("cv2", "numpy", "hbm_runtime"):
                    self.assertNotIn(banned, packages, f"{args} imported {banned}")

    def test_cli_exposes_listing_dry_run_labels_and_saving(self):
        from samples.vision.fcos.runtime.python import cli

        for name in ("build_parser", "list_models", "dry_run", "load_labels", "save_result"):
            self.assertTrue(callable(getattr(cli, name, None)), name)

    def test_preprocess_geometry_lives_in_the_task_module(self):
        from samples.vision.fcos.runtime.python import fcos

        for name in ("ImageContext", "PreparedInput", "prepare"):
            obj = getattr(fcos, name, None)
            self.assertIsNotNone(obj, name)
            self.assertTrue(obj.__module__.endswith("fcos"), name)

    def test_main_visibly_constructs_task_and_calls_predict(self):
        tree = ast.parse(MAIN.read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)]
        constructed = any(
            (isinstance(call.func, ast.Name) and call.func.id == "FCOSTask")
            or (isinstance(call.func, ast.Attribute) and call.func.attr == "FCOSTask")
            for call in calls)
        self.assertTrue(constructed, "main() must construct FCOSTask itself")
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and call.func.attr == "predict"
                            and isinstance(call.func.value, ast.Name)
                            for call in calls), "main() must call predict itself")

    def test_tensor_io_module_is_gone(self):
        self.assertFalse((SAMPLE / "runtime/python/tensor_io.py").exists(),
                         "tensor_io must be folded into fcos.py")

    def test_main_does_not_assemble_the_sdk_runner_itself(self):
        tree = ast.parse(MAIN.read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        names = [node.id for node in ast.walk(main_fn) if isinstance(node, ast.Name)]
        self.assertNotIn("RuntimeModelRunner", names,
                         "the task class owns runner construction and binding")
        loads = [node for node in ast.walk(main_fn)
                 if isinstance(node, ast.Attribute) and node.attr == "load"
                 and isinstance(node.value, ast.Name) and node.value.id != "json"]
        self.assertFalse(loads, "main must not call runner.load(); the task binds internally")

    def test_task_constructs_from_selection_with_injected_sdk(self):
        from test_fcos_contract import fixture_outputs, metadata_for
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import resolve_selection

        values = metadata_for("efficientnetb0")
        raw = fixture_outputs()

        class FakeRuntime:
            model_names = [values["model_name"]]
            input_names = {values["model_name"]: values["input_names"]}
            input_shapes = {values["model_name"]: values["input_shapes"]}
            input_dtypes = {values["model_name"]: values["input_dtypes"]}
            output_names = {values["model_name"]: values["output_names"]}
            output_shapes = {values["model_name"]: values["output_shapes"]}
            output_dtypes = {values["model_name"]: values["output_dtypes"]}
            output_quants = {values["model_name"]: values["output_quants"]}

            def run(self, inputs):
                return {values["model_name"]: raw}

        task = FCOSTask(resolve_selection("x5"), runtime_factory=lambda path: FakeRuntime())
        result = task.predict(np.zeros((512, 512, 3), dtype=np.uint8))
        self.assertEqual(result.boxes.shape[1], 4)
        self.assertEqual(result.class_ids.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
