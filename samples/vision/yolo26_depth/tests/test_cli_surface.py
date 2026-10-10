"""Boundary checks for the simplified YOLO26-depth CLI/task split.

Pins the ordinary-sample shape after the Runtime simplification: the depth
display helper (``colorize_depth``) lives in ``cli.py`` with the other
presentation code instead of a one-function standalone module, and
``main.py`` keeps the visible construct → predict sequence. Geometry and
depth restoration stay shared with the offline evaluator in their own
modules. No board SDK is loaded and no board inference is claimed.
"""

from pathlib import Path
import ast
import unittest

SAMPLE = Path(__file__).resolve().parents[1]


class CliSurfaceTests(unittest.TestCase):
    def test_colorize_depth_lives_in_the_cli_module(self):
        from samples.vision.yolo26_depth.runtime.python import cli

        colorize = getattr(cli, "colorize_depth", None)
        self.assertTrue(callable(colorize), "cli must own the depth display helper")
        self.assertTrue(colorize.__module__.endswith("cli"))

    def test_visualization_module_is_gone(self):
        self.assertFalse((SAMPLE / "runtime/python/visualization.py").exists(),
                         "colorize_depth must live in cli.py")

    def test_main_visibly_constructs_task_and_calls_predict(self):
        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)]
        constructed = any(
            (isinstance(call.func, ast.Name) and call.func.id == "Yolo26DepthTask")
            or (isinstance(call.func, ast.Attribute) and call.func.attr == "Yolo26DepthTask")
            for call in calls)
        self.assertTrue(constructed, "main() must construct Yolo26DepthTask itself")
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and call.func.attr == "predict"
                            and isinstance(call.func.value, ast.Name)
                            for call in calls), "main() must call predict itself")

    def test_main_does_not_assemble_the_sdk_runner_itself(self):
        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
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
        import io
        import contextlib
        from types import SimpleNamespace
        from unittest import mock

        import numpy as np

        from test_depth import metadata
        from samples.vision.yolo26_depth.runtime.python import yolo26_depth as model_runner
        from samples.vision.yolo26_depth.runtime.python.cli import resolve_selection
        from samples.vision.yolo26_depth.runtime.python.yolo26_depth import Yolo26DepthTask

        raw = np.linspace(-5, 6, 192 * 192, dtype=np.float32).reshape(1, 192, 192, 1)
        values = metadata(False)
        runtime = SimpleNamespace(
            **{k: v for k, v in values.items() if k != "model_name"},
            run=lambda inputs: {"depth": {"output0": raw}},
            set_scheduling_params=lambda **kw: None,
        )
        real = model_runner.RuntimeModelRunner
        with mock.patch(
            "samples.vision.yolo26_depth.runtime.python.yolo26_depth.RuntimeModelRunner",
            side_effect=lambda selection: real(selection, runtime=runtime),
        ), contextlib.redirect_stdout(io.StringIO()):
            task = Yolo26DepthTask(resolve_selection("x5"))
            result = task.predict(np.zeros((48, 64, 3), np.uint8))
        self.assertEqual(result.depth_native.shape, (48, 64))


if __name__ == "__main__":
    unittest.main()
