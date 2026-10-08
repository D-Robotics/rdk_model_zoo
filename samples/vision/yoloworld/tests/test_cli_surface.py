"""Boundary checks for the simplified YOLOWorld CLI/task split.

Pins the ordinary-sample shape: the option/listing/dry-run/drawing surface
lives in ``cli.py`` without importing NumPy/OpenCV for the model-free modes,
``main.py`` keeps the visible construct → predict sequence, and drawing is
no longer a standalone module. No board SDK is loaded; no board inference
is claimed.
"""

from pathlib import Path
import ast
import json
import subprocess
import sys
import unittest

import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]


class CliSurfaceTests(unittest.TestCase):
    def test_cli_module_imports_without_cv2_or_numpy(self):
        script = (
            "import sys;"
            "import samples.vision.yoloworld.runtime.python.cli as cli;"
            "assert callable(cli.build_parser);"
            "assert 'cv2' not in sys.modules, 'cli must stay cv2-free at import';"
            "assert 'numpy' not in sys.modules, 'cli must stay numpy-free at import'"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=SAMPLE.parents[2],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_exposes_listing_dry_run_prompts_and_drawing(self):
        from samples.vision.yoloworld.runtime.python import cli

        for name in ("build_parser", "list_models", "dry_run", "parse_prompts",
                     "draw_results", "save_image"):
            self.assertTrue(callable(getattr(cli, name, None)), name)

    def test_main_visibly_constructs_task_and_calls_predict(self):
        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "YOLOWorldTask"
                            for call in calls), "main() must construct YOLOWorldTask itself")
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and call.func.attr == "predict"
                            and isinstance(call.func.value, ast.Name)
                            for call in calls), "main() must call predict itself")

    def test_visualization_module_is_gone(self):
        self.assertFalse((SAMPLE / "runtime/python/visualization.py").exists(),
                         "drawing must live in cli.py")

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
        from test_yoloworld import FakeRuntime, SAMPLE as YOLOWORLD_SAMPLE
        from samples.vision.yoloworld.runtime.python.model_binding import resolve_selection
        from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask

        runtime = FakeRuntime()
        runtime.scores[0, 7, 0] = 0.9
        runtime.boxes[0, 7] = [1, 2, 100, 200]
        vocabulary = json.loads(
            (YOLOWORLD_SAMPLE / "test_data/offline_vocabulary_embeddings.json").read_text())
        task = YOLOWorldTask(resolve_selection("x5"), vocabulary,
                             runtime_factory=lambda path: runtime)
        result = task.predict(np.zeros((40, 60, 3), np.uint8), ["dog"])
        self.assertEqual(len(result.scores), 1)
        task.set_scheduling_params(priority=3, bpu_cores=[0])
        self.assertEqual(runtime.scheduling,
                         {"priority": {"yolo": 3}, "bpu_cores": {"yolo": [0]}})


if __name__ == "__main__":
    unittest.main()
