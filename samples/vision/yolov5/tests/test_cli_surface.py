"""Boundary checks for the simplified YOLOv5 CLI/task split.

These pin the ordinary-sample shape after the Runtime simplification: the
option/listing/dry-run/drawing surface lives in ``cli.py`` without pulling
NumPy/OpenCV for the model-free modes, the real preprocess implementation
(prepare/DetectionContext) lives in the task file ``detection.py``, and
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


class CliSurfaceTests(unittest.TestCase):
    def test_cli_module_imports_without_cv2_or_numpy(self):
        script = (
            "import sys;"
            "import samples.vision.yolov5.runtime.python.cli as cli;"
            "assert callable(cli.build_parser);"
            "assert 'cv2' not in sys.modules, 'cli must stay cv2-free at import';"
            "assert 'numpy' not in sys.modules, 'cli must stay numpy-free at import'"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=SAMPLE.parents[2],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_list_and_dry_run_work_with_heavy_imports_blocked(self):
        """The real entry modes must not import cv2/NumPy/hbm_runtime.

        A ``find_spec`` meta-path hook (the only finder protocol Python 3.12+
        consults) raises ImportError for cv2/numpy/hbm_runtime and their
        submodules; the blocker is proven active first. Each mode runs as its
        own child and must print an explicit completion marker, so a green
        run proves every mode actually executed; ``--help`` exits through
        argparse's SystemExit(0). Parser defaults must equal the cli-owned
        ANCHORS/STRIDES constants.
        """
        prelude = (
            "import sys\n"
            "import importlib.abc\n"
            "class _Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        root = fullname.split('.')[0]\n"
            "        if root in ('cv2', 'numpy', 'hbm_runtime'):\n"
            "            raise ImportError('blocked ' + root)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
        )
        # 0. the blocker is genuinely active for every heavy root
        probe = prelude + (
            "for heavy in ('cv2', 'numpy.hbm_runtime_stub', 'hbm_runtime'):\n"
            "    try:\n"
            "        __import__(heavy)\n"
            "    except ImportError as exc:\n"
            "        assert str(exc).startswith('blocked '), exc\n"
            "    else:\n"
            "        raise AssertionError('blocker inactive for ' + heavy)\n"
            "print('BLOCKER-ACTIVE')\n"
        )
        result = subprocess.run([sys.executable, "-c", probe], cwd=SAMPLE.parents[2],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BLOCKER-ACTIVE", result.stdout)
        # 1. --help goes through argparse SystemExit(0)
        help_script = prelude + (
            "from samples.vision.yolov5.runtime.python import main\n"
            "try:\n"
            "    main.main(['--help'])\n"
            "except SystemExit as exc:\n"
            "    assert exc.code in (0, None), exc.code\n"
            "else:\n"
            "    raise AssertionError('--help must exit through argparse')\n"
            "print('HELP-DONE')\n"
        )
        result = subprocess.run([sys.executable, "-c", help_script], cwd=SAMPLE.parents[2],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HELP-DONE", result.stdout)
        # 2. real modes each complete and stay import-clean
        modes_script = prelude + (
            "from samples.vision.yolov5.runtime.python import main\n"
            "assert main.main(['--list-models', '--target', 'x5']) == 0\n"
            "print('LIST-DONE')\n"
            "assert main.main(['--dry-run', '--target', 'x5', '--model-path', "
            "'/tmp/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin', '--asset-id', "
            "'x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin']) == 0\n"
            "print('DRYRUN-DONE')\n"
            "from samples.vision.yolov5.runtime.python import cli\n"
            "parser = cli.build_parser()\n"
            "args = parser.parse_args(['--target', 'x5'])\n"
            "cli.validate_options(args)\n"
            "assert args.anchors == cli.ANCHORS and tuple(args.strides) == cli.STRIDES\n"
            "print('PARSER-DONE')\n"
            "for heavy in ('cv2', 'numpy', 'hbm_runtime'):\n"
            "    assert heavy not in sys.modules, heavy + ' was imported'\n"
            "print('CLEAN-DONE')\n"
        )
        result = subprocess.run([sys.executable, "-c", modes_script], cwd=SAMPLE.parents[2],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for marker in ("LIST-DONE", "DRYRUN-DONE", "PARSER-DONE", "CLEAN-DONE"):
            self.assertIn(marker, result.stdout)

    def test_cli_exposes_listing_dry_run_and_drawing(self):
        from samples.vision.yolov5.runtime.python import cli

        for name in ("build_parser", "list_models", "dry_run", "draw_detections"):
            self.assertTrue(callable(getattr(cli, name, None)), name)

    def test_preprocess_geometry_lives_in_the_task_module(self):
        from samples.vision.yolov5.runtime.python import detection

        for name in ("DetectionContext", "PreparedInput", "prepare_image"):
            obj = getattr(detection, name, None)
            self.assertIsNotNone(obj, name)
            self.assertTrue(obj.__module__.endswith("detection"), name)

    def test_main_visibly_constructs_task_and_calls_predict(self):
        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "YOLOv5Task"
                            for call in calls), "main() must construct YOLOv5Task itself")
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and call.func.attr == "predict"
                            and isinstance(call.func.value, ast.Name)
                            for call in calls), "main() must call predict itself")

    def test_tensor_io_and_visualization_modules_are_gone(self):
        runtime = SAMPLE / "runtime/python"
        self.assertFalse((runtime / "tensor_io.py").exists(),
                         "tensor_io must be folded into detection.py")
        self.assertFalse((runtime / "visualization.py").exists(),
                         "drawing must live in cli.py")

    def test_main_does_not_assemble_the_sdk_runner_itself(self):
        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        names = [node.id for node in ast.walk(main_fn) if isinstance(node, ast.Name)]
        self.assertNotIn("RuntimeModelRunner", names,
                         "the task class owns runner construction and binding")
        attrs = [node.attr for node in ast.walk(main_fn) if isinstance(node, ast.Attribute)]
        self.assertNotIn("load", attrs,
                         "main must not call runner.load(); the task binds internally")

    def test_task_constructs_from_selection_with_injected_sdk(self):
        from test_yolov5 import FakeRuntime
        from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
        from samples.vision.yolov5.runtime.python.cli import resolve_selection

        runtime = FakeRuntime("s100")
        task = YOLOv5Task(resolve_selection("s100"),
                          runtime_factory=lambda path: runtime)
        result = task.predict(np.zeros((45, 71, 3), np.uint8))
        self.assertEqual(result.boxes.shape[1], 4)
        self.assertEqual(result.class_ids.dtype, np.int32)
        task.set_scheduling_params(priority=7, bpu_cores=[0])
        self.assertEqual(runtime.schedule,
                         {"priority": {"detector": 7}, "bpu_cores": {"detector": [0]}})


if __name__ == "__main__":
    unittest.main()
