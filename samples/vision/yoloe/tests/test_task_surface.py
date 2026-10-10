"""Boundary checks for the simplified YOLOE task-file layout.

Pins the ordinary-sample shape after the Runtime simplification: the task
file ``yoloe.py`` owns the real prepared/result types, the per-variant
decode, and the validation helpers that used to be spread over
``pipeline_io.py``/``postprocess.py``/``decode.py``. The configuration
module stays shared with the native launcher. No board SDK is loaded and no
board inference is claimed.
"""

from pathlib import Path
import subprocess
import sys
import unittest

SAMPLE = Path(__file__).resolve().parents[1]


class TaskSurfaceTests(unittest.TestCase):
    def test_preview_runs_without_image_library_or_board_sdk(self):
        script = '''
import importlib.abc
import sys
class BlockRuntimeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv2', 'hbm_runtime'):
            raise ImportError('unexpected preview dependency: ' + fullname)
sys.meta_path.insert(0, BlockRuntimeDependencies())
from samples.vision.yoloe.runtime.python.main import main
raise SystemExit(main(sys.argv[1:]))
'''
        for options in (['--target', 'x5', '--list-models'],
                        ['--target', 'x5', '--dry-run'],
                        ['--target', 's100', '--dry-run']):
            with self.subTest(options=options):
                result = subprocess.run(
                    [sys.executable, '-c', script, *options],
                    cwd=SAMPLE.parents[2], capture_output=True, text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())

    def test_task_module_owns_prepared_result_and_decode(self):
        from samples.vision.yoloe.runtime.python import yoloe

        for name in ("Prepared", "Result", "validate_context", "decode_result",
                     "decode_x5", "validate_semantic", "YOLOE", "Config"):
            obj = getattr(yoloe, name, None)
            self.assertIsNotNone(obj, name)

    def test_preprocess_body_is_defined_in_the_task_module(self):
        import inspect

        from samples.vision.yoloe.runtime.python import yoloe

        self.assertTrue(inspect.getsourcefile(yoloe.YOLOE.preprocess).endswith("yoloe.py"))
        self.assertNotIn("pipeline_io", inspect.getsource(yoloe.YOLOE.preprocess))

    def test_split_modules_are_gone(self):
        runtime = SAMPLE / "runtime/python"
        for name in ("pipeline_io.py", "postprocess.py", "decode.py"):
            self.assertFalse((runtime / name).exists(), f"{name} must be folded into yoloe.py")

    def test_config_module_stays_shared_with_the_native_launcher(self):
        # The C++ launcher imports Config/validate_config from config.py; the
        # merge must not move that shared surface.
        self.assertTrue((SAMPLE / "runtime/python/yoloe.py").exists())
        from samples.vision.yoloe.runtime.python.cli import Config, validate_config  # noqa: F401

    def test_main_does_not_assemble_the_sdk_runner_itself(self):
        import ast

        tree = ast.parse((SAMPLE / "runtime/python/main.py").read_text(encoding="utf-8"))
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        names = [node.id for node in ast.walk(main_fn) if isinstance(node, ast.Name)]
        self.assertNotIn("build_runner", names,
                         "the model class owns runner construction")
        loads = [node for node in ast.walk(main_fn)
                 if isinstance(node, ast.Attribute) and node.attr == "load"
                 and isinstance(node.value, ast.Name) and node.value.id != "json"]
        self.assertFalse(loads, "main must not call runner.load(); the model binds internally")

    def test_model_exposes_scheduling_on_itself(self):
        from samples.vision.yoloe.runtime.python import yoloe

        self.assertTrue(callable(getattr(yoloe.YOLOE, "set_scheduling_params", None)),
                        "main applies scheduling through the model, not a bare runner")


if __name__ == "__main__":
    unittest.main()
