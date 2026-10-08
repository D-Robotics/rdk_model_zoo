"""Boundary checks for the Runtime-simplification surface of this sample.

The multi-task layout keeps one real task module per task/protocol; the
pure re-export shims (``yolo_detect``, ``yolo26_cls``), the legacy
``pre_process_with_transform`` adapter module, and the duplicated
``run_inference`` entry are gone, and dispatch resolves task classes with
direct imports instead of string tables. No board SDK is loaded and no
board inference is claimed.
"""

from pathlib import Path
import unittest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "python"


class SimplificationSurfaceTests(unittest.TestCase):
    def test_pure_reexport_shims_are_gone(self):
        self.assertFalse((RUNTIME / "yolo_detect.py").exists(),
                         "yolo_detect.py was a pure re-export of detect.py")
        self.assertFalse((RUNTIME / "yolo26_cls.py").exists(),
                         "yolo26_cls.py was a pure re-export of yolo_cls.py")

    def test_legacy_adapter_module_is_gone(self):
        self.assertFalse((RUNTIME / "legacy.py").exists(),
                         "legacy.py only forwarded pre_process_with_transform")

    def test_task_classes_no_longer_expose_the_transform_tuple_adapter(self):
        import sys

        if str(RUNTIME) not in sys.path:
            sys.path.insert(0, str(RUNTIME))
        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect
        from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import YoloSeg
        from samples.vision.ultralytics_yolo.runtime.python.yolo_pose import YoloPose
        from samples.vision.ultralytics_yolo.runtime.python.yolo26_det import YOLO26Detect
        from samples.vision.ultralytics_yolo.runtime.python.yolo26_obb import YOLO26OBB

        for cls in (YoloDetect, YoloSeg, YoloPose, YOLO26Detect, YOLO26OBB):
            self.assertFalse(hasattr(cls, "pre_process_with_transform"),
                             f"{cls.__name__} must not carry the tuple adapter")

    def test_main_has_no_duplicate_run_inference_and_accepts_argv(self):
        import ast

        tree = ast.parse((RUNTIME / "main.py").read_text(encoding="utf-8"))
        names = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertNotIn("run_inference", names,
                         "main() owns the construct→predict sequence; no duplicate entry")
        main_fn = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "main")
        arguments = [arg.arg for arg in main_fn.args.args]
        self.assertIn("argv", arguments, "main() must accept an explicit argv for tests")

    def test_dispatch_uses_no_string_module_table(self):
        import sys

        if str(RUNTIME) not in sys.path:
            sys.path.insert(0, str(RUNTIME))
        import yolo_dispatch

        source = Path(yolo_dispatch.__file__).read_text(encoding="utf-8")
        self.assertNotIn("importlib", source,
                         "dispatch must resolve task classes with direct imports")


if __name__ == "__main__":
    unittest.main()
