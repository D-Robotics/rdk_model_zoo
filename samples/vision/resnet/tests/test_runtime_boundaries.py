"""Acceptance checks for the ResNet runtime responsibility boundaries."""

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_predict_entry import _fake_runtime, _score_vector


class RuntimeBoundaryTests(unittest.TestCase):
    def test_model_module_has_no_catalog_or_custom_selection_api(self):
        from samples.vision.resnet.runtime.python import classify
        for name in ("BINDING_TABLE", "resolve_selection", "list_available_assets", "custom_selection"):
            self.assertFalse(hasattr(classify, name), name)

    def test_model_free_cli_never_imports_execution_dependencies(self):
        main = Path(__file__).resolve().parents[1] / "runtime/python/main.py"
        guard = """import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('hbm_runtime', 'cv2', 'numpy'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
        for args in (("--help",), ("--list-models",), ("--dry-run", "--target", "x5")):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                run = subprocess.run([sys.executable, "-c", guard, str(main), *args],
                                     cwd=directory, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)

    def test_model_uses_shared_image_and_label_utilities(self):
        from samples.vision.resnet.runtime.python import classify
        from utils.py_utils.image import read_bgr_image
        from utils.py_utils.labels import validate_labels

        self.assertIs(classify.read_bgr_image, read_bgr_image)
        self.assertIs(classify.validate_labels, validate_labels)

    def test_construct_and_predict_with_model_path(self):
        from samples.vision.resnet.runtime.python import classify

        image = np.full((40, 60, 3), 127, dtype=np.uint8)
        for target in ("x5", "s100", "s600"):
            with self.subTest(target=target):
                calls = []
                runtime = _fake_runtime(target, [_score_vector(42)], calls)

                with tempfile.TemporaryDirectory() as directory:
                    artifact = Path(directory) / "model.bin"
                    artifact.touch()
                    with patch("utils.py_utils.runtime._default_runtime_factory", return_value=lambda path: runtime), \
                            patch("utils.py_utils.platforms.require_execution_target"), \
                            patch("utils.py_utils.cls_binding.read_manifest_asset_records",
                                  side_effect=AssertionError("model must not read the catalog")):
                        model = classify.ResNetClassifier(artifact, target=target, top_k=1)
                        result = model.predict(image)
                self.assertEqual(result.class_ids.tolist(), [42])
                self.assertEqual(len(calls), 1)
                tensors = calls[0][model.binding.model_name]
                shapes = sorted(tuple(t.shape) for t in tensors.values())
                expected = [(75264,)] if target == "x5" else [(1, 112, 112, 2), (1, 224, 224, 1)]
                self.assertEqual(shapes, expected)

    def test_cli_predict_schedules_prints_and_saves(self):
        import cv2
        from samples.vision.resnet.runtime.python.main import main

        for target in ("x5", "s100", "s600"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / "model.bin"
                artifact.touch()
                image = Path(directory) / "input.png"
                cv2.imwrite(str(image), np.full((60, 80, 3), 127, dtype=np.uint8))
                output = Path(directory) / "result.png"
                calls = []
                runtime = _fake_runtime(target, [_score_vector(42)], calls)
                runtime.set_scheduling_params = Mock()
                asset = ("x5:resnet:resnet18_224x224_nv12.bin" if target == "x5" else
                         f"s:resnet18:{target}/resnet18_224x224_nv12.hbm")
                stdout = io.StringIO()
                with patch("utils.py_utils.platforms.require_execution_target") as gate, \
                        patch("utils.py_utils.runtime._default_runtime_factory", return_value=lambda path: runtime), \
                        redirect_stdout(stdout):
                    status = main(["--target", target, "--asset-id", asset,
                                   "--model-path", str(artifact), "--test-img", str(image),
                                   "--top-k", "1", "--priority", "7", "--bpu-cores", "0",
                                   "--img-save-path", str(output)])
                self.assertEqual(status, 0)
                gate.assert_called_once_with(target)
                self.assertEqual(len(calls), 1)
                self.assertIn("Rank 1: class=42,", stdout.getvalue())
                self.assertEqual(cv2.imread(str(output)).shape, (60, 80, 3))
                model_name = runtime.model_names[0]
                runtime.set_scheduling_params.assert_called_once_with(
                    priority={model_name: 7}, bpu_cores={model_name: [0]})

    def test_wrong_board_is_rejected_before_sdk_import(self):
        from samples.vision.resnet.runtime.python.main import main

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.bin"
            artifact.touch()
            stderr = io.StringIO()
            with patch("utils.py_utils.platforms.require_execution_target",
                       side_effect=ValueError("board target mismatch")), \
                    patch("utils.py_utils.runtime._default_runtime_factory") as sdk, \
                    redirect_stderr(stderr):
                status = main(["--target", "x5", "--asset-id",
                               "x5:resnet:resnet18_224x224_nv12.bin", "--model-path", str(artifact)])
            self.assertEqual(status, 2)
            self.assertIn("board target mismatch", stderr.getvalue())
            sdk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
