"""Preparation works on hosts; inference cannot use target flags as evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / 'runtime/python/main.py'


class ExecutionTargetTests(unittest.TestCase):
    def test_legacy_yolo26_library_imports_outside_checkout(self):
        root = ENTRY.parents[5]
        entry = root / 'platforms/s/samples/vision/ultralytics_yolo26/runtime/python/yolo26_det.py'
        code = ('import importlib.util,sys; '
                f's=importlib.util.spec_from_file_location("legacy", {str(entry)!r}); '
                'm=importlib.util.module_from_spec(s); sys.modules[s.name]=m; '
                's.loader.exec_module(m); from unittest.mock import patch; '
                'p=patch("samples._shared.platforms.detect_target", return_value="s100"); '
                'p.start(); m.YOLO26DetectConfig(model_path="model.hbm")')
        result = subprocess.run([sys.executable, '-c', code], cwd=tempfile.gettempdir(),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_url_comes_from_manifest_not_filename_formula(self):
        from samples._shared import assets
        sys.path.insert(0, str(ENTRY.parent))
        from yolo_assets import model_url
        from yolo_platform import resolve_platform
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'platforms/x5/docs/release/models.yaml'
            path.parent.mkdir(parents=True)
            path.write_text('models:\n- id: ultralytics_yolo\n  assets:\n  - filename: yolov8n_detect_bayese_640x640_nv12.bin\n    format: bin\n    url: https://example.invalid/revised-model.bin\n    sha256: null\n', encoding='utf-8')
            assets._models.cache_clear()
            try:
                with patch.object(assets, '_ROOT', Path(directory)):
                    self.assertEqual(model_url(resolve_platform('x5'), 'yolov8', 'detect', 'n'),
                                     'https://example.invalid/revised-model.bin')
            finally:
                assets._models.cache_clear()

    def test_target_alias_works_for_offline_plan(self):
        result = subprocess.run([sys.executable, str(ENTRY), '--target', 's600', '--dry-run'],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('s600', result.stdout)

    def test_qualified_asset_selects_exact_model_and_rejects_wrong_target(self):
        reference = 'x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin'
        for target, expected in [('x5', 0), ('s600', 2)]:
            result = subprocess.run([sys.executable, str(ENTRY), '--target', target,
                                     '--asset-id', reference, '--dry-run'],
                                    cwd=tempfile.gettempdir(), capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
            if expected == 0:
                self.assertIn(reference, result.stdout)
                self.assertIn('yolov8', result.stdout)

    def test_host_inference_rejected_before_download(self):
        spec = importlib.util.spec_from_file_location('yolo_target_entry', ENTRY)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with patch.object(sys, 'argv', [str(ENTRY), '--platform', 'x5']):
            with patch('samples._shared.platforms.detect_target', return_value=None):
                with patch.object(entry, 'ensure_model', side_effect=AssertionError('download before identity')):
                    self.assertEqual(entry.main(), 2)


if __name__ == '__main__':
    unittest.main()
