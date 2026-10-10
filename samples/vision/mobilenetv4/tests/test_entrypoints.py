"""SDK-free command checks for the MobileNetV4 entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "samples" / "vision" / "mobilenetv4" / "runtime" / "python" / "main.py"


class EntrypointTests(unittest.TestCase):
    def _run(self, *args):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(MAIN), *args],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )

    def test_help_does_not_need_board_sdk(self):
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--target", completed.stdout)

    def test_list_models_publishes_every_manifest_filename(self):
        completed = self._run("--list-models", "--target", "auto")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in (
            'mobilenetv4_conv_small_bayese_224x224_nv12.bin',
            'mobilenetv4_conv_medium_bayese_224x224_nv12.bin',
            's100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm',
            's100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm',
            's100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm',
            's100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm',
            's600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm',
            's600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm',
        ):
            self.assertIn(filename, completed.stdout)

    def test_dry_run_per_target_reports_source_contract(self):
        shorter = {'small': 256, 'medium': 235}
        for variant in ('small', 'medium'):
            for target in ('x5', 's100', 's100p', 's600'):
                self._check_dry_run(target, variant, {
                    'protocol': 'packed_nv12' if target == 'x5' else 'split_nv12',
                    'geometry': '224x224', 'policy': 'softmax',
                    'preprocess': f'resize_type=2, resize_shorter={shorter[variant]}'})

    def _check_dry_run(self, target, variant, expected):
        completed = self._run("--dry-run", "--target", target, "--variant", variant)
        label = f"{variant}/{target}"
        self.assertEqual(completed.returncode, 0, f"{label}: {completed.stderr}")
        self.assertIn(f"input_protocol: {expected['protocol']}", completed.stdout, label)
        self.assertIn(f"input_geometry: {expected['geometry']}", completed.stdout, label)
        self.assertIn(f"output_score_policy: {expected['policy']}", completed.stdout, label)
        self.assertIn(f"preprocess: {expected['preprocess']}", completed.stdout, label)
        self.assertIn("No model is downloaded", completed.stdout, label)
        self.assertNotIn("hbm_runtime", completed.stdout, label)

    def test_missing_model_is_a_visible_error_without_implicit_download(self):
        completed = self._run(
            "--target", "x5",
            "--asset-id", "x5:mobilenetv4:mobilenetv4_conv_medium_bayese_224x224_nv12.bin",
            "--model-path", str(ROOT / "missing-mobilenetv4-model.bin"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
