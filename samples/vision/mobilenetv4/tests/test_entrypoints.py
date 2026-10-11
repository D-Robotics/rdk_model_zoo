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
        from testsupport import PUBLISHED

        completed = self._run("--list-models", "--target", "auto")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in PUBLISHED.values():
            self.assertIn(filename, completed.stdout)

    def test_dry_run_per_target_reports_source_contract(self):
        from testsupport import PUBLISHED, VARIANTS

        for (variant, target) in PUBLISHED:
            size, shorter, _ = VARIANTS[variant]
            self._check_dry_run(target, variant, {
                'protocol': 'packed_nv12' if target == 'x5' else 'split_nv12',
                'geometry': f'{size}x{size}', 'policy': 'softmax',
                'preprocess': f'resize_type=2, resize_shorter={shorter}'})

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
            "--target", "s600",
            "--asset-id", "s:mobilenetv4:s600/mobilenetv4_conv_large_nashp_256x256_nv12.hbm",
            "--model-path", str(ROOT / "missing-mobilenetv4-model.hbm"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
