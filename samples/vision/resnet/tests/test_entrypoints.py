"""SDK-free command checks for the new entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "samples" / "vision" / "resnet" / "runtime" / "python" / "main.py"


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

    def test_list_and_dry_run_are_sdk_free_and_do_not_download(self):
        listed = self._run("--list-models", "--target", "x5")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("asset_id: x5:resnet:resnet18_224x224_nv12.bin", listed.stdout)

        dry_run = self._run("--dry-run", "--target", "x5")
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("No model is downloaded", dry_run.stdout)
        self.assertNotIn("hbm_runtime", dry_run.stdout)

    def test_missing_model_is_a_visible_error_without_implicit_download(self):
        completed = self._run(
            "--target", "x5", "--asset-id", "x5:resnet:resnet18_224x224_nv12.bin",
            "--model-path", str(ROOT / "missing-pilot-model.bin"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
