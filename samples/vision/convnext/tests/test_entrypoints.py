"""SDK-free command checks for the ConvNeXt entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "samples" / "vision" / "convnext" / "runtime" / "python" / "main.py"

PUBLISHED_FILENAMES = (
    "ConvNeXt_atto_224x224_nv12.bin",
)


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
        self.assertIn("--variant", completed.stdout)

    def test_list_models_publishes_every_manifest_filename(self):
        completed = self._run("--list-models", "--target", "auto")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in PUBLISHED_FILENAMES:
            self.assertIn(filename, completed.stdout, filename)

    def test_dry_run_reports_source_contract_per_target_and_variant(self):
        cases = (
            ("x5", "atto", "224x224"),
        )
        for target, variant, geometry in cases:
            with self.subTest(target=target, variant=variant):
                completed = self._run(
                    "--dry-run", "--target", target, "--variant", variant
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("input_protocol: packed_nv12", completed.stdout)
                self.assertIn(f"input_geometry: {geometry}", completed.stdout)
                self.assertIn("output_score_policy: softmax", completed.stdout)
                self.assertIn("No model is downloaded", completed.stdout)
                self.assertNotIn("hbm_runtime", completed.stdout)

    def test_s_target_selection_is_an_explicit_error(self):
        completed = self._run("--dry-run", "--target", "s100", "--variant", "atto")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("error:", completed.stderr.lower())

    def test_missing_model_is_a_visible_error_without_implicit_download(self):
        completed = self._run(
            "--target", "x5",
            "--asset-id", "x5:convnext:ConvNeXt_atto_224x224_nv12.bin",
            "--model-path", str(ROOT / "missing-convnext-model.bin"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
