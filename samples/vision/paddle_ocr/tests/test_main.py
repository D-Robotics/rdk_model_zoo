"""SDK-free entrypoint checks for arbitrary working directories."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_MAIN = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "python"
    / "main.py"
)


class MainTests(unittest.TestCase):
    def _run(self, *arguments: str):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as current_directory:
            return subprocess.run(
                [sys.executable, str(_MAIN), *arguments],
                cwd=current_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_help_is_available_without_runtime_modules(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--det-asset-id", result.stdout)
        self.assertIn("--prepare", result.stdout)

    def test_list_and_dry_run_are_sdk_free(self):
        listed = self._run("--list-models", "--target", "auto")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("x5:paddleocr:", listed.stdout)
        self.assertIn("s:paddle_ocr:", listed.stdout)

        dry = self._run("--dry-run", "--target", "x5")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("No model is downloaded", dry.stdout)
        self.assertIn("en_PP-OCRv3_det_infer-deploy_640x640_nv12", dry.stdout)

    def test_unknown_published_target_fails_without_fallback(self):
        result = self._run("--dry-run", "--target", "s100p")
        self.assertEqual(result.returncode, 2)
        self.assertIn("No audited PaddleOCR pair", result.stderr)

    def test_execution_auto_resolves_detected_target_before_default_pair(self):
        from unittest.mock import patch

        from samples.vision.paddle_ocr.runtime.python import main as entry

        args = entry.build_parser().parse_args(["--target", "auto"])
        with patch(
            "samples._shared.platforms.resolve_target", return_value="x5"
        ) as resolve_target:
            pair = entry._resolve_pair_from_args(args, for_execution=True)
        self.assertEqual(pair.target, "x5")
        resolve_target.assert_called_once_with("auto")

    def test_prepare_rejects_destination_collision_after_model_dir_rewrite(self):
        from samples.vision.paddle_ocr.runtime.python import main as entry

        args = entry.build_parser().parse_args(
            [
                "--prepare",
                "--target",
                "x5",
                "--det-asset-id",
                "x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin",
                "--rec-asset-id",
                "x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin",
                "--det-model-path",
                "det.bin",
                "--rec-model-path",
                "det.bin",
                "--model-dir",
                "collision-test",
            ]
        )
        with self.assertRaises(entry.BindingError):
            entry._prepare(args)


if __name__ == "__main__":
    unittest.main()
