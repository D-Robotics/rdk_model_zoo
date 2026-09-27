"""Exporter help and invalid inputs must not import a checkpoint or download assets."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from samples.vision.yoloe.conversion.export import export_checkpoint


class ExportEntryTests(unittest.TestCase):
    def test_help_runs_without_site_packages(self):
        root = Path(__file__).resolve().parents[5]
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "samples/vision/yoloe/conversion/export.py",
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--variant", result.stdout)

    def test_invalid_paths_and_options_fail_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(
                weights=root / "missing.pt", variant="11s", output_dir=root / "export"
            )
            with self.assertRaises(FileNotFoundError):
                export_checkpoint(**options)
            self.assertFalse((root / "export").exists())
            for override in ({"variant": "11n"}, {"threads": 0}, {"threads": True}):
                with self.assertRaises(ValueError):
                    export_checkpoint(**(options | override))
            (root / "checkpoint.pt").write_bytes(b"not-loaded")
            (root / "export").mkdir()
            with self.assertRaises(FileExistsError):
                export_checkpoint(**(options | {"weights": root / "checkpoint.pt"}))
