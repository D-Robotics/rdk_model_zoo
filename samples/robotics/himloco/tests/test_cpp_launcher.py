"""The native launcher must preview without building or loading a board SDK."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
LAUNCHER = SAMPLE / "runtime/cpp/launcher.py"


class NativeLauncherTests(unittest.TestCase):
    def test_preview_and_rejections_do_not_build(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"

            def call(*args):
                return subprocess.run(
                    [sys.executable, str(LAUNCHER), "--build-dir", str(build), *args],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )

            run = call("--target", "x5", "--dry-run")
            self.assertEqual(run.returncode, 0, run.stderr)
            preview = json.loads(run.stdout)
            self.assertEqual(preview["target"], "x5")
            self.assertFalse(preview["built"])
            self.assertEqual(preview["native_argv"][1:3], ["--target", "x5"])
            for args in [
                ("--target", "s100", "--dry-run"),
                ("--target", "auto", "--dry-run"),
                ("--target", "x5", "--warmup", "-1"),
                ("--target", "x5"),
            ]:
                self.assertEqual(call(*args).returncode, 2)
            self.assertFalse(build.exists())
            self.assertEqual(call("--help").returncode, 0)
            self.assertEqual(call("--list-models").returncode, 0)
