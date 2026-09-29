"""Real package preparation, substituting only HTTP response bytes."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "samples/speech/paraformer/model/download.py"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            SCRIPT.is_file(), "Missing Paraformer model package preparation"
        )
        spec = importlib.util.spec_from_file_location("paraformer_download", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_preview_has_six_entries_and_never_creates_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            run = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", "--output-dir", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(run.stdout.splitlines()), 6)
            self.assertFalse(output.exists())

    def test_package_download_and_local_files_are_complete_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = []

            def response(url, **kwargs):
                requests.append(url)
                if url.endswith("tokens.json"):
                    return io.BytesIO(
                        (
                            ROOT
                            / "docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json"
                        ).read_bytes()
                    )
                return io.BytesIO(b"synthetic-host-test-model-not-HBM")

            with patch("urllib.request.urlopen", response), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(self.module.main(["--output-dir", directory]), 0)
                self.assertEqual(self.module.main(["--output-dir", directory]), 0)
            files = tuple((Path(directory) / "s100").iterdir())
            self.assertEqual(len(files), 6)
            self.assertEqual(len(requests), 4)
            for name in ("am.mvn", "paraformer_config.yaml"):
                self.assertEqual(
                    (Path(directory) / "s100" / name).read_bytes(),
                    (SCRIPT.parent / name).read_bytes(),
                )
            protected = Path(directory) / "s100/am.mvn"
            protected.write_bytes(b"user-changed")
            with patch("urllib.request.urlopen", response), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.module.main(["--output-dir", directory]), 2)
            self.assertEqual(protected.read_bytes(), b"user-changed")


if __name__ == "__main__":
    unittest.main()
