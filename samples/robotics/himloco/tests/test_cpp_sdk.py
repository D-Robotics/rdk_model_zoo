"""Host builds with explicit link-time doubles, never claiming board execution."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import yaml

SAMPLE = Path(__file__).resolve().parents[1]
ROOT = SAMPLE.parents[2]
CPP = SAMPLE / "runtime/cpp"
SHARED = ROOT / "samples/_shared/cpp"


class NativeSdkTests(unittest.TestCase):
    def compile_and_run(self, sources, args=(), fixtures=False):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++17 compiler missing; SDK fixture check not-run")
        with tempfile.TemporaryDirectory(prefix="himloco-sdk-") as directory:
            binary = Path(directory) / "check"
            argv = [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(CPP),
                "-I",
                str(SHARED),
            ]
            if fixtures:
                argv += ["-I", str(CPP / "tests/fixtures")]
            argv += [str(CPP / path) for path in sources] + ["-o", str(binary)]
            built = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run(
                [str(binary), *[str(Path(directory) / p) for p in args]],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("passed", run.stdout)

    def test_sdk_metadata_memory_and_failures(self):
        self.compile_and_run(
            ["sdk_runner.cc", "policy.cc", "tests/test_sdk.cc"], fixtures=True
        )

    def test_production_preflight_rejections(self):
        self.compile_and_run(
            ["model_preflight.cc", "tests/test_preflight.cc"], ["fixture.bin"]
        )

    def test_published_digest_matches_active_manifest(self):
        document = yaml.safe_load((ROOT / "docs/release/x5/models.yaml").read_text())
        # Active manifest may wrap model rows in a top-level key.
        rows = document["models"] if isinstance(document, dict) else document
        row = next(item for item in rows if item["id"] == "himloco")
        digest = row["assets"][0]["sha256"]
        source = (CPP / "model_preflight.cc").read_text()
        self.assertEqual(re.findall(r'"([a-f0-9]{64})"', source), [digest])
