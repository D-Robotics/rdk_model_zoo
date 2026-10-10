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
SHARED = ROOT / "utils/c_utils"


class NativeSdkTests(unittest.TestCase):
    def compile_and_run(self, sources, args=(), fixtures=False, enable_dnn=True):
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
                str(CPP / "inc"),
                "-I",
                str(SHARED),
            ]
            if fixtures:
                argv += [
                    "-I",
                    str(CPP / "tests/fixtures"),
                ]
                if enable_dnn:
                    argv += ["-DHIMLOCO_ENABLE_DNN=1"]
            argv += [
                str(source if source.is_absolute() else CPP / source)
                for source in map(Path, sources)
            ] + ["-o", str(binary)]
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
            ["src/policy.cpp", SHARED / "platform_identity.cc", "tests/test_sdk.cc"],
            fixtures=True,
        )

    def test_sdk_free_build_ignores_visible_fake_headers(self):
        # The DNN binding must follow the explicit HIMLOCO_ENABLE_DNN
        # definition, not header visibility: with the fake SDK headers on the
        # include path but no definition, policy.cpp stays SDK-free and links
        # without the declaration-only fixture stubs. Under the previous
        # __has_include gate this build failed to link with undefined hbDNN*
        # symbols even though HIMLOCO_BUILD_SDK was OFF.
        self.compile_and_run(
            ["src/policy.cpp", "tests/test_preflight.cc"],
            ["fixture.bin"],
            fixtures=True,
            enable_dnn=False,
        )

    def test_production_preflight_rejections(self):
        self.compile_and_run(
            ["src/policy.cpp", "tests/test_preflight.cc"], ["fixture.bin"]
        )

    def test_published_digest_matches_active_manifest(self):
        document = yaml.safe_load((ROOT / "docs/release/x5/models.yaml").read_text())
        # Active manifest may wrap model rows in a top-level key.
        rows = document["models"] if isinstance(document, dict) else document
        row = next(item for item in rows if item["id"] == "himloco")
        digest = row["assets"][0]["sha256"]
        source = (CPP / "src/policy.cpp").read_text()
        self.assertEqual(re.findall(r'"([a-f0-9]{64})"', source), [digest])
