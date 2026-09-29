"""KV allocation and state tests with host memory replacing UCP allocation."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"


class KvCacheTests(unittest.TestCase):
    def test_allocation_state_and_prefix_retention(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; KV host checks not-run")
        with tempfile.TemporaryDirectory(prefix="gemma-kv-") as directory:
            binary = Path(directory) / "kv_cache"
            command = [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(NATIVE / "sdk_fixtures"),
                "-I",
                str(CPP / "inc"),
                str(NATIVE / "kv_cache_test.cpp"),
                str(CPP / "src/gemma4_kv_cache.cpp"),
                "-o",
                str(binary),
            ]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            for case in (
                "reset",
                "transaction",
                "null",
                "invalid",
                "append",
                "alias",
                "reallocate",
            ):
                with self.subTest(case=case):
                    run = subprocess.run(
                        [str(binary), case], capture_output=True, text=True
                    )
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
