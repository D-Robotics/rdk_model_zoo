"""Constructor resource checks without SDK, model weights, or inference."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"


class TextOwnershipTests(unittest.TestCase):
    def test_constructor_and_tensor_ownership(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; Text ownership not-run")
        with tempfile.TemporaryDirectory(prefix="gemma-text-owners-") as directory:
            for name in ("model_io", "text_resources"):
                with self.subTest(name=name):
                    binary = Path(directory) / name
                    command = [
                        compiler,
                        "-std=c++17",
                        "-I",
                        str(NATIVE / "sdk_fixtures"),
                        "-I",
                        str(CPP / "inc"),
                        str(NATIVE / f"{name}_test.cpp"),
                    ]
                    if name == "text_resources":
                        command += [
                            str(CPP / "src/gemma4_text_engine.cpp"),
                            str(CPP / "src/gemma4_kv_cache.cpp"),
                        ]
                    command += ["-o", str(binary)]
                    built = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    run = subprocess.run([str(binary)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
