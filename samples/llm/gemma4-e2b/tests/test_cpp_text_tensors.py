"""Fixed Text export tensor contract tests compiled from production sources."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"

# The helper test covers the physical contract alone; the flow test drives
# the production engine, cache, and helper against a host SDK double.
CASES = {
    "text_tensor_test": [CPP / "src/gemma4_text_tensor.cpp"],
    "text_tensor_flow_test": [
        CPP / "src/gemma4_text_tensor.cpp",
        CPP / "src/gemma4_kv_cache.cpp",
        CPP / "src/gemma4_text_engine.cpp",
    ],
}


class TextTensorContractTests(unittest.TestCase):
    def test_contract_and_generation_flow(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; Text tensor checks not-run")
        with tempfile.TemporaryDirectory(prefix="gemma-text-tensor-") as directory:
            for name, sources in CASES.items():
                with self.subTest(name=name):
                    binary = Path(directory) / name
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
                        str(NATIVE / f"{name}.cpp"),
                        *[str(source) for source in sources],
                        "-ldl",
                        "-o",
                        str(binary),
                    ]
                    built = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    run = subprocess.run([str(binary)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
