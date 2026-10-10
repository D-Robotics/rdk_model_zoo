"""Fixed Text export tensor contract tests compiled from production sources."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"

# Both tests drive the folded production engine source (input preparation,
# tensor contract, transport and orchestration in one TU) against the host
# SDK double; the flow test adds the cache and session algorithm sources.
ENGINE_SOURCES = [
    CPP / "src/gemma4_text_engine.cpp",
    CPP / "src/gemma4_kv_cache.cpp",
    CPP / "src/gemma4_text_session.cpp",
]
CASES = {
    "text_tensor_test": ENGINE_SOURCES,
    "text_tensor_flow_test": ENGINE_SOURCES,
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
                        # text_tensor_test links the folded engine TU through
                        # the shared host SDK double in tests/native.
                        "-I",
                        str(NATIVE),
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
