"""Text stage/session refactor tests compiled from production sources.

Host SDK doubles only: no vendor SDK, weights, or board. The README stage
example is compiled and executed so the documented code and outputs stay
runnable.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"

# The folded text engine TU carries input preparation, the tensor contract
# and SDK transport alongside the orchestration; the session policy and KV
# cache remain independent algorithm sources.
STAGE_SOURCES = [
    CPP / "src/gemma4_text_engine.cpp",
    CPP / "src/gemma4_text_session.cpp",
    CPP / "src/gemma4_kv_cache.cpp",
]

CASES = {
    "text_session_test": [CPP / "src/gemma4_text_session.cpp"],
    "text_stages_test": STAGE_SOURCES,
    "text_engine_flow_test": STAGE_SOURCES,
    "readme_text_stages_example": STAGE_SOURCES,
}

EXPECTED_OUTPUT = {
    "readme_text_stages_example": (
        "stage first token: 104\n"
        "session out: 11 22 33 44 55 104 100\n"
        "session processed: 7\n"
    ),
}


class TextStageRefactorTests(unittest.TestCase):
    def test_stages_session_flow_and_readme_example(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; Text stage checks not-run")
        with tempfile.TemporaryDirectory(prefix="gemma-text-stages-") as directory:
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
                    expected = EXPECTED_OUTPUT.get(name)
                    if expected is not None:
                        # The README example asserts its documented output.
                        self.assertEqual(run.stdout, expected)
                    else:
                        # Observable completion: each native test ends with
                        # its own "... passed" summary line.
                        self.assertRegex(run.stdout, r"passed\s*$")


if __name__ == "__main__":
    unittest.main()
