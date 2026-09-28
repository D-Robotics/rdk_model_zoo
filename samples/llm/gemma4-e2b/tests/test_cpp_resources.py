"""Link-time SDK doubles for error cleanup; never claims a board/ABI test."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"
CASES = {
    "tensor": ["none", "alloc", "alloc_null", "zero_bytes", "input_props"],
    "constructor": [
        "none",
        "load",
        "model_null",
        "handle",
        "handle_null",
        "input_count",
        "output_count",
        "input_props",
        "output_props",
        "alloc",
        "second_alloc",
        "alloc_null",
    ],
    "all": [
        "none",
        "clean",
        "infer",
        "infer_null",
        "cores",
        "submit",
        "wait",
        "invalidate",
        "task_props",
        "release",
    ],
    "selective": [
        "none",
        "clean",
        "infer",
        "infer_null",
        "cores",
        "submit",
        "wait",
        "invalidate",
        "task_props",
        "release",
    ],
}


class SdkResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest(
                "C++17 compiler unavailable; resource fixture not-run"
            )
        cls.directory = tempfile.TemporaryDirectory(prefix="gemma-sdk-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binaries = {}
        for target in ("s100", "s600"):
            binary = Path(cls.directory.name) / target
            argv = [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DSOC_" + target.upper(),
                "-I",
                str(NATIVE / "sdk_fixtures"),
                "-I",
                str(CPP / "inc"),
                str(NATIVE / "sdk_resources_test.cpp"),
                str(CPP / "src/gemma4_vision_engine.cpp"),
                "-o",
                str(binary),
            ]
            if sys.platform.startswith("linux"):
                argv.append("-ldl")
            built = subprocess.run(argv, capture_output=True, text=True)
            if built.returncode:
                raise AssertionError(built.stdout + built.stderr)
            cls.binaries[target] = binary

    def check(self, action):
        env = os.environ.copy()
        env.pop("GEMMA4_USE_DNN_V3", None)
        for target, binary in self.binaries.items():
            for failure in CASES[action]:
                if target == "s100" and failure == "cores":
                    continue  # source S100 branch does not query compiled core count
                with self.subTest(target=target, failure=failure):
                    run = subprocess.run(
                        [str(binary), action, failure],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("passed", run.stdout)

    def test_tensor_allocation_ownership(self):
        self.check("tensor")

    def test_vision_constructor_ownership(self):
        self.check("constructor")

    def test_full_inference_task_ownership(self):
        self.check("all")

    def test_selective_inference_task_ownership(self):
        self.check("selective")


if __name__ == "__main__":
    unittest.main()
