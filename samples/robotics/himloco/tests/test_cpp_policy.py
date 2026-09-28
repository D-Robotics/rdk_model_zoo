"""Compile and execute the SDK-free policy contract on the host."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"


class NativePolicyTests(unittest.TestCase):
    def test_native_policy_contract(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable; native check not-run")
        with tempfile.TemporaryDirectory(prefix="himloco-policy-") as directory:
            binary = Path(directory) / "test_policy"
            build = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(CPP), str(CPP / "policy.cc"),
                 str(CPP / "tests/test_policy.cc"), "-o", str(binary)],
                capture_output=True, text=True,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(binary), str(SAMPLE / "test_data/obs_history")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("21 source observations", run.stdout)
