"""Compile production native sources on host; this is not an SDK/board build."""

from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/yolo26_depth"
CPP = SAMPLE / "runtime/cpp"


def opencv_flags():
    import subprocess as sp
    for pkg in ('opencv5', 'opencv4'):
        if sp.run(['pkg-config', '--exists', pkg]).returncode == 0:
            cflags = sp.run(['pkg-config', '--cflags', pkg], capture_output=True, text=True).stdout.split()
            libs = sp.run(['pkg-config', '--libs', pkg], capture_output=True, text=True).stdout.split()
            return cflags, libs
    raise RuntimeError('host tests need opencv5 or opencv4 via pkg-config')


class NativeTests(unittest.TestCase):
    def test_native_tensor_geometry_and_identity_contract(self):
        cflags, libs = opencv_flags()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "contract"
            result = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *cflags,
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    "-I",
                    str(CPP / "inc"),
                    str(SAMPLE / "tests/test_native_contract.cpp"),
                    str(CPP / "src/depth.cpp"),
                    str(SAMPLE / "tests/fake_sdk/stub_impl.cpp"),
                    *libs,
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_actual_runner_resource_cleanup_against_fake_sdk(self):
        cflags, libs = opencv_flags()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "resources"
            result = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *cflags,
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    "-I",
                    str(CPP / "inc"),
                    str(SAMPLE / "tests/test_resources.cpp"),
                    str(CPP / "src/depth.cpp"),
                    *libs,
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run(
                [str(executable), str(Path(tmp) / "fixture.bin")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_native_cli_and_numpy_serialization(self):
        import json
        import numpy as np

        cflags, libs = opencv_flags()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "io"
            result = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *cflags,
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    "-I",
                    str(CPP / "inc"),
                    str(SAMPLE / "tests/test_native_io.cpp"),
                    str(CPP / "src/cli.cpp"),
                    str(CPP / "src/depth.cpp"),
                    str(SAMPLE / "tests/fake_sdk/stub_impl.cpp"),
                    *libs,
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            prefix = Path(tmp) / "array"
            run = subprocess.run(
                [str(executable), str(prefix)], capture_output=True, text=True
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("quoted", json.loads(run.stdout))
            np.testing.assert_array_equal(
                np.load(prefix.with_suffix(".npy")),
                np.arange(1, 7, dtype=np.float32).reshape(2, 3),
            )
            np.testing.assert_array_equal(
                np.fromfile(prefix.with_suffix(".f32"), dtype="<f4"),
                np.arange(1, 7, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
