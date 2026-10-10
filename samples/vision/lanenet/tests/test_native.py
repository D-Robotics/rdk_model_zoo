"""Host native contract builds; no vendor SDK/OpenCV image pipeline evidence.

The consolidated production sources (src/segment.cpp, src/cli.cpp) include the
OpenCV headers, so every native test compiles them against the host OpenCV
(pkg-config opencv5/opencv4) and the checked-in fake SDK headers; the pure
contract and CLI tests link the link-only stubs in tests/fake_sdk/stub_impl.cpp,
while the resource test defines its own fault-injecting SDK implementation.
"""

from pathlib import Path
import subprocess, tempfile, unittest

SAMPLE = Path(__file__).resolve().parents[1]


def opencv_flags():
    for name in ("opencv5", "opencv4"):
        probe = subprocess.run(
            ["pkg-config", "--cflags", "--libs", name],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return probe.stdout.split()
    raise RuntimeError(
        "pkg-config cannot locate OpenCV (opencv5/opencv4) for the native tests"
    )


class NativeTests(unittest.TestCase):
    def test_native_layout_roles_numeric_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "contract"
            p = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(SAMPLE / "runtime/cpp/inc"),
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    str(SAMPLE / "tests/test_native_contract.cpp"),
                    str(SAMPLE / "runtime/cpp/src/segment.cpp"),
                    str(SAMPLE / "tests/fake_sdk/stub_impl.cpp"),
                    *opencv_flags(),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_actual_sdk_owner_failure_cleanup_with_fake_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "resources"
            p = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    "-I",
                    str(SAMPLE / "runtime/cpp/inc"),
                    str(SAMPLE / "tests/test_resources.cpp"),
                    str(SAMPLE / "runtime/cpp/src/segment.cpp"),
                    *opencv_flags(),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            r = subprocess.run(
                [str(exe), str(Path(tmp) / "fixture.hbm")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_native_cli_and_exact_typed_numpy_serialization(self):
        import json
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "io"
            p = subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(SAMPLE / "runtime/cpp/inc"),
                    "-I",
                    str(SAMPLE / "tests/fake_sdk"),
                    str(SAMPLE / "tests/test_native_io.cpp"),
                    str(SAMPLE / "runtime/cpp/src/cli.cpp"),
                    str(SAMPLE / "runtime/cpp/src/segment.cpp"),
                    str(SAMPLE / "tests/fake_sdk/stub_impl.cpp"),
                    *opencv_flags(),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            r = subprocess.run([str(exe), tmp], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            np.testing.assert_array_equal(
                np.load(Path(tmp) / "integers.npy"),
                np.array([[0, 1], [9007199254740993, -2]], np.int64),
            )
            np.testing.assert_array_equal(
                np.load(Path(tmp) / "floats.npy"),
                np.array([-0.5, 0.1, 1.2], np.float32),
            )
            self.assertEqual(np.load(Path(tmp) / "labels.npy").dtype, np.uint8)
            self.assertEqual(
                json.loads((Path(tmp) / "escaped.json").read_text())["value"],
                'a"\\\n\t',
            )


if __name__ == "__main__":
    unittest.main()
