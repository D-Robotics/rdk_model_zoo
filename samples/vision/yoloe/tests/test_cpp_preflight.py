"""Compare shared native identity and hashing with the Python authorities."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from utils.py_utils import platforms

ROOT = Path(__file__).resolve().parents[4]


class NativePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("Native compiler required")
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.probe = cls.directory / "identity_probe"
        subprocess.run([
            compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-I", str(ROOT / "utils/c_utils"),
            str(ROOT / "samples/vision/yoloe/runtime/cpp/tests/identity_probe.cc"),
            "-o", str(cls.probe),
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe_identity(self, values):
        return subprocess.check_output([str(self.probe), *values], text=True).strip() or None

    def python_identity(self, values):
        paths = [self.directory / name for name in ("soc", "board", "socinfo", "tree")]
        for path, value in zip(paths, values):
            path.write_text(value)
        with patch.multiple(platforms, SOC_NAME_PATH=paths[0], BOARD_TYPE_PATH=paths[1],
                            SOCINFO_NAME_PATH=paths[2], DEVICE_TREE_MODEL_PATH=paths[3]):
            return platforms.detect_target()

    def test_registry_aliases_and_precedence_match_python(self):
        registry = json.loads((ROOT / "docs/release/platforms.json").read_text())
        cases = []
        for target in registry["targets"]:
            for soc in target["soc_names"]:
                cases.extend([(soc, "", "", ""), (" " + soc.upper() + "\n", "", "", "")])
            for board in target.get("board_types", ()):
                cases.append((target["base_soc"], board, "", ""))
            for socinfo in target.get("socinfo_names", ()):
                cases.append(("", "", socinfo, ""))
            for tree in target.get("device_tree_models", ()):
                cases.append(("", "", "", tree))
        cases.extend([
            ("unknown", "", "x5u", "D-Robotics RDK X5 V1.0"),
            ("", "", "unknown", "D-Robotics RDK X5 V1.0"),
            ("s100", "unknown", "x5u", ""),
            ("s600", "s100p", "", ""), ("x3", "", "", ""),
            ("", "", "", "d-robotics rdk x5 v1.0"), ("", "", "", ""),
        ])
        for case in cases:
            with self.subTest(identity=case):
                self.assertEqual(self.probe_identity(case), self.python_identity(case))

    def test_streaming_hash_matches_hashlib_at_padding_and_read_boundaries(self):
        path = self.directory / "bytes"
        for length in (0, 1, 55, 56, 63, 64, 65, 65535, 65536, 65537, 1048577):
            data = bytes((i * 37 + 11) % 256 for i in range(length))
            path.write_bytes(data)
            actual = subprocess.check_output([str(self.probe), "sha256", str(path)], text=True).strip()
            self.assertEqual(actual, hashlib.sha256(data).hexdigest(), length)
        for invalid in (self.directory / "missing", self.directory):
            result = subprocess.run([str(self.probe), "sha256", str(invalid)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
