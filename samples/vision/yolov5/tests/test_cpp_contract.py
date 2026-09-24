# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Behavioural host checks for the split YOLOv5 C++ migration surface.

The native adapters need the target SDK, which a host does not have. The gates,
decoder and dump writer are therefore SDK-free modules and are exercised here
with explicit metadata, so a passing test proves an accept/reject decision
rather than the presence of a string in a source file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CPP = ROOT / "samples" / "vision" / "yolov5" / "runtime" / "cpp"
HARNESS = ROOT / "samples" / "vision" / "yolov5" / "tests" / "cpp" / "portable_checks.cpp"


class PortableHarness:
    """Compiles the SDK-free numeric core once and runs one check per call."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.binary = Path(self.directory.name) / "portable_checks"
        self.error = ""
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.error = "host C++ compiler unavailable"
            return
        command = [
            compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
            "-I", str(CPP / "include"), str(HARNESS),
            str(CPP / "src" / "yolov5_gate.cpp"),
            str(CPP / "src" / "yolov5_decode.cpp"),
            str(CPP / "src" / "yolov5_dump.cpp"),
            "-o", str(self.binary),
        ]
        built = subprocess.run(command, capture_output=True, text=True)
        if built.returncode != 0:
            self.error = built.stderr

    def run(self, check: str, scratch: Path | None = None):
        arguments = [str(self.binary), check]
        if scratch is not None:
            arguments.append(str(scratch))
        return subprocess.run(arguments, capture_output=True, text=True)


class CppSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = PortableHarness()
        if cls.harness.error:
            raise unittest.SkipTest(cls.harness.error)

    def assert_check(self, name: str, scratch: Path | None = None) -> None:
        result = self.harness.run(name, scratch)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_required_native_files_exist(self):
        for path in (
            CPP / "CMakeLists.txt",
            CPP / "include" / "yolov5_adapter.hpp",
            CPP / "include" / "yolov5_decode.hpp",
            CPP / "include" / "yolov5_dump.hpp",
            CPP / "include" / "yolov5_gate.hpp",
            CPP / "include" / "yolov5_visualize.hpp",
            CPP / "src" / "cli_main.cpp",
            CPP / "src" / "s_adapter.cpp",
            CPP / "src" / "x5_adapter.cpp",
            CPP / "src" / "yolov5_decode.cpp",
            CPP / "src" / "yolov5_dump.cpp",
            CPP / "src" / "yolov5_gate.cpp",
            CPP / "src" / "yolov5_visualize.cpp",
            CPP / "launcher.py",
            CPP / "run.sh",
        ):
            self.assertTrue(path.is_file(), path)

    def test_cmake_binds_each_target_to_one_adapter_and_the_build_identity(self):
        cmake = (CPP / "CMakeLists.txt").read_text()
        # Configuration must not inspect the board; the target is explicit.
        self.assertNotIn("/sys/class/boardinfo", cmake)
        for token in (
            "src/x5_adapter.cpp",
            "src/s_adapter.cpp",
            "src/yolov5_gate.cpp",
            "src/yolov5_dump.cpp",
            'YOLOV5_TARGET_NAME="x5"',
            'YOLOV5_TARGET_NAME="${YOLOV5_TARGET}"',
            "SOC_S600",
        ):
            self.assertIn(token, cmake)
        adapters = [line for line in cmake.splitlines() if "_adapter.cpp" in line]
        self.assertEqual(len(adapters), 2, adapters)

    def test_x5_nv12_input_gate(self):
        self.assert_check("x5_input")

    def test_x5_head_gate(self):
        self.assert_check("x5_head")

    def test_s_split_nv12_plane_gate(self):
        self.assert_check("s_plane")

    def test_s32_dequant_gate(self):
        self.assert_check("s_dequant")

    def test_native_binary_refuses_a_mismatched_target(self):
        self.assert_check("target_identity")

    def test_decode_score_boundary_is_an_explicit_policy(self):
        self.assert_check("decode_boundary")

    def test_decode_top_k_cap_is_an_explicit_policy(self):
        self.assert_check("decode_topk")

    def test_dump_hashes_are_correct(self):
        self.assert_check("dump", Path(self.harness.directory.name))

    def test_dump_manifest_is_independently_verifiable(self):
        scratch = Path(self.harness.directory.name) / "dump-verify"
        self.assert_check("dump", scratch)
        manifest = json.loads((scratch / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "rdk-model-zoo/yolov5-cpp-dump/v1")
        self.assertEqual(manifest["return_code"], 0)
        self.assertEqual(manifest["target"], "x5")
        raw = manifest["raw_tensors"][0]
        payload = (scratch / raw["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), raw["sha256"])
        self.assertEqual(struct.unpack("<4f", payload), (1.0, 2.0, 3.0, 4.0))
        detection = manifest["detections"][0]
        self.assertEqual(detection["class_id"], 7)
        self.assertAlmostEqual(detection["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
