# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the split YOLOv5 C++ migration surface."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CPP = ROOT / "samples" / "vision" / "yolov5" / "runtime" / "cpp"


class CppContractTests(unittest.TestCase):
    def test_target_adapters_and_launcher_are_separate_from_numeric_core(self):
        for path in (CPP / "CMakeLists.txt", CPP / "src" / "cli_main.cpp", CPP / "src" / "x5_adapter.cpp", CPP / "src" / "s_adapter.cpp", CPP / "src" / "yolov5_decode.cpp", CPP / "src" / "yolov5_visualize.cpp", CPP / "include" / "yolov5_decode.hpp", CPP / "include" / "yolov5_visualize.hpp", CPP / "run.sh"):
            self.assertTrue(path.is_file(), path)
        cmake = (CPP / "CMakeLists.txt").read_text()
        self.assertNotIn("/sys/class/boardinfo", cmake)
        self.assertNotIn("../../models", (CPP / "src" / "x5_adapter.cpp").read_text())
        self.assertIn("YOLOV5_TARGET", cmake)
        self.assertIn("x5_adapter.cpp", cmake)
        self.assertIn("s_adapter.cpp", cmake)
        self.assertIn("SOC_S600", cmake)
        self.assertIn("nn_math.cpp", cmake)
        self.assertIn("OpenCV_INCLUDE_DIRS", cmake)

    def test_source_safety_invariants_are_explicit(self):
        x5 = (CPP / "src" / "x5_adapter.cpp").read_text()
        s = (CPP / "src" / "s_adapter.cpp").read_text()
        main = (CPP / "src" / "cli_main.cpp").read_text()
        self.assertIn("input_count != 1", x5)
        self.assertIn("output_count != 3", x5)
        self.assertRegex(x5, r"3\s*\*\s*\(5\s*\+\s*classes")
        self.assertIn("std::set<int>", x5)
        self.assertIn("HB_UCP_BPU_CORE_ANY", s)
        self.assertNotIn("param_to_use->priority = 0", s)
        self.assertIn("dequantizeTensorS32", s)
        self.assertNotIn("std::vector<hbDNNTensor>*", s)
        self.assertIn("HB_DNN_TENSOR_TYPE_F32", x5)
        self.assertIn("HB_DNN_IMG_TYPE_NV12", x5)
        self.assertIn("hbSysAllocCachedMem input failed", x5)
        self.assertIn("letterbox_to_nv12", x5)
        self.assertIn("schedule.priority = options.priority", s)
        self.assertIn("load_labels(options.label_path)", s)
        self.assertIn("load_labels(options.label_path)", x5)
        self.assertIn("--asset-id", main)
        self.assertIn("--target", main)
        self.assertIn("require_execution_target", (CPP / "launcher.py").read_text())
        self.assertIn("verify_asset_file", (CPP / "launcher.py").read_text())

    def test_launcher_uses_exact_python_binding_and_no_download(self):
        text = (CPP / "launcher.py").read_text()
        self.assertIn("samples.vision.yolov5.runtime.python.model_binding", text)
        self.assertIn("asset_id", text)
        self.assertIn("model_path", text)
        self.assertNotIn("download", text.lower())

    def test_pure_decode_compiles_and_rejects_ambiguous_head_shapes(self):
        compiler = subprocess.run(["c++", "--version"], capture_output=True, text=True)
        if compiler.returncode != 0:
            self.skipTest("host C++ compiler unavailable")
        source = r'''
#include "yolov5_decode.hpp"
#include <iostream>
int main() {
  std::vector<yolov5::HeadShape> good{{80,80,255},{40,40,255},{20,20,255}};
  std::vector<yolov5::HeadShape> bad{{80,80,255},{80,80,255},{20,20,255}};
  if (!yolov5::validate_head_shapes(good,640,80)) return 1;
  if (yolov5::validate_head_shapes(bad,640,80)) return 2;
  std::vector<yolov5::HeadShape> two_class{{8,8,21},{4,4,21},{2,2,21}};
  std::vector<std::vector<float>> raw{{8.0F,0.0F,0.0F,0.0F,8.0F,8.0F,7.0F},
                                      std::vector<float>(4*4*21, 0.0F),
                                      std::vector<float>(2*2*21, 0.0F)};
  raw[0].resize(8*8*21, 0.0F);
  raw[0][4] = 8.0F; raw[0][5] = 8.0F; raw[0][6] = 7.0F;
  std::array<float,18> anchors{};
  anchors.fill(10.0F);
  auto detections = yolov5::decode_heads(raw, two_class, 64, 2, 0.9F, 0.45F, anchors);
  if (detections.size() != 1 || detections[0].class_id != 0) return 3;
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "check.cpp"
            binary = Path(directory) / "check"
            src.write_text(source)
            result = subprocess.run(["c++", "-std=c++17", "-I", str(CPP / "include"), str(src), str(CPP / "src" / "yolov5_decode.cpp"), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(subprocess.run([str(binary)]).returncode, 0)

    def test_cpp_readmes_document_target_and_source_differences(self):
        for name in ("README.md", "README_cn.md"):
            text = (CPP / name).read_text()
            self.assertIn("--asset-id", text)
            self.assertIn("letterbox", text.lower())
            self.assertIn("stretch", text.lower())
            self.assertIn("not-run", text)
            self.assertIn("priority", text.lower())


if __name__ == "__main__":
    unittest.main()
