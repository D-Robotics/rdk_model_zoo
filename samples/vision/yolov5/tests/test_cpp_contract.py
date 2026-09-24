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
C_UTILS = ROOT / "utils" / "c_utils" / "inc"

# Minimal SDK-shape stubs derived from the on-board header evidence captured
# 2026-09-24 (docs/releases/unified-migration/evidence/2026-09-24-b7-native-sdk-preflight.json).
# They encode the differences that broke the real S100 build: the S100
# hbDNNTensorProperties has no alignedShape, hbDNNQuantiType has no SHIFT,
# stride/alignedByteSize are int64, sysMem is a single hbUCPSysMem, and there
# is no hb_dnn_ext.h; the X5 header has all of those plus sysMem[4]. A real
# board re-verification is still required; these stubs only keep each adapter
# honest about its own SDK's shape on a host.
S100_STUBS = {
    "hobot/hb_ucp.h": """\
#pragma once
#include <stdint.h>
typedef struct hbUCPSysMem { void *virAddr; uint64_t phyAddr; uint32_t memSize; } hbUCPSysMem;
typedef void *hbUCPTaskHandle_t;
typedef struct hbUCPSchedParam { uint64_t backend; int32_t priority; } hbUCPSchedParam;
// Real on-board definitions (2026-09-24 evidence): the scheduler backend is a
// bitmask, CORE_0..3 at 1ULL<<0..3 and ANY at 1ULL<<7.
#define HB_UCP_BPU_CORE_0 (1ULL << 0)
#define HB_UCP_BPU_CORE_1 (1ULL << 1)
#define HB_UCP_BPU_CORE_2 (1ULL << 2)
#define HB_UCP_BPU_CORE_3 (1ULL << 3)
#define HB_UCP_BPU_CORE_ANY (1ULL << 7)
enum { HB_SYS_MEM_CACHE_CLEAN = 1, HB_SYS_MEM_CACHE_INVALIDATE = 2 };
#define HB_UCP_INITIALIZE_SCHED_PARAM(param) \\
  do { (param)->backend = HB_UCP_BPU_CORE_ANY; (param)->priority = 0; } while (0)
int32_t hbUCPFree(hbUCPSysMem *mem);
int32_t hbUCPMallocCached(hbUCPSysMem *mem, uint32_t size, uint32_t align);
int32_t hbUCPMemFlush(hbUCPSysMem *mem, int type);
int32_t hbUCPSubmitTask(hbUCPTaskHandle_t task, hbUCPSchedParam *sched_param);
int32_t hbUCPWaitTaskDone(hbUCPTaskHandle_t task, int32_t timeout);
int32_t hbUCPReleaseTask(hbUCPTaskHandle_t task);
""",
    "hobot/dnn/hb_dnn.h": """\
#pragma once
#include <stdint.h>
#include "../hb_ucp.h"
#define HB_DNN_TENSOR_MAX_DIMENSIONS 8
typedef enum {
  HB_DNN_TENSOR_TYPE_S8 = 0, HB_DNN_TENSOR_TYPE_U8, HB_DNN_TENSOR_TYPE_S16,
  HB_DNN_TENSOR_TYPE_F32, HB_DNN_TENSOR_TYPE_S32, HB_DNN_TENSOR_TYPE_U32,
  HB_DNN_TENSOR_TYPE_F64, HB_DNN_TENSOR_TYPE_S64, HB_DNN_TENSOR_TYPE_U64,
  HB_DNN_TENSOR_TYPE_BOOL8
} hbDNNDataType;
typedef struct hbDNNTensorShape {
  int32_t dimensionSize[HB_DNN_TENSOR_MAX_DIMENSIONS];
  int32_t numDimensions;
} hbDNNTensorShape;
typedef struct hbDNNQuantiScale {
  int32_t scaleLen; float *scaleData; int32_t zeroPointLen; int32_t *zeroPointData;
} hbDNNQuantiScale;
typedef enum { NONE, SCALE } hbDNNQuantiType;
typedef struct hbDNNTensorProperties {
  hbDNNTensorShape validShape;
  int32_t tensorType;
  hbDNNQuantiScale scale;
  hbDNNQuantiType quantiType;
  int32_t quantizeAxis;
  int64_t alignedByteSize;
  int64_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
} hbDNNTensorProperties;
typedef struct hbDNNTensor { hbUCPSysMem sysMem; hbDNNTensorProperties properties; } hbDNNTensor;
// The real S100 SDK spells the packed handle hbDNNPackedHandle_t; the X5-only
// hbPackedDNNHandle_t must fail to compile here.
typedef void *hbDNNPackedHandle_t;
typedef void *hbDNNHandle_t;
int32_t hbDNNInitializeFromFiles(hbDNNPackedHandle_t*, const char**, uint32_t);
int32_t hbDNNGetModelNameList(const char***, int32_t*, hbDNNPackedHandle_t);
int32_t hbDNNGetModelHandle(hbDNNHandle_t*, hbDNNPackedHandle_t, const char*);
int32_t hbDNNGetInputCount(int32_t*, hbDNNHandle_t);
int32_t hbDNNGetOutputCount(int32_t*, hbDNNHandle_t);
int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
int32_t hbDNNInferV2(hbUCPTaskHandle_t*, hbDNNTensor*, const hbDNNTensor*, hbDNNHandle_t);
void hbDNNRelease(hbDNNPackedHandle_t);
""",
    "opencv2/core/mat.hpp": """\
#pragma once
#include <cstddef>
#include <string>
namespace cv {
struct Point { int x = 0; int y = 0; };
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, int type) : rows(rows), cols(cols) {}
  bool empty() const { return rows <= 0 || cols <= 0; }
  int rows = 0;
  int cols = 0;
};
}
#define CV_8UC3 16
""",
    "opencv2/imgcodecs.hpp": """\
#pragma once
#include "core/mat.hpp"
namespace cv { Mat imread(const std::string& path, int flags = 1); }
""",
    "opencv2/imgproc.hpp": "#pragma once\n#include \"core/mat.hpp\"\n",
    "opencv2/core.hpp": "#pragma once\n#include \"core/mat.hpp\"\n",
}

X5_STUBS = {
    "dnn/hb_dnn.h": """\
#pragma once
#include <stdint.h>
#define HB_DNN_TENSOR_MAX_DIMENSIONS 8
typedef enum {
  HB_DNN_TENSOR_TYPE_F32 = 0, HB_DNN_TENSOR_TYPE_S32, HB_DNN_TENSOR_TYPE_U32,
  HB_DNN_TENSOR_TYPE_F64, HB_DNN_TENSOR_TYPE_S64, HB_DNN_TENSOR_TYPE_U64,
  HB_DNN_TENSOR_TYPE_BOOL8, HB_DNN_TENSOR_TYPE_MAX,
  HB_DNN_TENSOR_TYPE_S8, HB_DNN_TENSOR_TYPE_U8, HB_DNN_TENSOR_TYPE_S16
} hbDNNDataType;
typedef enum { HB_DNN_IMG_TYPE_NV12 = 0 } hbDNNImageType;
typedef struct {
  int32_t dimensionSize[HB_DNN_TENSOR_MAX_DIMENSIONS];
  int32_t numDimensions;
} hbDNNTensorShape;
typedef struct { int32_t shiftLen; uint8_t *shiftData; } hbDNNQuantiShift;
typedef struct {
  int32_t scaleLen; float *scaleData; int32_t zeroPointLen; int8_t *zeroPointData;
} hbDNNQuantiScale;
typedef enum { NONE, SHIFT, SCALE } hbDNNQuantiType;
typedef struct {
  hbDNNTensorShape validShape;
  hbDNNTensorShape alignedShape;
  int32_t tensorLayout;
  int32_t tensorType;
  hbDNNQuantiShift shift;
  hbDNNQuantiScale scale;
  hbDNNQuantiType quantiType;
  int32_t quantizeAxis;
  int32_t alignedByteSize;
  int32_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
} hbDNNTensorProperties;
typedef struct hbSysMem { void *virAddr; uint64_t phyAddr; uint32_t memSize; } hbSysMem;
typedef struct { hbSysMem sysMem[4]; hbDNNTensorProperties properties; } hbDNNTensor;
typedef void *hbPackedDNNHandle_t;
typedef void *hbDNNHandle_t;
typedef void *hbDNNTaskHandle_t;
typedef struct hbDNNInferCtrlParam { int32_t more; } hbDNNInferCtrlParam;
#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM(param) do { (param)->more = 0; } while (0)
enum { HB_SYS_MEM_CACHE_CLEAN = 1, HB_SYS_MEM_CACHE_INVALIDATE = 2 };
int32_t hbDNNInitializeFromFiles(hbPackedDNNHandle_t*, const char**, uint32_t);
int32_t hbDNNGetModelNameList(const char***, int32_t*, hbPackedDNNHandle_t);
int32_t hbDNNGetModelHandle(hbDNNHandle_t*, hbPackedDNNHandle_t, const char*);
int32_t hbDNNGetInputCount(int32_t*, hbDNNHandle_t);
int32_t hbDNNGetOutputCount(int32_t*, hbDNNHandle_t);
int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
int32_t hbDNNInfer(hbDNNTaskHandle_t*, hbDNNTensor**, const hbDNNTensor*, hbDNNHandle_t,
                  hbDNNInferCtrlParam*);
int32_t hbDNNWaitTaskDone(hbDNNTaskHandle_t, int32_t timeout);
void hbDNNReleaseTask(hbDNNTaskHandle_t);
void hbDNNRelease(hbPackedDNNHandle_t);
int32_t hbSysAllocCachedMem(hbSysMem*, uint32_t);
int32_t hbSysFreeMem(hbSysMem*);
int32_t hbSysFlushMem(hbSysMem*, int);
""",
    "dnn/hb_dnn_ext.h": '#pragma once\n#include "hb_dnn.h"\n',
    "opencv2/core/mat.hpp": """\
#pragma once
#include <cstddef>
#include <string>
namespace cv {
struct Point { int x = 0; int y = 0; };
struct Size { int width = 0; int height = 0; Size(int w, int h) : width(w), height(h) {} };
struct Scalar {
  double v[4] = {0, 0, 0, 0};
  Scalar(double s0, double s1, double s2) { v[0] = s0; v[1] = s1; v[2] = s2; }
};
struct Rect {
  int x = 0; int y = 0; int width = 0; int height = 0;
  Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
};
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, int type) : rows(rows), cols(cols) {}
  Mat(int rows, int cols, int type, const Scalar& s) : rows(rows), cols(cols) { (void)s; }
  bool empty() const { return rows <= 0 || cols <= 0; }
  Mat operator()(const Rect& roi) const { (void)roi; return Mat(); }
  void copyTo(Mat& dst) const { dst = *this; }
  void copyTo(Mat&& dst) const { (void)dst; }
  int rows = 0;
  int cols = 0;
  unsigned char* data = nullptr;
};
}
#define CV_8UC3 16
""",
    "opencv2/imgcodecs.hpp": """\
#pragma once
#include "core/mat.hpp"
namespace cv { Mat imread(const std::string& path, int flags = 1); }
""",
    "opencv2/imgproc.hpp": """\
#pragma once
#include "core/mat.hpp"
namespace cv {
enum { COLOR_BGR2YUV_I420 = 84 };
void resize(const Mat& src, Mat& dst, Size dsize, double fx = 0, double fy = 0,
            int interpolation = 1);
void cvtColor(const Mat& src, Mat& dst, int code);
}
""",
    "opencv2/opencv.hpp": """\
#pragma once
#include "core/mat.hpp"
#include "imgcodecs.hpp"
#include "imgproc.hpp"
""",
}


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
            str(CPP / "src" / "yolov5_s_native.cpp"),
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

    def test_s_private_dequantizer_values(self):
        self.assert_check("s_dequant_values")

    def test_bpu_core_maps_to_backend_bitmask(self):
        self.assert_check("core_mapping")

    def test_native_binary_refuses_a_mismatched_target(self):
        self.assert_check("target_identity")

    def test_decode_score_boundary_is_an_explicit_policy(self):
        self.assert_check("decode_boundary")

    def test_decode_top_k_cap_is_an_explicit_policy(self):
        self.assert_check("decode_topk")

    def test_dump_hashes_are_correct(self):
        self.assert_check("dump", Path(self.harness.directory.name))

    def test_s_adapter_compiles_against_the_recorded_s100_sdk_shape(self):
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            for relative, text in S100_STUBS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            (root / "hobot" / "dnn" / "hb_dnn_ext.h").write_text("#pragma once\n")
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-fsyntax-only",
                 "-I", str(CPP / "include"), "-I", str(root), "-I", str(C_UTILS),
                 str(CPP / "src" / "s_adapter.cpp")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_s_adapter_rejects_x5_only_sdk_spellings(self):
        # The mirror check: an X5-shaped header set (alignedShape, SHIFT,
        # hb_dnn_ext.h API) must NOT silently satisfy the S adapter's includes.
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            for relative, text in X5_STUBS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            for extra in ("opencv2/core.hpp",):
                path = root / extra
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(S100_STUBS[extra])
            result = subprocess.run(
                ["c++", "-std=c++17", "-fsyntax-only",
                 "-I", str(CPP / "include"), "-I", str(root), "-I", str(C_UTILS),
                 str(CPP / "src" / "s_adapter.cpp")],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0,
                                "s_adapter must not compile against an X5-shaped SDK")

    def test_x5_adapter_compiles_against_the_recorded_x5_sdk_shape(self):
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            for relative, text in X5_STUBS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-fsyntax-only",
                 "-I", str(CPP / "include"), "-I", str(root),
                 str(CPP / "src" / "x5_adapter.cpp")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dump_manifest_is_independently_verifiable(self):
        scratch = Path(self.harness.directory.name) / "dump-verify"
        self.assert_check("dump", scratch)
        manifest = json.loads((scratch / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "rdk-model-zoo/yolov5-cpp-dump/v2")
        self.assertEqual(manifest["return_code"], 0)
        self.assertEqual(manifest["target"], "x5")
        self.assertIsNone(manifest["binary_sha256"])
        reported = manifest["outputs"][0]
        self.assertEqual(reported["aligned_byte_size"], 16)
        self.assertEqual(reported["stride"], [16, 8, 4, 4])
        self.assertEqual(reported["aligned"], [1, 2, 2, 4])
        self.assertEqual(reported["quantize_axis"], 3)
        self.assertEqual(reported["scale_values"], [])
        self.assertEqual(reported["zero_point_values"], [])
        unreported = manifest["inputs"][0]
        self.assertIsNone(unreported["aligned_byte_size"])
        self.assertEqual(unreported["stride"], [None, None, None, None])
        self.assertEqual(unreported["aligned"], [None, None, None, None])
        # Each stage writes into its own subdirectory, so the raw and the
        # transformed payload of one output can never overwrite each other.
        raw = manifest["raw_tensors"][0]
        transformed = manifest["transformed_tensors"][0]
        input_entry = manifest["input_tensors"][0]
        self.assertEqual(raw["file"], "raw/0-output0.bin")
        self.assertEqual(transformed["file"], "transformed/0-output0.bin")
        self.assertEqual(input_entry["file"], "input/0-input0.bin")
        raw_payload = (scratch / raw["file"]).read_bytes()
        transformed_payload = (scratch / transformed["file"]).read_bytes()
        input_payload = (scratch / input_entry["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw_payload).hexdigest(), raw["sha256"])
        self.assertEqual(hashlib.sha256(transformed_payload).hexdigest(),
                         transformed["sha256"])
        self.assertEqual(hashlib.sha256(input_payload).hexdigest(), input_entry["sha256"])
        self.assertNotEqual(raw_payload, transformed_payload)
        self.assertEqual(struct.unpack("<4i", raw_payload), (11, 22, 33, 44))
        self.assertEqual(struct.unpack("<4f", transformed_payload), (1.0, 2.0, 3.0, 4.0))
        detection = manifest["detections"][0]
        self.assertEqual(detection["class_id"], 7)
        self.assertAlmostEqual(detection["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
