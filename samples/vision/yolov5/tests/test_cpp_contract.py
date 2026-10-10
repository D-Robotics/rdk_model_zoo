# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Behavioural host checks for the consolidated YOLOv5 C++ runtime surface.

The board backend needs the target SDK, which a host does not have. The gates,
decoder, dequantizer and dump writer live in detect.cpp and cli.cpp and are
exercised here with explicit metadata, so a passing test proves an
accept/reject decision rather than the presence of a string in a source file.
The SDK backends themselves are compile-checked against stub headers that
encode each SDK's recorded shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


# Bounded subprocess budgets (seconds). A wedged compiler or a hung binary
# surfaces as a failed check instead of an unbounded test run.
COMPILE_TIMEOUT = 240
RUN_TIMEOUT = 60

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
# board re-verification is still required; these stubs only keep each backend
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
const char *hbUCPGetErrorDesc(int32_t error_code);
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
const char *hbDNNGetErrorDesc(int32_t error_code);
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
  Mat(int rows, int cols, int type) : rows(rows), cols(cols) { (void)type; }
  Mat(int rows, int cols, int type, unsigned char* data)
      : rows(rows), cols(cols), data(data) { (void)type; }
  bool empty() const { return rows <= 0 || cols <= 0; }
  template <typename T>
  const T* ptr() const {
    return reinterpret_cast<const T*>(data);
  }
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
void cvtColor(const Mat& src, Mat& dst, int code);
}
""",
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
  Rect(int x_, int y_, int w, int h_) : x(x_), y(y_), width(w), height(h_) {}
};
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, int type) : rows(rows), cols(cols) { (void)type; }
  Mat(int rows, int cols, int type, const Scalar& s) : rows(rows), cols(cols) { (void)s; }
  Mat(int rows, int cols, int type, unsigned char* data)
      : rows(rows), cols(cols), data(data) { (void)type; }
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

# The CLI renders through OpenCV, so the host harness needs a linkable stub of
# exactly the drawing API render_detections uses (nothing more).
HOST_RENDER_STUBS = {
    "opencv2/core/mat.hpp": """\
#pragma once
#include <cstddef>
#include <string>
namespace cv {
struct Point {
  int x = 0;
  int y = 0;
  Point() = default;
  Point(int px, int py) : x(px), y(py) {}
};
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, int type) : rows(rows), cols(cols) { (void)type; }
  bool empty() const { return rows <= 0 || cols <= 0; }
  long long total() const { return static_cast<long long>(rows) * cols; }
  std::size_t elemSize() const { return 3; }
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
#include <cstddef>
namespace cv {
Mat imread(const std::string& path, int flags = 1);
bool imwrite(const std::string& path, const Mat& image);
}
""",
    "opencv2/imgproc.hpp": """\
#pragma once
#include "core/mat.hpp"
namespace cv {
enum { FONT_HERSHEY_SIMPLEX = 0 };
struct Scalar {
  double v[4] = {0, 0, 0, 0};
  Scalar(double s0, double s1, double s2) { v[0] = s0; v[1] = s1; v[2] = s2; }
};
void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1);
void putText(Mat& img, const std::string& text, Point org, int fontFace, double fontScale,
             const Scalar& color, int thickness = 1);
}
""",
    "cv_render_stub.cpp": """\
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
namespace cv {
Mat imread(const std::string&, int) { return Mat(); }
bool imwrite(const std::string&, const Mat&) { return true; }
void rectangle(Mat&, Point, Point, const Scalar&, int) {}
void putText(Mat&, const std::string&, Point, int, double, const Scalar&, int) {}
}
""",
}

# A linkable fake of the X5 HB-DNN entry points detect.cpp uses, plus a driver
# that proves the per-call stage-ownership contract with REAL OpenCV image
# math: preprocess(A), preprocess(B), infer(prepared_A) must submit A's NV12
# payload, not whatever the most recent preprocess left behind. The fake
# records every submitted input buffer; its outputs are quiet constants that
# decode to no detections, so the assertions focus on ownership and payload
# identity. Real-SDK board runs remain the authoritative check.
X5_FAKE = {
    "x5_fake.h": """\
#pragma once
#include <cstddef>
#include <vector>
namespace yolov5_fake {
// Bytes submitted with the most recent hbDNNInfer call.
const std::vector<unsigned char>& last_submitted_input();
std::size_t infer_calls();
}
""",
    "x5_fake.cpp": """\
#include "x5_fake.h"
#include "dnn/hb_dnn.h"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace yolov5_fake {
namespace {
std::vector<unsigned char> g_last_submitted;
std::size_t g_infer_calls = 0;
}  // namespace

const std::vector<unsigned char>& last_submitted_input() { return g_last_submitted; }
std::size_t infer_calls() { return g_infer_calls; }
}  // namespace yolov5_fake

namespace {
const int32_t kInput = 640;
const int32_t kClasses = 80;
const int32_t kHeadSizes[3] = {80, 40, 20};

void fill_input_properties(hbDNNTensorProperties* props) {
  std::memset(props, 0, sizeof(*props));
  props->tensorType = HB_DNN_IMG_TYPE_NV12;
  props->quantiType = NONE;
  props->validShape.numDimensions = 4;
  props->validShape.dimensionSize[0] = 1;
  props->validShape.dimensionSize[1] = 3;
  props->validShape.dimensionSize[2] = kInput;
  props->validShape.dimensionSize[3] = kInput;
  props->alignedShape = props->validShape;
  props->alignedByteSize = kInput * kInput * 3 / 2;
}
}  // namespace

int32_t hbDNNInitializeFromFiles(hbPackedDNNHandle_t* packed, const char** files,
                                 uint32_t count) {
  (void)files;
  (void)count;
  *packed = reinterpret_cast<hbPackedDNNHandle_t>(0x1234);
  return 0;
}

int32_t hbDNNGetModelNameList(const char*** names, int32_t* count,
                              hbPackedDNNHandle_t packed) {
  (void)packed;
  static const char* model = "fake-yolov5";
  *names = &model;
  *count = 1;
  return 0;
}

int32_t hbDNNGetModelHandle(hbDNNHandle_t* handle, hbPackedDNNHandle_t packed,
                            const char* name) {
  (void)packed;
  (void)name;
  *handle = reinterpret_cast<hbDNNHandle_t>(0x5678);
  return 0;
}

int32_t hbDNNGetInputCount(int32_t* count, hbDNNHandle_t model) {
  (void)model;
  *count = 1;
  return 0;
}

int32_t hbDNNGetOutputCount(int32_t* count, hbDNNHandle_t model) {
  (void)model;
  *count = 3;
  return 0;
}

int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties* props, hbDNNHandle_t model,
                                      int32_t index) {
  (void)model;
  if (index != 0) return -1;
  fill_input_properties(props);
  return 0;
}

int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties* props, hbDNNHandle_t model,
                                       int32_t index) {
  (void)model;
  if (index < 0 || index > 2) return -1;
  std::memset(props, 0, sizeof(*props));
  props->tensorType = HB_DNN_TENSOR_TYPE_F32;
  props->quantiType = NONE;
  props->validShape.numDimensions = 4;
  props->validShape.dimensionSize[0] = 1;
  props->validShape.dimensionSize[1] = kHeadSizes[index];
  props->validShape.dimensionSize[2] = kHeadSizes[index];
  props->validShape.dimensionSize[3] = 3 * (5 + kClasses);
  props->alignedShape = props->validShape;
  props->alignedByteSize =
      kHeadSizes[index] * kHeadSizes[index] * props->validShape.dimensionSize[3] * 4;
  return 0;
}

int32_t hbSysAllocCachedMem(hbSysMem* mem, uint32_t size) {
  mem->virAddr = std::calloc(size, 1);
  mem->phyAddr = 0;
  mem->memSize = size;
  return mem->virAddr != nullptr ? 0 : -1;
}

int32_t hbSysFreeMem(hbSysMem* mem) {
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}

int32_t hbSysFlushMem(hbSysMem* mem, int type) {
  (void)mem;
  (void)type;
  return 0;
}

int32_t hbDNNInfer(hbDNNTaskHandle_t* task, hbDNNTensor** outputs,
                   const hbDNNTensor* input, hbDNNHandle_t model,
                   hbDNNInferCtrlParam* ctrl) {
  (void)model;
  (void)ctrl;
  using yolov5_fake::g_last_submitted;
  using yolov5_fake::g_infer_calls;
  ++g_infer_calls;
  const unsigned char* begin = static_cast<const unsigned char*>(input->sysMem[0].virAddr);
  g_last_submitted.assign(begin, begin + input->properties.alignedByteSize);
  // Quiet deterministic outputs: every raw value -10.0f decodes to nothing at
  // the default threshold, keeping the ownership assertions independent of
  // decode noise. The SDK receives hbDNNTensor** as the address of a pointer
  // to the contiguous caller-allocated tensor array, matching how detect.cpp
  // passes lease.outputs.
  hbDNNTensor* tensor_array = *outputs;
  for (int i = 0; i < 3; ++i) {
    hbDNNTensor& out = tensor_array[i];
    const int32_t count = out.properties.validShape.dimensionSize[1] *
                          out.properties.validShape.dimensionSize[2] *
                          out.properties.validShape.dimensionSize[3];
    float value = -10.0F;
    unsigned char* base = static_cast<unsigned char*>(out.sysMem[0].virAddr);
    for (int32_t k = 0; k < count; ++k) std::memcpy(base + 4 * k, &value, sizeof(value));
  }
  *task = reinterpret_cast<hbDNNTaskHandle_t>(0x9abc);
  return 0;
}

int32_t hbDNNWaitTaskDone(hbDNNTaskHandle_t task, int32_t timeout) {
  (void)task;
  (void)timeout;
  return 0;
}

void hbDNNReleaseTask(hbDNNTaskHandle_t task) { (void)task; }
void hbDNNRelease(hbPackedDNNHandle_t packed) { (void)packed; }
""",
    "interleaved_driver.cpp": """\
#include "detect.hpp"
#include "x5_fake.h"
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\\n", what);
    ++failures;
  }
}
}  // namespace

int main() {
  yolov5::Yolov5::Config config;
  config.target = "x5";
  config.model_path = "fake-model.bin";
  yolov5::Yolov5 model(config);

  // An empty/short Prepared is rejected at infer entry, before any SDK
  // allocation or copy could index into it.
  bool rejected_empty = false;
  try {
    const yolov5::Yolov5::Prepared empty;
    model.infer(empty);
  } catch (const std::invalid_argument& error) {
    rejected_empty = std::string(error.what()).find("prepared input") != std::string::npos;
  }
  check(rejected_empty, "infer rejects an empty Prepared before any SDK work");
  check(yolov5_fake::infer_calls() == 0, "the rejected call never reached the SDK");

  yolov5::Yolov5::Input a;
  a.source_cols = 96;
  a.source_rows = 64;
  a.bgr.assign(static_cast<std::size_t>(96) * 64 * 3, 0);
  for (std::size_t i = 0; i < a.bgr.size(); ++i)
    a.bgr[i] = static_cast<unsigned char>((i * 7) % 251);
  yolov5::Yolov5::Input b;
  b.source_cols = 64;
  b.source_rows = 96;
  b.bgr.assign(static_cast<std::size_t>(64) * 96 * 3, 231);

  const yolov5::Yolov5::Prepared pa = model.preprocess(a);
  const yolov5::Yolov5::Prepared pb = model.preprocess(b);
  const std::size_t frame = 640u * 640u * 3 / 2;
  check(pa.nv12.size() == frame, "prepared A payload size");
  check(pb.nv12.size() == frame, "prepared B payload size");
  check(pa.nv12 != pb.nv12, "prepared payloads differ");
  check(pa.source_cols == 96 && pa.source_rows == 64, "prepared A geometry");
  check(pb.source_cols == 64 && pb.source_rows == 96, "prepared B geometry");
  check(pb.y_plane.empty() && pb.uv_plane.empty(), "X5 prepared carries no S planes");

  // THE interleaved case: B was preprocessed last, so a model whose infer
  // consumed instance buffers mutated by preprocess would submit B. infer
  // must upload exactly its explicit argument and run A.
  const yolov5::Yolov5::RawResult raw = model.infer(pa);
  check(yolov5_fake::infer_calls() == 1, "one infer call so far");
  check(yolov5_fake::last_submitted_input() == pa.nv12,
        "infer submitted prepared A, not the later preprocess B");
  check(yolov5_fake::last_submitted_input() != pb.nv12, "submitted payload is not B's");
  check(raw.source_cols == 96 && raw.source_rows == 64, "raw geometry follows prepared A");
  check(raw.heads.size() == 3 && raw.shapes.size() == 3, "three heads");
  const std::size_t expected_head_sizes[3] = {80u * 80u * 255, 40u * 40u * 255,
                                              20u * 20u * 255};
  bool heads_quiet = raw.heads.size() == 3;
  for (std::size_t i = 0; i < raw.heads.size(); ++i) {
    if (raw.heads[i].size() != expected_head_sizes[i]) heads_quiet = false;
    for (float value : raw.heads[i])
      if (value != -10.0F) heads_quiet = false;
  }
  check(heads_quiet, "raw heads are the copied quiet fake outputs");
  check(raw.inputs.size() == 1 && raw.input_tensors.size() == 1, "input evidence carried");
  check(raw.input_tensors[0].bytes == pa.nv12, "input evidence payload is prepared A");

  const yolov5::Yolov5::Result result = model.postprocess(raw);
  check(result.detections.empty(), "quiet fake heads decode to nothing");

  const yolov5::Yolov5::Prediction run = model.predict(a);
  check(yolov5_fake::infer_calls() == 2, "predict ran its own chain");
  check(yolov5_fake::last_submitted_input() == pa.nv12, "predict submitted A");
  check(run.result.detections.empty(), "predict result quiet");
  check(run.evidence.input_tensors.size() == 1 &&
            run.evidence.input_tensors[0].bytes == pa.nv12,
        "predict evidence payload is A's");
  check(run.evidence.outputs.size() == 3 && run.evidence.raw_tensors.size() == 3 &&
            run.evidence.transformed_tensors.size() == 3,
        "predict evidence outputs");
  check(run.evidence.target == "x5" && run.evidence.build_target == "x5",
        "evidence records the build identity");
  check(run.evidence.model_path == "fake-model.bin", "evidence records the model path");
  check(run.evidence.detections.empty() && run.evidence.detections_original.empty(),
        "no detections decoded");

  if (failures == 0) std::printf("interleaved-ok\\n");
  return failures == 0 ? 0 : 1;
}
""",
}


def write_stubs(root: Path, stubs: dict[str, str]) -> None:
    for relative, text in stubs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def real_opencv_flags() -> tuple[list[str], str] | None:
    """Compiler/linker flags for a real OpenCV, or None when none is found.

    Discovery order: explicit overrides (RDK_TEST_OPENCV_INCLUDE + _LIB, for
    CI or board toolchains without pkg-config), then pkg-config opencv5 /
    opencv4, then the Homebrew OpenCV 5 keg. The fixture needs the real
    letterbox/NV12 math, so a missing dependency is a skip, not a stub.
    """
    include_override = os.environ.get("RDK_TEST_OPENCV_INCLUDE")
    lib_override = os.environ.get("RDK_TEST_OPENCV_LIB")
    if include_override and lib_override:
        return (["-I", include_override, "-L", lib_override],
                f"overrides {include_override}/{lib_override}")
    for module in ("opencv5", "opencv4"):
        probe = subprocess.run(["pkg-config", "--cflags", "--libs", module],
                               capture_output=True, text=True,
                               timeout=RUN_TIMEOUT)
        if probe.returncode == 0 and probe.stdout.strip():
            return (shlex.split(probe.stdout), f"pkg-config {module}")
    homebrew_include = Path("/opt/homebrew/include/opencv5")
    homebrew_lib = Path("/opt/homebrew/lib")
    if (homebrew_include / "opencv2" / "opencv.hpp").is_file():
        return (["-I", str(homebrew_include), "-L", str(homebrew_lib)],
                "homebrew opencv5 fallback")
    return None


class PortableHarness:
    """Compiles the SDK-free core once and runs one check per call."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.binary = root / "portable_checks"
        self.error = ""
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.error = "host C++ compiler unavailable"
            return
        stub_root = root / "opencv-stub"
        write_stubs(stub_root, HOST_RENDER_STUBS)
        command = [
            compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
            "-I", str(CPP / "inc"), "-I", str(stub_root), str(HARNESS),
            str(CPP / "src" / "detect.cpp"),
            str(CPP / "src" / "cli.cpp"),
            str(stub_root / "cv_render_stub.cpp"),
            "-o", str(self.binary),
        ]
        built = subprocess.run(command, capture_output=True, text=True,
                               timeout=COMPILE_TIMEOUT)
        if built.returncode != 0:
            self.error = built.stderr

    def run(self, check: str, scratch: Path | None = None):
        arguments = [str(self.binary), check]
        if scratch is not None:
            arguments.append(str(scratch))
        return subprocess.run(arguments, capture_output=True, text=True,
                              timeout=RUN_TIMEOUT)


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
            CPP / "inc" / "cli.hpp",
            CPP / "inc" / "detect.hpp",
            CPP / "src" / "main.cpp",
            CPP / "src" / "cli.cpp",
            CPP / "src" / "detect.cpp",
            CPP / "launcher.py",
            CPP / "run.sh",
        ):
            self.assertTrue(path.is_file(), path)

    def test_the_consolidated_layout_has_no_leftover_fragments(self):
        # The gate/dump/decode/visualize/s_native fragments and the per-board
        # adapter files were merged into the five-file layout; none of them may
        # come back as separate delivery files.
        for path in (
            CPP / "include",
            CPP / "src" / "cli_main.cpp",
            CPP / "src" / "x5_adapter.cpp",
            CPP / "src" / "s_adapter.cpp",
            CPP / "src" / "yolov5_decode.cpp",
            CPP / "src" / "yolov5_dump.cpp",
            CPP / "src" / "yolov5_gate.cpp",
            CPP / "src" / "yolov5_s_native.cpp",
            CPP / "src" / "yolov5_visualize.cpp",
        ):
            self.assertFalse(path.exists(), path)

    def test_cmake_binds_each_target_to_one_backend_and_the_build_identity(self):
        cmake = (CPP / "CMakeLists.txt").read_text()
        # Configuration must not inspect the board; the target is explicit.
        self.assertNotIn("/sys/class/boardinfo", cmake)
        for token in (
            "src/detect.cpp",
            "YOLOV5_TARGET_X5=1",
            "YOLOV5_TARGET_S=1",
            'YOLOV5_TARGET_NAME="x5"',
            'YOLOV5_TARGET_NAME="${YOLOV5_TARGET}"',
            "SOC_S600",
        ):
            self.assertIn(token, cmake)
        # The five-file layout is one production unit; no adapter split remains.
        self.assertNotIn("_adapter.cpp", cmake)

    def test_main_constructs_the_model_and_reports_through_the_cli(self):
        main_src = (CPP / "src" / "main.cpp").read_text()
        for token in (
            "yolov5::Yolov5 model(config)",
            "model.predict(input)",
            "const yolov5::Yolov5::Prediction run = model.predict(input)",
            "yolov5::report(options, run, source, model.input_size())",
            "yolov5::SourceImage source(options)",
            "yolov5::write_failure_record",
        ):
            self.assertIn(token, main_src)
        # The CLI owns the file IO; the model must receive pixels, not a path.
        self.assertIn("input.bgr = source.bgr()", main_src)
        self.assertNotIn("image_path", main_src)

    def test_stage_data_is_owned_per_call(self):
        # The reviewer-corrected contract: preprocess returns the owned NV12
        # payload, infer uploads its explicit argument, predict returns the
        # result/evidence bundle. Last-call accessors and instance evidence
        # must not come back.
        header = (CPP / "inc" / "detect.hpp").read_text()
        self.assertIn("Prediction predict(const Input& input)", header)
        self.assertIn("std::vector<unsigned char> nv12", header)
        self.assertIn("std::vector<unsigned char> y_plane", header)
        self.assertNotIn("RunEvidence evidence() const", header)
        self.assertNotIn("const cv::Mat& image() const", header)
        source = (CPP / "src" / "detect.cpp").read_text()
        self.assertNotIn("(void)prepared", source)
        self.assertNotIn("cv::imread", source)

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

    def test_x5_backend_compiles_against_the_recorded_x5_sdk_shape(self):
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            write_stubs(root, X5_STUBS)
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-fsyntax-only",
                 "-DYOLOV5_TARGET_X5=1", '-DYOLOV5_TARGET_NAME="x5"',
                 "-I", str(CPP / "inc"), "-I", str(root),
                 str(CPP / "src" / "detect.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_x5_backend_rejects_s_only_sdk_spellings(self):
        # The mirror check: an S-shaped header set (single sysMem, no
        # alignedShape, hbDNNPackedHandle_t, no hb_dnn_ext.h API) must NOT
        # silently satisfy the X5 backend's includes.
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            write_stubs(root, S100_STUBS)
            result = subprocess.run(
                ["c++", "-std=c++17", "-fsyntax-only",
                 "-DYOLOV5_TARGET_X5=1", '-DYOLOV5_TARGET_NAME="x5"',
                 "-I", str(CPP / "inc"), "-I", str(root),
                 str(CPP / "src" / "detect.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertNotEqual(result.returncode, 0,
                                "the X5 backend must not compile against an S-shaped SDK")

    def test_s_backend_compiles_against_the_recorded_s100_sdk_shape(self):
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            write_stubs(root, S100_STUBS)
            (root / "hobot" / "dnn" / "hb_dnn_ext.h").write_text("#pragma once\n")
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-fsyntax-only",
                 "-DYOLOV5_TARGET_S=1", '-DYOLOV5_TARGET_NAME="s100"',
                 "-I", str(CPP / "inc"), "-I", str(root), "-I", str(C_UTILS),
                 str(CPP / "src" / "detect.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_s_backend_rejects_x5_only_sdk_spellings(self):
        # The X5-shaped header set (alignedShape, SHIFT, hb_dnn_ext.h API,
        # hbPackedDNNHandle_t, sysMem[4]) must NOT satisfy the S backend.
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            write_stubs(root, X5_STUBS)
            result = subprocess.run(
                ["c++", "-std=c++17", "-fsyntax-only",
                 "-DYOLOV5_TARGET_S=1", '-DYOLOV5_TARGET_NAME="s100"',
                 "-I", str(CPP / "inc"), "-I", str(root),
                 str(CPP / "src" / "detect.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertNotEqual(result.returncode, 0,
                                "the S backend must not compile against an X5-shaped SDK")

    def test_host_build_compiles_without_a_board_target(self):
        # Without a target macro the translation unit still compiles; the
        # resulting binary refuses to run for lack of a build identity, and the
        # SDK-free core stays linkable for host behaviour checks.
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            write_stubs(root, HOST_RENDER_STUBS)
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-fsyntax-only",
                 "-I", str(CPP / "inc"), "-I", str(root),
                 str(CPP / "src" / "detect.cpp"),
                 str(CPP / "src" / "cli.cpp"),
                 str(CPP / "src" / "main.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_interleaved_preprocess_runs_its_explicit_argument(self):
        # The per-call ownership contract, proven with real OpenCV image math
        # and a linkable fake X5 SDK: after preprocess(A) and preprocess(B),
        # infer(prepared_A) must submit A's NV12 payload. OpenCV is discovered
        # (overrides, pkg-config, Homebrew fallback) so a Linux CI with
        # OpenCV 4 runs the fixture instead of silently skipping it.
        discovered = real_opencv_flags()
        if discovered is None:
            raise unittest.SkipTest(
                "no real OpenCV found (pkg-config opencv5/opencv4, Homebrew or "
                "RDK_TEST_OPENCV_INCLUDE/_LIB overrides); interleaved preprocess "
                "check not-run")
        opencv_flags, source = discovered
        with tempfile.TemporaryDirectory() as stub_root:
            root = Path(stub_root)
            # Only the SDK headers are stubbed; the OpenCV headers must be the
            # real ones so the letterbox/NV12 arithmetic is the shipped math,
            # not a stub of it.
            write_stubs(root, {k: v for k, v in X5_STUBS.items() if k.startswith("dnn/")})
            write_stubs(root, X5_FAKE)
            binary = root / "interleaved_driver"
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                 "-DYOLOV5_TARGET_X5=1", '-DYOLOV5_TARGET_NAME="x5"',
                 "-I", str(CPP / "inc"), "-I", str(root),
                 str(CPP / "src" / "detect.cpp"),
                 str(root / "x5_fake.cpp"),
                 str(root / "interleaved_driver.cpp"),
                 *opencv_flags,
                 "-lopencv_core", "-lopencv_imgproc",
                 "-o", str(binary)],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0,
                             f"opencv via {source}: {result.stderr}")
            run = subprocess.run([str(binary)], capture_output=True, text=True,
                                 timeout=RUN_TIMEOUT)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("interleaved-ok", run.stdout)

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
