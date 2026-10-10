/*
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Model vocabulary and the four task model contracts of the ultralytics_yolo
// C++ runtime. This header is deliberately free of board-SDK and OpenCV
// headers so the decode/binding vocabulary stays host-testable:
//   * pure output views and shared head contracts (tensor view, decode),
//   * pure NV12 plane packing (nv12 geometry),
//   * stride-validated compact output copies (task outputs),
//   * the 1000-way classification math (classification),
//   * the task model classes themselves (pimpl; implementations live in
//     src/{detect,segment,pose,classify}.cpp on top of inc/backend.hpp).
//
// Every task model follows the owned-stage contract: preprocess converts
// caller-owned BGR pixels into an owned NV12 payload without touching the
// SDK, infer validates and uploads exactly its explicit Prepared argument,
// runs the synchronous forward pass and copies the outputs it produced, and
// postprocess only decodes those owned values (it never executes a further
// SDK stage). predict chains the three stages and returns the owned bundle.

#ifndef YOLO_RUNTIME_CPP_INC_YOLO_HPP_
#define YOLO_RUNTIME_CPP_INC_YOLO_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo {

// ---------------------------------------------------------------------------
// Tensor views and shared head contracts (pure, host-testable).
// ---------------------------------------------------------------------------

// Stride-aware read-only view over one FLOAT32 NHWC output tensor.
// `h/w/channels` follow the tensor validShape; `row_step`/`cell_step` are
// the physical strides in floats between consecutive rows/cells. They
// default to the tightly packed values derived from the valid shape.
struct TensorView {
  const float* data = nullptr;
  int h = 0;
  int w = 0;
  int channels = 0;
  int row_step = 0;
  int cell_step = 0;

  const float* cell(int y, int x) const {
    return data + static_cast<std::ptrdiff_t>(y) * row_step +
           static_cast<std::ptrdiff_t>(x) * cell_step;
  }
};

// Compact shape record used for output discovery.
struct OutputShape {
  OutputShape() = default;
  OutputShape(int h_value, int w_value, int c_value)
      : h(h_value), w(w_value), c(c_value) {}

  int h = 0;
  int w = 0;
  int c = 0;
};

// Returns the index of the unique output whose shape equals (h, w, c), or -1
// when no output matches. Mirrors the detect sample behaviour of rejecting
// ambiguous layouts.
inline int find_output_by_shape(const std::vector<OutputShape>& outputs,
                                int h, int w, int c) {
  int match = -1;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const OutputShape& shape = outputs[i];
    if (shape.h == h && shape.w == w && shape.c == c) {
      if (match >= 0) return -1;
      match = static_cast<int>(i);
    }
  }
  return match;
}

// ---------------------------------------------------------------------------
// Shared box-decode primitives for the two YOLO head contracts published by
// this sample:
//   * direct-LTRB heads (YOLO26): box maps carry 4 float distances per cell.
//   * DFL heads (YOLOv5u/v8/v9/v10/yolo11/yolo12/yolov13): box maps carry
//     4 x 16 raw logits per cell that softmax into distance distributions.
// The math mirrors the Python task modules (runtime/python/detect.py,
// pose.py, segment.py) so the C++ and Python runtimes stay
// contract-compatible.
// ---------------------------------------------------------------------------

// Box-decode protocols selectable per model.
enum class BoxDecode { kUnknown, kDirectLtrb, kDfl };

// DFL heads distribute each side distance over this many bins.
const int kDflBins = 16;

// Channels carried by one box-map cell under each protocol.
inline int box_channels(BoxDecode decode) {
  if (decode == BoxDecode::kDirectLtrb) return 4;
  if (decode == BoxDecode::kDfl) return 4 * kDflBins;
  return 0;
}

// Maps an observed box-map channel count to a decode protocol.
inline BoxDecode box_decode_from_channels(int channels) {
  if (channels == 4) return BoxDecode::kDirectLtrb;
  if (channels == 4 * kDflBins) return BoxDecode::kDfl;
  return BoxDecode::kUnknown;
}

inline float sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exp_value = std::exp(value);
  return exp_value / (1.0f + exp_value);
}

// Raw-logit threshold equivalent to `sigmoid(raw) >= score_threshold`.
// Thresholds outside (0, 1) degenerate to +/- infinity, matching the
// behaviour of the released detect sample.
inline float raw_logit_threshold(float score_threshold) {
  if (score_threshold <= 0.0f) return -std::numeric_limits<float>::infinity();
  if (score_threshold >= 1.0f) return std::numeric_limits<float>::infinity();
  return -std::log(1.0f / score_threshold - 1.0f);
}

// Letterbox/resize transform mapping model-input coordinates back to the
// source image.
struct ImageTransform {
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  int shift_x = 0;
  int shift_y = 0;
};

// Direct-LTRB heads: the four distances are stored as-is.
inline void decode_box_ltrb(const float* box4, float* ltrb4) {
  ltrb4[0] = box4[0];
  ltrb4[1] = box4[1];
  ltrb4[2] = box4[2];
  ltrb4[3] = box4[3];
}

// DFL heads: softmax over 16 bins per side, then the bin-index expectation
// is the distance (numerically stabilised like the Python softmax).
inline void decode_box_dfl(const float* box64, float* ltrb4) {
  for (int side = 0; side < 4; ++side) {
    const float* bins = box64 + side * kDflBins;
    float max_value = bins[0];
    for (int j = 1; j < kDflBins; ++j) {
      if (bins[j] > max_value) max_value = bins[j];
    }
    float sum = 0.0f;
    float expectation = 0.0f;
    for (int j = 0; j < kDflBins; ++j) {
      const float weight = std::exp(bins[j] - max_value);
      sum += weight;
      expectation += weight * j;
    }
    ltrb4[side] = expectation / sum;
  }
}

// Converts per-cell distances around a grid centre to input-image corners.
inline void box_from_distances(float grid_center_x, float grid_center_y,
                               const float* ltrb4, float stride, float* x1,
                               float* y1, float* x2, float* y2) {
  *x1 = (grid_center_x - ltrb4[0]) * stride;
  *y1 = (grid_center_y - ltrb4[1]) * stride;
  *x2 = (grid_center_x + ltrb4[2]) * stride;
  *y2 = (grid_center_y + ltrb4[3]) * stride;
}

// Maps input-image coordinates back to source-image coordinates and clamps
// them to the image bounds. Returns false for degenerate boxes.
inline bool map_to_source(float* x1, float* y1, float* x2, float* y2,
                          const ImageTransform& transform, int image_w,
                          int image_h) {
  *x1 = (*x1 - transform.shift_x) / transform.scale_x;
  *y1 = (*y1 - transform.shift_y) / transform.scale_y;
  *x2 = (*x2 - transform.shift_x) / transform.scale_x;
  *y2 = (*y2 - transform.shift_y) / transform.scale_y;
  *x1 = std::max(0.0f, std::min(*x1, static_cast<float>(image_w)));
  *y1 = std::max(0.0f, std::min(*y1, static_cast<float>(image_h)));
  *x2 = std::max(0.0f, std::min(*x2, static_cast<float>(image_w)));
  *y2 = std::max(0.0f, std::min(*y2, static_cast<float>(image_h)));
  return *x2 > *x1 && *y2 > *y1;
}

// ---------------------------------------------------------------------------
// Oriented-box geometry for the YOLO26 OBB head (pure, host-testable). The
// math mirrors runtime/python/obb.py, including its platform policy:
// X5 wraps angles to [-pi/2, pi/2), runs per-class rotated NMS and clips
// restored boxes; the S series keeps unclipped geometry.
// ---------------------------------------------------------------------------

struct RotatedBox {
  float cx = 0.0f;
  float cy = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float angle_rad = 0.0f;
};

inline bool decode_obb_cell(const float* ltrb_raw, float angle_raw,
                            float grid_x, float grid_y, float stride,
                            float angle_sign, float angle_offset_rad,
                            RotatedBox* box) {
  if (ltrb_raw == nullptr || box == nullptr || !std::isfinite(angle_raw) ||
      !std::isfinite(stride) || stride <= 0.0f) {
    return false;
  }
  const float left = std::fabs(ltrb_raw[0]);
  const float top = std::fabs(ltrb_raw[1]);
  const float right = std::fabs(ltrb_raw[2]);
  const float bottom = std::fabs(ltrb_raw[3]);
  if (!std::isfinite(left) || !std::isfinite(top) ||
      !std::isfinite(right) || !std::isfinite(bottom)) {
    return false;
  }
  const float angle = angle_raw * angle_sign + angle_offset_rad;
  if (!std::isfinite(angle)) return false;
  const float half_w = (right - left) * 0.5f;
  const float half_h = (bottom - top) * 0.5f;
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  box->cx = (grid_x + half_w * cosine - half_h * sine) * stride;
  box->cy = (grid_y + half_w * sine + half_h * cosine) * stride;
  box->width = (left + right) * stride;
  box->height = (top + bottom) * stride;
  box->angle_rad = angle;
  return std::isfinite(box->cx) && std::isfinite(box->cy) &&
         std::isfinite(box->width) && std::isfinite(box->height) &&
         box->width > 0.0f && box->height > 0.0f;
}

inline void regularize_obb(RotatedBox* box, bool regularize,
                           bool wrap_half_turn) {
  if (box == nullptr) return;
  const float half_pi = static_cast<float>(std::acos(-1.0) * 0.5);
  const float pi = half_pi * 2.0f;
  if (regularize && box->width < box->height) {
    std::swap(box->width, box->height);
    box->angle_rad += half_pi;
  }
  if (wrap_half_turn) {
    box->angle_rad = std::fmod(box->angle_rad + half_pi, pi);
    if (box->angle_rad < 0.0f) box->angle_rad += pi;
    box->angle_rad -= half_pi;
  }
}

// Restores a model-input box to source pixels. `clip` bounds the centre and
// size to the image, matching the X5 policy of runtime/python/obb.py;
// the S-series policy keeps the unclipped geometry.
inline bool map_obb_to_source(RotatedBox* box,
                              const ImageTransform& transform, int image_w,
                              int image_h, bool clip) {
  if (box == nullptr || image_w <= 0 || image_h <= 0 ||
      !std::isfinite(transform.scale_x) ||
      !std::isfinite(transform.scale_y) || transform.scale_x <= 0.0f ||
      transform.scale_y <= 0.0f) {
    return false;
  }
  box->cx = (box->cx - transform.shift_x) / transform.scale_x;
  box->cy = (box->cy - transform.shift_y) / transform.scale_y;
  box->width /= transform.scale_x;
  box->height /= transform.scale_y;
  if (!clip) return std::isfinite(box->angle_rad);
  box->cx = std::max(0.0f, std::min(box->cx, static_cast<float>(image_w)));
  box->cy = std::max(0.0f, std::min(box->cy, static_cast<float>(image_h)));
  box->width = std::max(0.0f, std::min(box->width, static_cast<float>(image_w)));
  box->height = std::max(0.0f, std::min(box->height, static_cast<float>(image_h)));
  return std::isfinite(box->angle_rad);
}

// ---------------------------------------------------------------------------
// Pure NV12 plane packing. Both supported BPU input protocols consume the
// same colour conversion, only the memory layout differs:
//   * packed NV12 (X5 .bin): one contiguous buffer, Y plane followed by an
//     interleaved UV plane, no row padding beyond width alignment.
//   * split NV12 (S100/S100P/S600 .hbm): two UINT8 tensors, Y [1,H,W,1] and
//     UV [1,H/2,W/2,2], each written row-by-row honouring its byte stride.
// ---------------------------------------------------------------------------

// Packs separate I420 planes (as produced by cv::cvtColor(...,
// COLOR_BGR2YUV_I420)) into one contiguous NV12 buffer of h*3/2 rows.
inline void i420_to_packed_nv12(const uint8_t* y, const uint8_t* u,
                                const uint8_t* v, int h, int w, uint8_t* dst) {
  const int y_size = h * w;
  const int uv_plane_size = y_size / 4;
  std::memcpy(dst, y, static_cast<size_t>(y_size));
  uint8_t* uv = dst + y_size;
  for (int i = 0; i < uv_plane_size; ++i) {
    uv[2 * i] = u[i];
    uv[2 * i + 1] = v[i];
  }
}

// Writes the same planes into two stride-padded tensors:
//   y_dst:  h rows of w bytes, advancing `y_stride` bytes per row.
//   uv_dst: h/2 rows of w bytes (u,v interleaved), advancing `uv_stride`
//           bytes per row.
inline void i420_to_split_nv12(const uint8_t* y, const uint8_t* u,
                               const uint8_t* v, int h, int w, uint8_t* y_dst,
                               int y_stride, uint8_t* uv_dst, int uv_stride) {
  for (int row = 0; row < h; ++row) {
    std::memcpy(y_dst + static_cast<std::ptrdiff_t>(row) * y_stride,
                y + static_cast<std::ptrdiff_t>(row) * w,
                static_cast<size_t>(w));
  }
  const int uv_h = h / 2;
  const int uv_w = w / 2;
  for (int row = 0; row < uv_h; ++row) {
    uint8_t* dst_row =
        uv_dst + static_cast<std::ptrdiff_t>(row) * uv_stride;
    const uint8_t* u_row = u + static_cast<std::ptrdiff_t>(row) * uv_w;
    const uint8_t* v_row = v + static_cast<std::ptrdiff_t>(row) * uv_w;
    for (int col = 0; col < uv_w; ++col) {
      dst_row[2 * col] = u_row[col];
      dst_row[2 * col + 1] = v_row[col];
    }
  }
}

// ---------------------------------------------------------------------------
// Stride-validated compact FLOAT32 output copies (pure, host-testable).
// ---------------------------------------------------------------------------

struct FloatOutputPlan {
  OutputShape shape;
  size_t row_bytes, cell_bytes, required_bytes;
};
inline size_t checked_span(size_t step, size_t count, size_t tail,
                           size_t limit) {
  if (tail > limit || (count && step > (limit - tail) / count))
    throw std::invalid_argument("Output strides exceed physical allocation.");
  return step * count + tail;
}
inline FloatOutputPlan nhwc_float_plan(const std::vector<int>& shape,
                                       const std::vector<size_t>& strides,
                                       size_t bytes, bool float32, bool none) {
  if (!float32 || !none || shape.size() != 4 || strides.size() != 4 ||
      shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 ||
      strides[3] != sizeof(float))
    throw std::invalid_argument(
        "Expected batch-one unquantized FLOAT32 NHWC output.");
  size_t cell = checked_span(sizeof(float), shape[3], 0, bytes);
  if (strides[2] < cell || strides[2] % sizeof(float) ||
      strides[1] % sizeof(float))
    throw std::invalid_argument("Invalid output cell/row stride.");
  size_t row = checked_span(strides[2], shape[2] - 1, cell, bytes);
  if (strides[1] < row) throw std::invalid_argument("Overlapping output rows.");
  size_t required = checked_span(strides[1], shape[1] - 1, row, bytes);
  return {{shape[1], shape[2], shape[3]}, strides[1], strides[2], required};
}
inline std::vector<float> copy_float_output(const void* data, size_t bytes,
                                            const FloatOutputPlan& plan) {
  // Revalidate even when called with a caller-constructed plan.
  auto validated =
      nhwc_float_plan({1, plan.shape.h, plan.shape.w, plan.shape.c},
                      {bytes, plan.row_bytes, plan.cell_bytes, sizeof(float)},
                      bytes, true, true);
  if (!data || validated.required_bytes != plan.required_bytes)
    throw std::invalid_argument("Invalid output buffer or plan.");
  std::vector<float> result;
  result.reserve(static_cast<size_t>(plan.shape.h) * plan.shape.w *
                 plan.shape.c);
  const auto* memory = static_cast<const unsigned char*>(data);
  for (int y = 0; y < plan.shape.h; ++y)
    for (int x = 0; x < plan.shape.w; ++x)
      for (int c = 0; c < plan.shape.c; ++c) {
        float value;
        std::memcpy(&value,
                    memory + y * plan.row_bytes + x * plan.cell_bytes +
                        c * sizeof(float),
                    sizeof(float));
        if (!std::isfinite(value))
          throw std::invalid_argument("Output contains nonfinite values.");
        result.push_back(value);
      }
  return result;
}
struct TaskHeadPlan {
  std::array<int, 3> cls, box, extra;
  int prototype = -1;
  bool direct_ltrb = false;
};
inline TaskHeadPlan bind_task_heads(const std::vector<OutputShape>& shapes,
                                    int h, int w, bool segment) {
  if (h <= 0 || h != w || h % 32 || shapes.size() != (segment ? 10u : 9u))
    throw std::invalid_argument(
        "Pose/segment require square stride-32 geometry and exactly 9/10 "
        "heads.");
  TaskHeadPlan plan;
  int box_channels = 0;
  for (int i = 0; i < 3; ++i) {
    const int grid = h / (8 << i);
    plan.cls[i] = find_output_by_shape(shapes, grid, grid, segment ? 80 : 1);
    plan.extra[i] = find_output_by_shape(shapes, grid, grid, segment ? 32 : 51);
    int direct = find_output_by_shape(shapes, grid, grid, 4);
    int dfl = find_output_by_shape(shapes, grid, grid, 64);
    if (plan.cls[i] < 0 || plan.extra[i] < 0 || (direct >= 0) == (dfl >= 0))
      throw std::invalid_argument(
          "Missing, ambiguous or incompatible task output roles.");
    int channels = direct >= 0 ? 4 : 64;
    if (box_channels && box_channels != channels)
      throw std::invalid_argument("Mixed DFL/LTRB output scales.");
    box_channels = channels;
    plan.box[i] = direct >= 0 ? direct : dfl;
  }
  if (segment) {
    plan.prototype = find_output_by_shape(shapes, h / 4, w / 4, 32);
    if (plan.prototype < 0)
      throw std::invalid_argument("Expected a unique stride-4 NHWC prototype.");
  }
  plan.direct_ltrb = box_channels == 4;
  return plan;
}

// Role order for the nine YOLO26 OBB outputs: per stride a class map, a
// 4-channel direct-LTRB box map and a 1-channel angle map.
struct ObbHeadPlan {
  std::array<int, 3> cls, box, angle;
};
inline ObbHeadPlan bind_obb_heads(const std::vector<OutputShape>& shapes,
                                  int h, int w, int classes) {
  if (h <= 0 || h != w || h % 32 || shapes.size() != 9u)
    throw std::invalid_argument(
        "OBB requires square stride-32 geometry and exactly 9 heads.");
  if (classes <= 0)
    throw std::invalid_argument("OBB requires a positive class count.");
  ObbHeadPlan plan;
  for (int i = 0; i < 3; ++i) {
    const int grid = h / (8 << i);
    plan.cls[i] = find_output_by_shape(shapes, grid, grid, classes);
    plan.box[i] = find_output_by_shape(shapes, grid, grid, 4);
    plan.angle[i] = find_output_by_shape(shapes, grid, grid, 1);
    if (plan.cls[i] < 0 || plan.box[i] < 0 || plan.angle[i] < 0)
      throw std::invalid_argument(
          "Expected unique class, LTRB box and angle NHWC outputs per "
          "stride; check --classes (1 and 4 collide with angle/box).");
  }
  return plan;
}

// ---------------------------------------------------------------------------
// 1000-way classification math (pure, host-testable).
// ---------------------------------------------------------------------------

struct ClassificationPlan {
  size_t class_stride;
  size_t required_bytes;
};
struct ClassificationScore {
  int id;
  float probability;
};

// One 1000-class vector, optionally surrounded by singleton dimensions.
// Physical padding is permitted; the class stride must be supplied by metadata.
inline ClassificationPlan classification_plan(const std::vector<int>& shape,
    const std::vector<size_t>& strides, size_t bytes, bool float32, bool unquantized) {
  if (!float32 || !unquantized || shape.empty() || shape.size()>4 || strides.size()!=shape.size())
    throw std::invalid_argument("Classification requires one unquantized FLOAT32 vector.");
  int axis=-1;
  for (size_t i=0;i<shape.size();++i) {
    if (shape[i]!=1) {
      if (shape[i]!=1000 || axis!=-1 || (shape.size()>1 && i==0))
        throw std::invalid_argument("Classification requires a single 1000-class axis and batch one.");
      axis=static_cast<int>(i);
    }
  }
  if (axis<0 || strides[axis]<sizeof(float) || strides[axis]%sizeof(float)!=0 ||
      bytes<sizeof(float) || strides[axis]>(bytes-sizeof(float))/999)
    throw std::invalid_argument("Classification class stride exceeds the output buffer or is invalid.");
  return {strides[axis],999*strides[axis]+sizeof(float)};
}

inline std::vector<ClassificationScore> classification_topk(const void* data,
    size_t bytes, const ClassificationPlan& plan, int topk) {
  if (!data || topk<1 || topk>1000 || plan.class_stride<sizeof(float) ||
      plan.class_stride%sizeof(float)!=0 || bytes<sizeof(float) ||
      plan.class_stride>(bytes-sizeof(float))/999 ||
      plan.required_bytes!=999*plan.class_stride+sizeof(float))
    throw std::invalid_argument("Invalid classification buffer, plan or Top-K.");
  std::vector<double> logits(1000);
  for (int i=0;i<1000;++i) {
    float value;
    std::memcpy(&value,static_cast<const unsigned char*>(data)+i*plan.class_stride,sizeof(value));
    if (!std::isfinite(value)) throw std::invalid_argument("Classification logits must be finite.");
    logits[i]=value;
  }
  const double maximum=*std::max_element(logits.begin(),logits.end());
  double total=0;
  for (double& value:logits) { value=std::exp(value-maximum);total+=value; }
  std::vector<ClassificationScore> result;
  for (int i=0;i<1000;++i) result.push_back({i,static_cast<float>(logits[i]/total)});
  std::partial_sort(result.begin(),result.begin()+topk,result.end(),
    [](const ClassificationScore& a,const ClassificationScore& b) {
      return a.probability!=b.probability ? a.probability>b.probability : a.id<b.id;
    });
  result.resize(topk);
  return result;
}

// ---------------------------------------------------------------------------
// Task model contracts. The declarations are SDK- and OpenCV-free; the
// implementations live in src/<task>.cpp and hold the board resources
// through inc/backend.hpp.
// ---------------------------------------------------------------------------

// Detection (direct-LTRB and DFL heads, six outputs, per-class NMS).
class YoloDetect {
 public:
  struct Config {
    std::string model_path;
    std::string head = "auto";  // auto | dfl | ltrb
    float score_threshold = 0.25f;
    float nms_threshold = 0.7f;
    int resize_type = 1;  // 0 = resize, 1 = letterbox
  };
  // Caller-owned BGR pixels plus source geometry; the CLI loads the image.
  struct Input {
    std::vector<uint8_t> bgr;
    int source_rows = 0;
    int source_cols = 0;
  };
  // Owned per call: compact NV12 planes (interleaved UV) plus the resized
  // frame and transform this call was produced from. No SDK state touches it.
  struct Prepared {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  // Owned per call: compact finite copies of the six head tensors in model
  // output order, the bound role order and the stage geometry.
  struct RawResult {
    std::vector<std::vector<float>> outputs;
    std::vector<OutputShape> shapes;
    std::array<int, 6> output_order = {{-1, -1, -1, -1, -1, -1}};
    bool direct_ltrb = false;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
  };
  struct Result {
    std::vector<Detection> detections;  // source-image coordinates
  };
  struct Prediction {
    Result result;
    std::vector<uint8_t> model_frame_bgr;
    int model_rows = 0;
    int model_cols = 0;
    ImageTransform transform;
  };

  explicit YoloDetect(const Config& config);
  ~YoloDetect();
  YoloDetect(const YoloDetect&) = delete;
  YoloDetect& operator=(const YoloDetect&) = delete;

  int input_h() const;
  int input_w() const;
  bool direct_ltrb() const;

  Prepared preprocess(const Input& input) const;
  RawResult infer(const Prepared& prepared);
  Result postprocess(const RawResult& raw) const;
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Instance segmentation (ten outputs, class-agnostic NMS, prototype masks).
class YoloSegment {
 public:
  struct Config {
    std::string model_path;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int resize_type = 1;  // 0 = resize, 1 = letterbox
  };
  struct Input {
    std::vector<uint8_t> bgr;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Prepared {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct RawResult {
    std::vector<std::vector<float>> outputs;
    TaskHeadPlan heads;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int model_rows = 0;
    int model_cols = 0;
    int source_rows = 0;
    int source_cols = 0;
  };
  // Binary instance mask (values 0 or 255) cropped to the detection box in
  // model-input space, exactly the region the fixed source composited. The
  // origin is the clamped box corner the crop was taken from.
  struct MaskPlane {
    int x = 0;
    int y = 0;
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> bytes;
  };
  struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float x1 = 0.0f;  // model-input coordinates
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    MaskPlane mask;
  };
  struct Result {
    std::vector<Detection> detections;
  };
  struct Prediction {
    Result result;
    std::vector<uint8_t> model_frame_bgr;
    int model_rows = 0;
    int model_cols = 0;
    ImageTransform transform;
  };

  explicit YoloSegment(const Config& config);
  ~YoloSegment();
  YoloSegment(const YoloSegment&) = delete;
  YoloSegment& operator=(const YoloSegment&) = delete;

  int input_h() const;
  int input_w() const;
  bool direct_ltrb() const;

  Prepared preprocess(const Input& input) const;
  RawResult infer(const Prepared& prepared);
  Result postprocess(const RawResult& raw) const;
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Human pose (nine outputs, class-agnostic NMS, COCO 17-keypoint skeletons).
class YoloPose {
 public:
  struct Config {
    std::string model_path;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float kpt_conf_threshold = 0.5f;
    int resize_type = 1;  // 0 = resize, 1 = letterbox
  };
  struct Input {
    std::vector<uint8_t> bgr;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Prepared {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct RawResult {
    std::vector<std::vector<float>> outputs;
    TaskHeadPlan heads;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;  // raw logit confidence, thresholded in logit space
  };
  struct Detection {
    float score = 0.0f;
    float x1 = 0.0f;  // source-image coordinates
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    std::array<Keypoint, 17> keypoints;
  };
  struct Result {
    std::vector<Detection> detections;
  };
  struct Prediction {
    Result result;
    std::vector<uint8_t> model_frame_bgr;
    int model_rows = 0;
    int model_cols = 0;
    ImageTransform transform;
  };

  explicit YoloPose(const Config& config);
  ~YoloPose();
  YoloPose(const YoloPose&) = delete;
  YoloPose& operator=(const YoloPose&) = delete;

  int input_h() const;
  int input_w() const;
  bool direct_ltrb() const;

  Prepared preprocess(const Input& input) const;
  RawResult infer(const Prepared& prepared);
  Result postprocess(const RawResult& raw) const;
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ImageNet classification (one 1000-way logit output).
class YoloClassify {
 public:
  struct Config {
    std::string model_path;
    int topk = 5;
    int resize_type = 1;  // 0 = resize, 1 = letterbox
  };
  struct Input {
    std::vector<uint8_t> bgr;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Prepared {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  // Owned per call: the raw output bytes plus the validated read plan.
  struct RawResult {
    std::vector<uint8_t> output_bytes;
    ClassificationPlan plan{};
    std::vector<uint8_t> model_frame_bgr;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Result {
    std::vector<ClassificationScore> topk;
  };
  struct Prediction {
    Result result;
    std::vector<uint8_t> model_frame_bgr;
    int model_rows = 0;
    int model_cols = 0;
    ImageTransform transform;
  };

  explicit YoloClassify(const Config& config);
  ~YoloClassify();
  YoloClassify(const YoloClassify&) = delete;
  YoloClassify& operator=(const YoloClassify&) = delete;

  int input_h() const;
  int input_w() const;

  Prepared preprocess(const Input& input) const;
  RawResult infer(const Prepared& prepared);
  Result postprocess(const RawResult& raw) const;
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Oriented boxes (YOLO26 direct-LTRB + angle head, nine outputs, DOTA
// classes). Follows the Python decoder's platform policy: X5 wraps angles
// to [-pi/2, pi/2), runs per-class rotated NMS and clips restored boxes;
// the S series runs class-agnostic rotated NMS and keeps unclipped geometry.
class YoloObb {
 public:
  struct Config {
    std::string model_path;
    int classes = 15;  // DOTA-v1 class count of the reference checkpoints
    float score_threshold = 0.25f;
    float nms_threshold = 0.2f;
    float angle_sign = 1.0f;
    float angle_offset_degrees = 0.0f;
    bool regularize_obb = true;
    int resize_type = 1;  // 0 = resize, 1 = letterbox
  };
  struct Input {
    std::vector<uint8_t> bgr;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Prepared {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  // Owned per call: compact finite copies of the nine head tensors in model
  // output order, the bound role order and the stage geometry.
  struct RawResult {
    std::vector<std::vector<float>> outputs;
    ObbHeadPlan heads;
    int classes = 15;
    ImageTransform transform;
    int source_rows = 0;
    int source_cols = 0;
  };
  struct Detection {
    int class_id = 0;
    float score = 0.0f;
    RotatedBox box;  // source-image pixels
  };
  struct Result {
    std::vector<Detection> detections;
  };
  struct Prediction {
    Result result;
    ImageTransform transform;
    int model_rows = 0;
    int model_cols = 0;
  };

  explicit YoloObb(const Config& config);
  ~YoloObb();
  YoloObb(const YoloObb&) = delete;
  YoloObb& operator=(const YoloObb&) = delete;

  int input_h() const;
  int input_w() const;
  int classes() const;

  Prepared preprocess(const Input& input) const;
  RawResult infer(const Prepared& prepared);
  Result postprocess(const RawResult& raw) const;
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo

#endif  // YOLO_RUNTIME_CPP_INC_YOLO_HPP_
