// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// YOLOv5 detect model, one translation unit. The SDK-free gates, head decoder
// and S dequantizer are compiled unconditionally so their accept/reject and
// numeric behaviour stays host-testable; the X5 HB-DNN and S UCP runtime stages
// live in the same file behind compile-time target guards
// (YOLOV5_TARGET_X5 / YOLOV5_TARGET_S), so there are no per-board task files.
// A build without a target macro still compiles and links, but the model
// refuses to construct because it has no build identity.

#include "detect.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(YOLOV5_TARGET_X5)
#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <opencv2/opencv.hpp>
#elif defined(YOLOV5_TARGET_S)
// The S UCP SDK installs its DNN headers under /usr/include/hobot (verified on
// a real S100 2026-09-24): hb_dnn.h lives at hobot/dnn/hb_dnn.h and there is no
// hb_dnn_ext.h. The c_utils headers below expect these to be included first.
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#endif

#ifndef YOLOV5_TARGET_NAME
#define YOLOV5_TARGET_NAME "unknown"
#endif

namespace yolov5 {
namespace {

void require_gate(const Gate& gate) {
  if (!gate) throw std::runtime_error(gate.reason);
}

constexpr long long kFloatBytes = 4;
constexpr long long kInt32Bytes = 4;

bool is_stride_level(int stride) {
  return stride == 8 || stride == 16 || stride == 32;
}

// A flat read is only correct when the aligned layout equals the valid layout.
// An aligned dimension that the runtime did not report (0) is treated as
// "unknown but not padded" so a sparse descriptor is not rejected outright,
// while a reported padded dimension still is.
bool same_layout(const TensorMeta& meta) {
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i) {
    if (meta.aligned[i] > 0 && meta.aligned[i] != meta.valid[i]) return false;
  }
  return true;
}

// Checked arithmetic for the extent proofs: a hostile or corrupted stride must
// be rejected, not wrap around into an accepting comparison.
bool checked_mul(long long a, long long b, long long* out) {
  return !__builtin_mul_overflow(a, b, out);
}

bool checked_add(long long a, long long b, long long* out) {
  return !__builtin_add_overflow(a, b, out);
}

float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

float iou(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return inter / std::max(area_a + area_b - inter, 1.0e-12F);
}

}  // namespace

// -------------------------------------------------- SDK-free gates (all) ----

Gate accept() { return Gate{true, {}}; }

Gate reject(std::string reason) { return Gate{false, std::move(reason)}; }

std::string dtype_name(int code) {
  switch (code) {
    case kDtypeF32: return "float32";
    case kDtypeS32: return "int32";
    case kDtypeS8: return "int8";
    case kDtypeU8: return "uint8";
    case kDtypeS16: return "int16";
    default: return "unknown";
  }
}

std::string quanti_name(int code) {
  switch (code) {
    case kQuantiNone: return "none";
    case kQuantiScale: return "scale";
    case kQuantiShift: return "shift";
    default: return "unknown";
  }
}

Gate check_x5_nv12_input(int image_type, const TensorMeta& meta,
                         long long expected_size) {
  if (image_type != kImageNv12) return reject("X5 input must be a NV12 image tensor");
  if (meta.num_dimensions != 4)
    return reject("X5 input must be a rank-4 NV12 tensor");
  if (meta.valid[0] != 1 || meta.valid[1] != 3 || meta.valid[2] != expected_size ||
      meta.valid[3] != expected_size)
    return reject("X5 input valid shape must be [1,3," + std::to_string(expected_size) +
                  "," + std::to_string(expected_size) + "]");
  if (!same_layout(meta))
    return reject("X5 input has a padded aligned layout; the compact NV12 copy "
                  "would not match the stored layout");
  const long long required = expected_size * expected_size * 3 / 2;
  if (meta.aligned_byte_size < required)
    return reject("X5 input alignedByteSize is smaller than a compact NV12 frame");
  if (meta.storage_bytes < required)
    return reject("X5 input allocation is smaller than a compact NV12 frame");
  return accept();
}

Gate check_x5_head(const TensorMeta& meta, long long input_size, long long classes,
                   int* stride_level) {
  if (stride_level == nullptr) return reject("X5 head gate requires a stride output");
  if (classes <= 0) return reject("X5 head gate requires a positive class count");
  if (meta.dtype != kDtypeF32)
    return reject("X5 YOLOv5 heads must be native F32 tensors");
  if (meta.quanti_type != kQuantiNone)
    return reject("X5 YOLOv5 heads must be unquantized (quantiType NONE)");
  if (meta.num_dimensions != 4)
    return reject("X5 YOLOv5 heads must be rank-4 NHWC tensors");
  if (meta.valid[0] != 1) return reject("X5 YOLOv5 heads must have batch 1");
  const long long height = meta.valid[1];
  const long long width = meta.valid[2];
  const long long channels = meta.valid[3];
  if (height <= 0 || width <= 0 || channels <= 0)
    return reject("X5 YOLOv5 head has a non-positive dimension");
  if (channels != 3 * (5 + classes))
    return reject("X5 YOLOv5 head channels are not 3*(5+classes)");
  if (height != width) return reject("X5 YOLOv5 head must be square");
  if (input_size % height != 0)
    return reject("X5 YOLOv5 head does not divide the network input");
  const long long stride = input_size / height;
  if (!is_stride_level(static_cast<int>(stride)))
    return reject("X5 YOLOv5 head stride is not 8, 16 or 32");
  if (!same_layout(meta))
    return reject("X5 YOLOv5 head has a padded aligned layout; a flat float "
                  "read would not match the stored layout");
  const long long count = height * width * channels;
  if (meta.aligned_byte_size < count * kFloatBytes)
    return reject("X5 YOLOv5 head alignedByteSize cannot hold its float elements");
  if (meta.storage_bytes < count * kFloatBytes)
    return reject("X5 YOLOv5 head allocation cannot hold its float elements");
  *stride_level = static_cast<int>(stride);
  return accept();
}

Gate check_s_nv12_plane(const TensorMeta& meta, long long rows, long long cols,
                        long long channels) {
  if (rows <= 0 || cols <= 0 || channels <= 0)
    return reject("S NV12 plane expects positive dimensions");
  if (meta.num_dimensions != 4)
    return reject("S NV12 plane must be a rank-4 tensor");
  if (meta.valid[0] != 1 || meta.valid[1] != rows || meta.valid[2] != cols ||
      meta.valid[3] != channels)
    return reject("S NV12 plane logical shape does not match the contract");
  const long long row_bytes = cols * channels;
  if (meta.stride[1] < row_bytes)
    return reject("S NV12 plane row stride does not cover a full row");
  if (meta.stride[0] < rows * meta.stride[1])
    return reject("S NV12 plane outer stride does not cover all rows");
  if (meta.storage_bytes < rows * meta.stride[1])
    return reject("S NV12 plane allocation does not cover all rows");
  return accept();
}

Gate check_s32_dequant(const TensorMeta& meta, long long* element_count) {
  if (element_count == nullptr) return reject("S dequant gate requires a count output");
  if (meta.num_dimensions != 4)
    return reject("S YOLOv5 output must be a rank-4 NHWC tensor");
  if (meta.valid[0] != 1) return reject("S YOLOv5 output must have batch 1");
  const long long height = meta.valid[1];
  const long long width = meta.valid[2];
  const long long channels = meta.valid[3];
  if (height <= 0 || width <= 0 || channels <= 0)
    return reject("S YOLOv5 output has a non-positive dimension");

  const long long element_bytes = meta.quanti_type == kQuantiScale ? kInt32Bytes
                                                                  : kFloatBytes;
  if (meta.quanti_type == kQuantiScale) {
    if (meta.dtype != kDtypeS32)
      return reject("S YOLOv5 scaled output must be native S32");
    // A scalar descriptor (length 1) is accepted because the model's private
    // dequant helper broadcasts it; the shared c_utils helper would read
    // scale_data[c] out of bounds, so passing such a tensor to it is forbidden.
    if (meta.scale_len <= 0)
      return reject("S YOLOv5 scaled output has no scale descriptor");
    if (meta.scale_len != 1 && meta.scale_len < channels)
      return reject("S YOLOv5 output scale descriptor is shorter than its channels");
    if (meta.zero_point_len != 0 && meta.zero_point_len != 1 &&
        meta.zero_point_len < channels)
      return reject("S YOLOv5 output zero-point descriptor is shorter than its channels");
    // A per-channel descriptor is only readable channel-wise when the SDK
    // declares the quantization axis as the channel axis (3 for NHWC
    // [N,H,W,C]). An unreported or different axis is rejected rather than
    // guessed; the real artifact's axis is visible in the board dump metadata.
    if (meta.scale_len != 1 && meta.quantize_axis != 3)
      return reject("S YOLOv5 per-channel descriptor requires quantizeAxis == 3 "
                    "(NHWC channel axis); got " +
                    std::to_string(meta.quantize_axis));
  } else if (meta.quanti_type == kQuantiNone) {
    if (meta.dtype != kDtypeF32)
      return reject("S unquantized YOLOv5 output must be native F32");
  } else {
    return reject("S YOLOv5 output has an unsupported quantization type");
  }

  // The addressing both the fixed-source helper and the model's private
  // dequantizer perform reads element (h, w, c) at byte offset
  // (h*W + w) * stride[2] + c * stride[3]:
  //   * stride[3] is the byte distance between channels of one pixel and must
  //     be a positive multiple of the element size (aligned int32/float reads);
  //   * stride[2] is the byte distance between pixels and must cover one full
  //     pixel, i.e. channels elements — a smaller value makes consecutive
  //     pixels overlap and must be rejected (a width*stride[3] bound would
  //     wrongly accept e.g. stride[2]=400 for 84x255 layouts);
  //   * stride[1] must equal width*stride[2] exactly, because the formula
  //     places every row uniformly stride[2] after the previous one.
  if (meta.stride[3] < element_bytes || meta.stride[3] % element_bytes != 0)
    return reject("S YOLOv5 output channel stride is not element-aligned");
  if (meta.stride[2] < element_bytes || meta.stride[2] % element_bytes != 0)
    return reject("S YOLOv5 output pixel stride is not element-aligned");
  long long pixel_bytes = 0;
  if (!checked_mul(channels, meta.stride[3], &pixel_bytes) ||
      meta.stride[2] < pixel_bytes)
    return reject("S YOLOv5 output pixel stride does not cover all channels");
  long long row_bytes = 0;
  if (!checked_mul(width, meta.stride[2], &row_bytes) || meta.stride[1] != row_bytes)
    return reject("S YOLOv5 output stride[1] must equal width*stride[2]; the "
                  "dequantizer addresses every row at a uniform row stride and "
                  "would misread H-level padding");
  long long count = 0;
  long long plane = 0;
  if (!checked_mul(height, width, &plane) || !checked_mul(plane, channels, &count))
    return reject("S YOLOv5 output element count overflows");
  // Exact last byte the addressing can touch: the final element of the final
  // pixel of the final row — (channels-1)*stride[3] + element_bytes inside the
  // last pixel, not channels*stride[3], which would also demand the channel
  // padding after the final element. Every product is overflow-checked.
  long long last_pixel = 0;
  long long tail = 0;
  long long required = 0;
  if (!checked_mul(plane - 1, meta.stride[2], &last_pixel) ||
      !checked_mul(channels - 1, meta.stride[3], &tail) ||
      !checked_add(last_pixel, tail, &required) ||
      !checked_add(required, element_bytes, &required))
    return reject("S YOLOv5 output layout extent overflows");
  if (meta.aligned_byte_size < required)
    return reject("S YOLOv5 output alignedByteSize cannot hold its stored layout");
  if (meta.storage_bytes < required)
    return reject("S YOLOv5 output allocation cannot hold its stored layout");
  *element_count = count;
  return accept();
}

Gate check_target_matches_build(const std::string& requested,
                                const std::string& build_target) {
  if (build_target.empty() || build_target == "unknown")
    return reject("Native binary has no compiled build target identity");
  if (requested != build_target)
    return reject("Requested target '" + requested + "' does not match the compiled "
                  "build target '" + build_target + "'");
  return accept();
}

// ----------------------------------------------------------- head decoder ----

bool validate_head_shapes(const std::vector<HeadShape>& heads,
                          int input_size, int classes) {
  if (input_size <= 0 || classes <= 0 || heads.size() != 3) return false;
  const int channels = 3 * (5 + classes);
  std::set<int> strides;
  for (const auto& head : heads) {
    if (head.height != head.width || head.height <= 0 ||
        head.channels != channels || input_size % head.height != 0) {
      return false;
    }
    const int stride = input_size / head.height;
    if (stride != 8 && stride != 16 && stride != 32) return false;
    strides.insert(stride);
  }
  return strides.size() == 3;
}

std::vector<int> order_heads_by_shape(const std::vector<HeadShape>& heads,
                                      int input_size, int classes) {
  if (!validate_head_shapes(heads, input_size, classes))
    throw std::invalid_argument("YOLOv5 output heads are not a unique 8/16/32 contract");
  std::vector<int> ordered;
  for (int stride : {8, 16, 32}) {
    auto it = std::find_if(heads.begin(), heads.end(), [&](const HeadShape& h) {
      return input_size / h.height == stride;
    });
    ordered.push_back(static_cast<int>(std::distance(heads.begin(), it)));
  }
  return ordered;
}

std::vector<Detection> decode_heads(
    const std::vector<std::vector<float>>& raw_heads,
    const std::vector<HeadShape>& heads, int input_size, int classes,
    const DecodePolicy& policy,
    const std::array<float, 18>& anchors) {
  if (raw_heads.size() != heads.size() || !std::isfinite(policy.score_threshold) ||
      !std::isfinite(policy.nms_threshold) || policy.score_threshold < 0.0F ||
      policy.score_threshold > 1.0F || policy.nms_threshold < 0.0F ||
      policy.nms_threshold > 1.0F || policy.top_k_per_class == 0 ||
      policy.top_k_per_class < -1)
    throw std::invalid_argument("Invalid YOLOv5 decode arguments");
  const auto ordered = order_heads_by_shape(heads, input_size, classes);
  const int channels = 3 * (5 + classes);
  std::vector<Detection> candidates;
  for (std::size_t level = 0; level < ordered.size(); ++level) {
    const int source = ordered[level];
    const auto& shape = heads[source];
    const auto& raw = raw_heads[source];
    const std::size_t expected = static_cast<std::size_t>(shape.height) *
                                 static_cast<std::size_t>(shape.width) * channels;
    if (raw.size() != expected)
      throw std::invalid_argument("YOLOv5 output buffer does not match metadata");
    const int stride = input_size / shape.height;
    for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {
        for (int a = 0; a < 3; ++a) {
          const std::size_t base = (static_cast<std::size_t>(y) * shape.width + x) * channels +
                                   static_cast<std::size_t>(a) * (5 + classes);
          const float objectness = sigmoid(raw[base + 4]);
          if (!std::isfinite(objectness)) continue;
          // Both preserved C++ sources select one maximum class per anchor before
          // confidence filtering; emitting every class changes the source contract.
          int cls = 0;
          for (int candidate = 1; candidate < classes; ++candidate)
            if (raw[base + 5 + candidate] > raw[base + 5 + cls]) cls = candidate;
          const float score = objectness * sigmoid(raw[base + 5 + cls]);
          if (!std::isfinite(score)) continue;
          if (policy.strict_score_boundary ? !(score > policy.score_threshold)
                                           : score < policy.score_threshold)
            continue;
          const float cx = (2.0F * sigmoid(raw[base]) - 0.5F + x) * stride;
          const float cy = (2.0F * sigmoid(raw[base + 1]) - 0.5F + y) * stride;
          const float w = std::pow(2.0F * sigmoid(raw[base + 2]), 2.0F) * anchors[level * 6 + a * 2];
          const float h = std::pow(2.0F * sigmoid(raw[base + 3]), 2.0F) * anchors[level * 6 + a * 2 + 1];
          candidates.push_back({cx - w / 2.0F, cy - h / 2.0F, cx + w / 2.0F,
                                cy + h / 2.0F, score, cls});
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });
  std::vector<Detection> result;
  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<int> kept_per_class(static_cast<std::size_t>(classes), 0);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) continue;
    const int cls = candidates[i].class_id;
    // Candidates are sorted by descending score, so once a class reaches its
    // cap every remaining candidate of that class is dropped as well; this is
    // the source X5 NMSBoxes top_k break.
    if (policy.top_k_per_class > 0 && kept_per_class[static_cast<std::size_t>(cls)] >=
                                          policy.top_k_per_class)
      continue;
    result.push_back(candidates[i]);
    ++kept_per_class[static_cast<std::size_t>(cls)];
    for (std::size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
          iou(candidates[i], candidates[j]) > policy.nms_threshold)
        suppressed[j] = true;
    }
  }
  return result;
}

// ------------------------------------------------------- S numeric helpers ----

std::vector<float> dequant_s32_nhwc(const unsigned char* base, const TensorMeta& meta,
                                    const float* scale_data, long long scale_len,
                                    const std::int32_t* zero_point_data,
                                    long long zero_point_len) {
  if (meta.quanti_type != kQuantiScale && meta.quanti_type != kQuantiNone)
    throw std::invalid_argument("dequant_s32_nhwc: unsupported quantization kind");
  if (meta.num_dimensions != 4 || meta.valid[0] != 1 || base == nullptr)
    throw std::invalid_argument("dequant_s32_nhwc: expected a batch-1 rank-4 buffer");
  if (meta.quanti_type == kQuantiScale &&
      (scale_data == nullptr || (scale_len != 1 && scale_len < meta.valid[3])))
    throw std::invalid_argument("dequant_s32_nhwc: unusable scale descriptor");
  if (meta.quanti_type == kQuantiScale && zero_point_len != 0 && zero_point_len != 1 &&
      zero_point_len < meta.valid[3])
    throw std::invalid_argument("dequant_s32_nhwc: unusable zero-point descriptor");
  // A declared zero point without a buffer must never reach the read below.
  if (meta.quanti_type == kQuantiScale && zero_point_len > 0 &&
      zero_point_data == nullptr)
    throw std::invalid_argument("dequant_s32_nhwc: zero-point length without data");

  const long long height = meta.valid[1];
  const long long width = meta.valid[2];
  const long long channels = meta.valid[3];
  std::vector<float> result(static_cast<std::size_t>(height * width * channels));
  const bool scalar_scale = scale_len == 1;
  const bool scalar_zero = zero_point_len == 1;
  std::size_t index = 0;
  for (long long h = 0; h < height; ++h) {
    for (long long w = 0; w < width; ++w) {
      const std::size_t pixel = static_cast<std::size_t>((h * width + w) * meta.stride[2]);
      for (long long c = 0; c < channels; ++c) {
        const std::size_t offset = pixel + static_cast<std::size_t>(c * meta.stride[3]);
        if (meta.quanti_type == kQuantiScale) {
          const float scale = scalar_scale ? scale_data[0] : scale_data[c];
          const std::int32_t zero =
              zero_point_len == 0 ? 0 : (scalar_zero ? zero_point_data[0] : zero_point_data[c]);
          std::int32_t quantized = 0;
          std::memcpy(&quantized, base + offset, sizeof(quantized));
          result[index] = (static_cast<float>(quantized) - static_cast<float>(zero)) * scale;
        } else {
          float value = 0.0F;
          std::memcpy(&value, base + offset, sizeof(value));
          result[index] = value;
        }
        ++index;
      }
    }
  }
  return result;
}

bool bpu_core_to_backend(long long bpu_core, unsigned long long* backend) {
  if (backend == nullptr) return false;
  if (bpu_core == -1) {
    *backend = 1ULL << 7;  // HB_UCP_BPU_CORE_ANY
    return true;
  }
  if (bpu_core < 0 || bpu_core > 3) return false;
  *backend = 1ULL << static_cast<unsigned>(bpu_core);  // HB_UCP_BPU_CORE_0..3
  return true;
}

// --------------------------------------------------------- evidence fills ----

DumpTensorInfo dump_tensor_info(const std::string& name, const TensorMeta& meta,
                                const float* scale_data,
                                const std::int32_t* zero_point_data) {
  DumpTensorInfo info;
  info.name = name;
  info.dtype = dtype_name(meta.dtype);
  info.shape.clear();
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i)
    info.shape.push_back(meta.valid[i]);
  info.quanti = quanti_name(meta.quanti_type);
  info.scale_len = meta.scale_len;
  info.aligned_byte_size = meta.aligned_byte_size;
  // A non-positive entry means the projection never saw a reported value
  // (TensorMeta zero-initializes; the X5 gate treats aligned 0 the same way),
  // so it is recorded as unreported rather than as a fake zero stride.
  for (int i = 0; i < 4; ++i) info.stride[i] = meta.stride[i] > 0 ? meta.stride[i] : -1;
  for (int i = 0; i < 4; ++i) info.aligned[i] = meta.aligned[i] > 0 ? meta.aligned[i] : -1;
  info.quantize_axis = meta.quantize_axis;
  if (meta.quanti_type == kQuantiScale) {
    if (scale_data == nullptr || meta.scale_len <= 0)
      throw std::invalid_argument("dump_tensor_info: SCALE tensor without a readable "
                                  "scale descriptor: " + name);
    if (meta.scale_len > kMaxQuantValues)
      throw std::invalid_argument("dump_tensor_info: scale descriptor exceeds the "
                                  "recorded bound (" + std::to_string(meta.scale_len) +
                                  ")");
    info.scale_values.assign(scale_data, scale_data + meta.scale_len);
    if (meta.zero_point_len > 0) {
      if (zero_point_data == nullptr)
        throw std::invalid_argument("dump_tensor_info: zero-point length without a "
                                    "buffer: " + name);
      if (meta.zero_point_len > kMaxQuantValues)
        throw std::invalid_argument("dump_tensor_info: zero-point descriptor exceeds "
                                    "the recorded bound");
      info.zero_point_values.assign(zero_point_data,
                                    zero_point_data + meta.zero_point_len);
    }
  }
  return info;
}

std::vector<Detection> map_to_original(const std::vector<Detection>& detections,
                                       int image_cols, int image_rows, int model_size) {
  // Mirrors render_detections: uniform letterbox scale to the square model
  // input, symmetric padding, then the inverse mapping per coordinate. The
  // results are unclamped floats — they describe the detection, not the
  // pixels the renderer ends up drawing.
  std::vector<Detection> mapped;
  if (image_cols <= 0 || image_rows <= 0 || model_size <= 0) return mapped;
  const double scale = std::min(static_cast<double>(model_size) / image_cols,
                                static_cast<double>(model_size) / image_rows);
  const double pad_x = (model_size - image_cols * scale) / 2.0;
  const double pad_y = (model_size - image_rows * scale) / 2.0;
  mapped.reserve(detections.size());
  for (const auto& detection : detections) {
    Detection out;
    out.x1 = static_cast<float>((detection.x1 - pad_x) / scale);
    out.y1 = static_cast<float>((detection.y1 - pad_y) / scale);
    out.x2 = static_cast<float>((detection.x2 - pad_x) / scale);
    out.y2 = static_cast<float>((detection.y2 - pad_y) / scale);
    out.score = detection.score;
    out.class_id = detection.class_id;
    mapped.push_back(out);
  }
  return mapped;
}

// ============================================================ X5 backend ====

#if defined(YOLOV5_TARGET_X5)

namespace {

constexpr int kClasses = 80;
constexpr int kInput = 640;
// Source X5 main.cc passes NMS_TOP_K = 300 to cv::dnn::NMSBoxes per class.
constexpr int kX5TopK = 300;
constexpr std::array<float, 18> kAnchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

int tensor_dtype_code(int32_t type) {
  if (type == HB_DNN_TENSOR_TYPE_F32) return kDtypeF32;
  if (type == HB_DNN_TENSOR_TYPE_S32) return kDtypeS32;
  if (type == HB_DNN_TENSOR_TYPE_S8) return kDtypeS8;
  if (type == HB_DNN_TENSOR_TYPE_U8) return kDtypeU8;
  if (type == HB_DNN_TENSOR_TYPE_S16) return kDtypeS16;
  return kDtypeUnknown;
}

int quanti_code(int32_t type) {
  if (type == NONE) return kQuantiNone;
  if (type == SCALE) return kQuantiScale;
  if (type == SHIFT) return kQuantiShift;
  return kQuantiUnknown;
}

int image_code(int32_t type) {
  if (type == HB_DNN_IMG_TYPE_NV12) return kImageNv12;
  return kImageUnknown;
}

TensorMeta project(const hbDNNTensorProperties& properties) {
  TensorMeta meta;
  meta.dtype = tensor_dtype_code(properties.tensorType);
  meta.quanti_type = quanti_code(properties.quantiType);
  meta.num_dimensions = properties.validShape.numDimensions;
  for (int i = 0; i < 4 && i < properties.validShape.numDimensions; ++i)
    meta.valid[i] = properties.validShape.dimensionSize[i];
  for (int i = 0; i < 4 && i < properties.alignedShape.numDimensions; ++i)
    meta.aligned[i] = properties.alignedShape.dimensionSize[i];
  meta.aligned_byte_size = properties.alignedByteSize;
  // The X5 backend allocates exactly alignedByteSize for every input and
  // output, so that is the storage the gates must hold the reads against.
  meta.storage_bytes = properties.alignedByteSize;
  for (int i = 0; i < 4; ++i) meta.stride[i] = properties.stride[i];
  meta.quantize_axis = properties.quantizeAxis;
  if (properties.quantiType == SCALE) {
    meta.scale_len = properties.scale.scaleLen;
    meta.zero_point_len = properties.scale.zeroPointLen;
  }
  return meta;
}

std::vector<long long> shape_of(const TensorMeta& meta) {
  std::vector<long long> shape;
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i) shape.push_back(meta.valid[i]);
  return shape;
}

struct DnnLease {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNTaskHandle_t task = nullptr;
  hbDNNTensor input{};
  std::vector<hbDNNTensor> outputs;
  std::vector<bool> output_allocated;
  bool input_allocated = false;
  ~DnnLease() {
    if (task) hbDNNReleaseTask(task);
    if (input_allocated) hbSysFreeMem(&input.sysMem[0]);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      if (i < output_allocated.size() && output_allocated[i])
        hbSysFreeMem(&outputs[i].sysMem[0]);
    if (packed) hbDNNRelease(packed);
  }
};

void check(int code, const char* what) {
  if (code != 0) throw std::runtime_error(what);
}

std::vector<unsigned char> letterbox_to_nv12(const cv::Mat& image) {
  if (image.empty()) throw std::invalid_argument("X5 image is empty");
  const double scale = std::min(static_cast<double>(kInput) / image.rows,
                                static_cast<double>(kInput) / image.cols);
  if (!(scale > 0.0)) throw std::invalid_argument("X5 image has invalid dimensions");
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(static_cast<int>(image.cols * scale),
                                      static_cast<int>(image.rows * scale)));
  cv::Mat boxed(kInput, kInput, CV_8UC3, cv::Scalar(127, 127, 127));
  resized.copyTo(boxed(cv::Rect((kInput - resized.cols) / 2,
                               (kInput - resized.rows) / 2,
                               resized.cols, resized.rows)));
  cv::Mat yuv;
  cv::cvtColor(boxed, yuv, cv::COLOR_BGR2YUV_I420);
  const std::size_t y_size = static_cast<std::size_t>(kInput) * kInput;
  const std::size_t uv_size = y_size / 4;
  std::vector<unsigned char> bytes(y_size + 2 * uv_size);
  std::memcpy(bytes.data(), yuv.data, y_size);
  const auto* u = yuv.data + y_size;
  const auto* v = u + uv_size;
  for (std::size_t i = 0; i < uv_size; ++i) {
    bytes[y_size + 2 * i] = u[i];
    bytes[y_size + 2 * i + 1] = v[i];
  }
  return bytes;
}

std::vector<unsigned char> to_bytes(const std::vector<float>& values) {
  std::vector<unsigned char> bytes(values.size() * sizeof(float));
  if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

}  // namespace

struct Yolov5::Impl {
  explicit Impl(const Config& cfg) : config(cfg) {
    require_gate(check_target_matches_build(config.target, YOLOV5_TARGET_NAME));
    if (config.priority != 0 || config.bpu_core != -1)
      throw std::invalid_argument(
          "X5 adapter has no verified HB-DNN mapping for --priority/--bpu-core; "
          "only the defaults (priority 0, bpu-core -1) are accepted on x5");

    hbPackedDNNHandle_t packed = nullptr;
    const char* file = config.model_path.c_str();
    check(hbDNNInitializeFromFiles(&packed, &file, 1), "hbDNNInitializeFromFiles failed");
    lease.packed = packed;
    const char** names = nullptr;
    int model_count = 0;
    check(hbDNNGetModelNameList(&names, &model_count, lease.packed),
          "hbDNNGetModelNameList failed");
    if (model_count != 1)
      throw std::runtime_error("YOLOv5 X5 asset must contain exactly one model");
    check(hbDNNGetModelHandle(&model, lease.packed, names[0]), "hbDNNGetModelHandle failed");
    int32_t input_count = 0;
    check(hbDNNGetInputCount(&input_count, model), "hbDNNGetInputCount failed");
    if (input_count != 1) throw std::runtime_error("YOLOv5 X5 requires one input");
    int32_t output_count = 0;
    check(hbDNNGetOutputCount(&output_count, model), "hbDNNGetOutputCount failed");
    if (output_count != 3) throw std::runtime_error("YOLOv5 X5 requires three outputs");

    hbDNNTensorProperties input_properties{};
    check(hbDNNGetInputTensorProperties(&input_properties, model, 0),
          "hbDNNGetInputTensorProperties failed");
    input_meta = project(input_properties);
    require_gate(
        check_x5_nv12_input(image_code(input_properties.tensorType), input_meta, kInput));
    // The X5 SDK encodes an image input's format in tensorType (HB_DNN_IMG_TYPE_
    // NV12 for this model), which no data-type name describes; the storage the
    // compact payload occupies is 8-bit, so the dump records uint8.
    input_meta.dtype = kDtypeU8;
    lease.input.properties = input_properties;

    run_options = {{"score_thres", std::to_string(config.score_threshold)},
                   {"nms_thres", std::to_string(config.nms_threshold)},
                   {"priority", std::to_string(config.priority)},
                   {"bpu_core", std::to_string(config.bpu_core)},
                   {"preprocess", "letterbox"},
                   {"nms_top_k_per_class", std::to_string(kX5TopK)},
                   {"score_boundary", "strict-greater-than"}};
    notes = {"X5 HB-DNN adapter; packed NV12 640x640 input; native F32 NONE-quantized heads.",
             "NMS preserves source X5 per-class cv::dnn::NMSBoxes semantics: strict score "
             "boundary and top_k=300.",
             "Dump contents are read back from the same buffers the decoder consumed.",
             "The input tensor file holds the compact NV12 payload actually submitted; "
             "storage beyond the payload is uninitialized and is not dumped."};
  }

  int input_size() const { return kInput; }

  // Pure pixel -> payload conversion. No SDK call and no instance mutation:
  // the returned Prepared owns everything this call produced, so a later
  // preprocess can never rewrite it.
  Prepared preprocess(const Input& input) {
    if (input.source_rows <= 0 || input.source_cols <= 0 ||
        input.bgr.size() != static_cast<std::size_t>(input.source_rows) *
                                static_cast<std::size_t>(input.source_cols) * 3)
      throw std::invalid_argument("X5 input pixels do not match the declared source geometry");
    const cv::Mat source(input.source_rows, input.source_cols, CV_8UC3,
                         const_cast<unsigned char*>(input.bgr.data()));
    auto nv12 = letterbox_to_nv12(source);
    if (nv12.size() > static_cast<std::size_t>(input_meta.aligned_byte_size))
      throw std::runtime_error("X5 NV12 payload exceeds aligned input storage");
    Prepared prepared;
    prepared.source_cols = input.source_cols;
    prepared.source_rows = input.source_rows;
    prepared.nv12 = std::move(nv12);
    return prepared;
  }

  // Uploads exactly the prepared argument, runs the forward pass and copies
  // the raw heads plus the stage evidence into the returned value.
  RawResult infer(const Prepared& prepared) {
    // The argument is the contract: exactly one packed NV12 frame of the
    // compiled input size with positive source geometry. A hand-built or
    // truncated Prepared is rejected before any SDK allocation or copy.
    if (prepared.source_cols <= 0 || prepared.source_rows <= 0 ||
        prepared.nv12.size() != static_cast<std::size_t>(kInput) * kInput * 3 / 2)
      throw std::invalid_argument(
          "X5 prepared input does not match the packed NV12 contract");
    const long long input_bytes = input_meta.aligned_byte_size;
    if (lease.input_allocated) {
      hbSysFreeMem(&lease.input.sysMem[0]);
      lease.input_allocated = false;
    }
    check(hbSysAllocCachedMem(&lease.input.sysMem[0], static_cast<int>(input_bytes)),
          "hbSysAllocCachedMem input failed");
    lease.input_allocated = true;
    std::memcpy(lease.input.sysMem[0].virAddr, prepared.nv12.data(), prepared.nv12.size());
    check(hbSysFlushMem(&lease.input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem input failed");

    const int output_count = 3;
    if (lease.task) {
      hbDNNReleaseTask(lease.task);
      lease.task = nullptr;
    }
    for (std::size_t i = 0; i < lease.outputs.size(); ++i) {
      if (i < lease.output_allocated.size() && lease.output_allocated[i]) {
        hbSysFreeMem(&lease.outputs[i].sysMem[0]);
        lease.output_allocated[i] = false;
      }
    }
    lease.outputs.assign(static_cast<std::size_t>(output_count), hbDNNTensor{});
    lease.output_allocated.assign(static_cast<std::size_t>(output_count), false);
    std::vector<HeadShape> shapes;
    std::set<int> seen_strides;
    RawResult raw;
    raw.source_cols = prepared.source_cols;
    raw.source_rows = prepared.source_rows;
    raw.inputs.push_back(dump_tensor_info("input0", input_meta));
    raw.input_tensors.push_back({"input0", "uint8", shape_of(input_meta), prepared.nv12});
    for (int i = 0; i < output_count; ++i) {
      hbDNNTensorProperties properties{};
      check(hbDNNGetOutputTensorProperties(&properties, model, i),
            "hbDNNGetOutputTensorProperties failed");
      const TensorMeta meta = project(properties);
      int stride_level = 0;
      const Gate gate = check_x5_head(meta, kInput, kClasses, &stride_level);
      if (!gate) throw std::runtime_error("X5 output " + std::to_string(i) + ": " + gate.reason);
      if (!seen_strides.insert(stride_level).second)
        throw std::runtime_error("X5 output stride is duplicated across heads");
      shapes.push_back({static_cast<int>(meta.valid[1]), static_cast<int>(meta.valid[2]),
                        static_cast<int>(meta.valid[3])});
      raw.outputs.push_back(dump_tensor_info("output" + std::to_string(i), meta));
      lease.outputs[static_cast<std::size_t>(i)].properties = properties;
      check(hbSysAllocCachedMem(&lease.outputs[static_cast<std::size_t>(i)].sysMem[0],
                                static_cast<int>(meta.aligned_byte_size)),
            "hbSysAllocCachedMem output failed");
      lease.output_allocated[static_cast<std::size_t>(i)] = true;
    }
    if (!validate_head_shapes(shapes, kInput, kClasses))
      throw std::runtime_error("X5 output metadata does not match published heads");

    hbDNNTensor* output_ptr = lease.outputs.data();
    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    check(hbDNNInfer(&lease.task, &output_ptr, &lease.input, model, &ctrl),
          "hbDNNInfer failed");
    check(hbDNNWaitTaskDone(lease.task, 0), "hbDNNWaitTaskDone failed");
    for (auto& tensor : lease.outputs)
      check(hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
            "hbSysFlushMem output failed");

    raw.shapes = shapes;
    raw.heads.reserve(lease.outputs.size());
    for (std::size_t i = 0; i < lease.outputs.size(); ++i) {
      const auto& shape = shapes[i];
      const std::size_t count = static_cast<std::size_t>(shape.height) *
                                static_cast<std::size_t>(shape.width) *
                                static_cast<std::size_t>(shape.channels);
      const float* values = static_cast<const float*>(lease.outputs[i].sysMem[0].virAddr);
      raw.heads.emplace_back(values, values + count);
      raw.raw_tensors.push_back({"output" + std::to_string(i), "float32",
                                 {shape.height, shape.width, shape.channels},
                                 to_bytes(raw.heads.back())});
      raw.transformed_tensors.push_back({"output" + std::to_string(i), "float32",
                                         {shape.height, shape.width, shape.channels},
                                         to_bytes(raw.heads.back())});
    }
    return raw;
  }

  Result postprocess(const RawResult& raw) {
    DecodePolicy policy;
    policy.score_threshold = config.score_threshold;
    policy.nms_threshold = config.nms_threshold;
    policy.top_k_per_class = kX5TopK;
    policy.strict_score_boundary = true;
    Result result;
    result.detections =
        decode_heads(raw.heads, raw.shapes, kInput, kClasses, policy, kAnchors);
    return result;
  }

  // Assembles the per-run evidence of exactly one call: the stage records
  // carried by the raw value plus the config-derived identity and the
  // detections mapped with this call's geometry.
  RunEvidence collect_evidence(const RawResult& raw, const Result& result) const {
    RunEvidence evidence;
    evidence.target = config.target;
    evidence.build_target = YOLOV5_TARGET_NAME;
    evidence.model_path = config.model_path;
    evidence.options = run_options;
    evidence.notes = notes;
    evidence.inputs = raw.inputs;
    evidence.outputs = raw.outputs;
    evidence.input_tensors = raw.input_tensors;
    evidence.raw_tensors = raw.raw_tensors;
    evidence.transformed_tensors = raw.transformed_tensors;
    evidence.detections = result.detections;
    evidence.detections_original =
        map_to_original(result.detections, raw.source_cols, raw.source_rows, kInput);
    return evidence;
  }

  Config config;
  DnnLease lease;
  hbDNNHandle_t model = nullptr;
  TensorMeta input_meta;
  std::vector<std::pair<std::string, std::string>> run_options;
  std::vector<std::string> notes;
};

// ============================================================== S backend ====

#elif defined(YOLOV5_TARGET_S)

namespace {

constexpr int kInputSize = 672;
constexpr int kClasses = 80;
constexpr std::array<float, 18> kAnchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

int tensor_dtype_code(int32_t type) {
  // Enum names confirmed present in the real S100 hb_dnn.h (on-board evidence
  // 2026-09-24): S8/U8/S16/F32/S32 among others. Values outside the known set
  // stay unknown so the gates reject them.
  if (type == HB_DNN_TENSOR_TYPE_F32) return kDtypeF32;
  if (type == HB_DNN_TENSOR_TYPE_S32) return kDtypeS32;
  if (type == HB_DNN_TENSOR_TYPE_S8) return kDtypeS8;
  if (type == HB_DNN_TENSOR_TYPE_U8) return kDtypeU8;
  if (type == HB_DNN_TENSOR_TYPE_S16) return kDtypeS16;
  return kDtypeUnknown;
}

int quanti_code(int32_t type) {
  // The real S100 hbDNNQuantiType enum only defines NONE and SCALE (verified
  // on-board); any other numeric value stays unknown so the gate rejects it
  // instead of silently misinterpreting it.
  if (type == NONE) return kQuantiNone;
  if (type == SCALE) return kQuantiScale;
  return kQuantiUnknown;
}

TensorMeta project(const hbDNNTensorProperties& properties) {
  TensorMeta meta;
  meta.dtype = tensor_dtype_code(properties.tensorType);
  meta.quanti_type = quanti_code(properties.quantiType);
  meta.num_dimensions = properties.validShape.numDimensions;
  for (int i = 0; i < 4 && i < properties.validShape.numDimensions; ++i)
    meta.valid[i] = properties.validShape.dimensionSize[i];
  // The real S100 hbDNNTensorProperties has no alignedShape field (verified
  // on-board): the stored layout is described by stride[] plus alignedByteSize,
  // which is exactly what the S gates validate, so aligned[] stays unreported.
  meta.aligned_byte_size = properties.alignedByteSize;
  // Outputs are allocated at alignedByteSize; input planes are allocated by
  // prepare_input_tensor as stride[0] * batch, which preprocess overrides.
  meta.storage_bytes = properties.alignedByteSize;
  for (int i = 0; i < 4; ++i) meta.stride[i] = properties.stride[i];
  meta.quantize_axis = properties.quantizeAxis;
  if (properties.quantiType == SCALE) {
    meta.scale_len = properties.scale.scaleLen;
    meta.zero_point_len = properties.scale.zeroPointLen;
  }
  return meta;
}

std::vector<long long> shape_of(const TensorMeta& meta) {
  std::vector<long long> shape;
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i) shape.push_back(meta.valid[i]);
  return shape;
}

// Converts the letterboxed BGR frame into owned compact NV12 planes with
// exactly the arithmetic c_utils bgr_to_nv12_tensor performs (BGR -> I420,
// then the U/V planes interleaved into the NV12 UV plane): the same
// cvtColor call, the same per-row copies, only the destination is the
// caller-owned Prepared payload at a compact pitch instead of SDK sysMem.
// The row-by-row upload into the strided SDK planes happens in infer.
void bgr_to_nv12_planes(const cv::Mat& boxed, int input_h, int input_w,
                        std::vector<unsigned char>* y_plane,
                        std::vector<unsigned char>* uv_plane) {
  cv::Mat yuv;
  cv::cvtColor(boxed, yuv, cv::COLOR_BGR2YUV_I420);
  const unsigned char* yuv_data = yuv.ptr<unsigned char>();
  const int uv_height = input_h / 2;
  const int uv_width = input_w / 2;
  y_plane->assign(yuv_data, yuv_data + input_h * input_w);
  const unsigned char* u_data = yuv_data + input_h * input_w;
  const unsigned char* v_data = u_data + uv_height * uv_width;
  uv_plane->assign(static_cast<std::size_t>(uv_height) * uv_width * 2, 0);
  for (int h = 0; h < uv_height; ++h) {
    for (int w = 0; w < uv_width; ++w) {
      (*uv_plane)[static_cast<std::size_t>(2 * (h * uv_width + w))] =
          u_data[h * uv_width + w];
      (*uv_plane)[static_cast<std::size_t>(2 * (h * uv_width + w)) + 1] =
          v_data[h * uv_width + w];
    }
  }
}

// Frees only tensors that were actually allocated: a partially failed
// allocation must not turn into a blind free of a zeroed sysMem.
struct SResourceGuard {
  // The real S100 SDK spells the packed handle hbDNNPackedHandle_t (on-board
  // compile evidence 2026-09-24); hbPackedDNNHandle_t is the X5 spelling.
  hbDNNPackedHandle_t packed = nullptr;
  hbUCPTaskHandle_t task = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  ~SResourceGuard() {
    if (task) hbUCPReleaseTask(task);
    for (auto& tensor : inputs)
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    for (auto& tensor : outputs)
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    if (packed) hbDNNRelease(packed);
  }
};

}  // namespace

struct Yolov5::Impl {
  explicit Impl(const Config& cfg) : config(cfg) {
    require_gate(check_target_matches_build(config.target, YOLOV5_TARGET_NAME));
    if (config.target != "s100" && config.target != "s600")
      throw std::invalid_argument("S adapter supports s100 and s600 YOLOv5 assets only");
    if (!bpu_core_to_backend(config.bpu_core, &backend))
      throw std::invalid_argument("--bpu-core must be -1 (any) or a core index 0..3");

    hbDNNPackedHandle_t packed = nullptr;
    const char* file = config.model_path.c_str();
    if (hbDNNInitializeFromFiles(&packed, &file, 1) != 0)
      throw std::runtime_error("hbDNNInitializeFromFiles failed");
    guard.packed = packed;
    const char** model_names = nullptr;
    int model_count = 0;
    if (hbDNNGetModelNameList(&model_names, &model_count, guard.packed) != 0 ||
        model_count != 1)
      throw std::runtime_error("S YOLOv5 asset must contain exactly one model");
    if (hbDNNGetModelHandle(&model, guard.packed, model_names[0]) != 0)
      throw std::runtime_error("hbDNNGetModelHandle failed");
    int32_t input_count = 0;
    int32_t output_count = 0;
    if (hbDNNGetInputCount(&input_count, model) != 0 || input_count != 2 ||
        hbDNNGetOutputCount(&output_count, model) != 0 || output_count != 3)
      throw std::runtime_error("S YOLOv5 requires two NV12 inputs and three outputs");

    guard.inputs.resize(static_cast<std::size_t>(input_count));
    guard.outputs.resize(static_cast<std::size_t>(output_count));
    for (auto& input : guard.inputs) std::memset(&input, 0, sizeof(input));
    for (auto& output : guard.outputs) std::memset(&output, 0, sizeof(output));
    for (int i = 0; i < input_count; ++i)
      if (hbDNNGetInputTensorProperties(
              &guard.inputs[static_cast<std::size_t>(i)].properties, model, i) != 0)
        throw std::runtime_error("S input metadata query failed");

    const auto& y_shape = guard.inputs[0].properties.validShape;
    const auto& uv_shape = guard.inputs[1].properties.validShape;
    if (y_shape.numDimensions != 4 || uv_shape.numDimensions != 4 ||
        y_shape.dimensionSize[0] != 1 || y_shape.dimensionSize[1] != kInputSize ||
        y_shape.dimensionSize[2] != kInputSize || y_shape.dimensionSize[3] != 1 ||
        uv_shape.dimensionSize[0] != 1 || uv_shape.dimensionSize[1] != kInputSize / 2 ||
        uv_shape.dimensionSize[2] != kInputSize / 2 ||
        uv_shape.dimensionSize[3] != 2)
      throw std::runtime_error("S YOLOv5 inputs must be Y[1,672,672,1] and UV[1,336,336,2]");

    for (int i = 0; i < output_count; ++i) {
      if (hbDNNGetOutputTensorProperties(
              &guard.outputs[static_cast<std::size_t>(i)].properties, model, i) != 0)
        throw std::runtime_error("S output metadata query failed");
      const TensorMeta meta =
          project(guard.outputs[static_cast<std::size_t>(i)].properties);
      if (meta.num_dimensions != 4 || meta.valid[0] != 1)
        throw std::runtime_error("S output must be rank-4 NHWC with batch 1");
      heads.push_back({static_cast<int>(meta.valid[1]), static_cast<int>(meta.valid[2]),
                       static_cast<int>(meta.valid[3])});
      output_meta.push_back(meta);
    }
    if (!validate_head_shapes(heads, kInputSize, kClasses))
      throw std::runtime_error("S output heads must be unique 84/42/21 metadata heads");

    run_options = {{"score_thres", std::to_string(config.score_threshold)},
                   {"nms_thres", std::to_string(config.nms_threshold)},
                   {"priority", std::to_string(config.priority)},
                   {"bpu_core", std::to_string(config.bpu_core)},
                   {"bpu_core_backend", std::to_string(backend)},
                   {"preprocess", "letterbox"},
                   {"nms_top_k_per_class", "-1"},
                   {"score_boundary", "greater-or-equal"}};
    notes = {"S UCP adapter; split NV12 672x672 Y/UV inputs; per-channel S32 dequantization.",
             "NMS preserves source S nms_bboxes semantics: keep score >= threshold, no "
             "per-class top_k.",
             "Scheduling honours the caller's priority/BPU core (source S forces priority 0); "
             "the core index is mapped to the SDK's backend bitmask.",
             "Input files hold the deterministic per-row payload the writer produced; "
             "inter-row padding bytes are uninitialized and are not dumped."};
  }

  int input_size() const { return kInputSize; }

  // Pure pixel -> payload conversion. No SDK call and no instance mutation:
  // the returned Prepared owns the split NV12 planes of exactly this call.
  Prepared preprocess(const Input& input) {
    if (input.source_rows <= 0 || input.source_cols <= 0 ||
        input.bgr.size() != static_cast<std::size_t>(input.source_rows) *
                                static_cast<std::size_t>(input.source_cols) * 3)
      throw std::invalid_argument("S input pixels do not match the declared source geometry");
    const cv::Mat source(input.source_rows, input.source_cols, CV_8UC3,
                         const_cast<unsigned char*>(input.bgr.data()));
    if (source.empty()) throw std::invalid_argument("S image is empty");
    cv::Mat boxed(kInputSize, kInputSize, CV_8UC3);
    letterbox_resize(source, boxed, 127);
    Prepared prepared;
    prepared.source_cols = input.source_cols;
    prepared.source_rows = input.source_rows;
    bgr_to_nv12_planes(boxed, kInputSize, kInputSize, &prepared.y_plane, &prepared.uv_plane);
    return prepared;
  }

  // Uploads exactly the prepared argument row by row into the strided SDK
  // planes, runs the forward pass and copies the dequantized heads plus the
  // stage evidence into the returned value.
  RawResult infer(const Prepared& prepared) {
    // The argument is the contract: exactly the two split NV12 planes of the
    // compiled input size with positive source geometry. A hand-built or
    // truncated Prepared is rejected before any SDK allocation or the row
    // upload could index out of bounds.
    if (prepared.source_cols <= 0 || prepared.source_rows <= 0 ||
        prepared.y_plane.size() != static_cast<std::size_t>(kInputSize) * kInputSize ||
        prepared.uv_plane.size() != static_cast<std::size_t>(kInputSize) * kInputSize / 2)
      throw std::invalid_argument(
          "S prepared input does not match the split NV12 contract");
    auto& inputs = guard.inputs;
    for (auto& plane : inputs) {
      if (plane.sysMem.virAddr != nullptr) {
        hbUCPFree(&plane.sysMem);
        plane.sysMem.virAddr = nullptr;
      }
    }
    if (prepare_input_tensor(inputs) != 0)
      throw std::runtime_error("S tensor allocation failed");

    // Allocations exist now, so the stride/capacity assumptions of the NV12
    // upload below are checked against what the runtime actually reported.
    TensorMeta y_gate = project(inputs[0].properties);
    y_gate.storage_bytes = y_gate.stride[0] * y_gate.valid[0];
    require_gate(check_s_nv12_plane(y_gate, kInputSize, kInputSize, 1));
    TensorMeta uv_gate = project(inputs[1].properties);
    uv_gate.storage_bytes = uv_gate.stride[0] * uv_gate.valid[0];
    require_gate(check_s_nv12_plane(uv_gate, kInputSize / 2, kInputSize / 2, 2));

    // The upload the fixed source performed inside bgr_to_nv12_tensor: each
    // row of valid bytes at the stride[1] pitch, the padding between rows
    // untouched, then a CLEAN flush of both planes.
    unsigned char* y_dst = static_cast<unsigned char*>(inputs[0].sysMem.virAddr);
    for (int row = 0; row < kInputSize; ++row)
      std::memcpy(y_dst + row * inputs[0].properties.stride[1],
                  &prepared.y_plane[static_cast<std::size_t>(row) * kInputSize], kInputSize);
    unsigned char* uv_dst = static_cast<unsigned char*>(inputs[1].sysMem.virAddr);
    const int uv_row_bytes = kInputSize;  // uv_width * 2 with uv_width = kInputSize/2
    for (int row = 0; row < kInputSize / 2; ++row)
      std::memcpy(uv_dst + row * inputs[1].properties.stride[1],
                  &prepared.uv_plane[static_cast<std::size_t>(row) * uv_row_bytes],
                  uv_row_bytes);
    hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    RawResult raw;
    raw.source_cols = prepared.source_cols;
    raw.source_rows = prepared.source_rows;
    // Input evidence is recorded from the allocated tensors, after
    // prepare_input_tensor has fixed any dynamic stride, and from the payload
    // actually submitted with this inference (the compact Prepared planes are
    // exactly the valid row bytes; inter-row padding is uninitialized memory
    // and is deliberately not dumped).
    const TensorMeta y_meta = project(inputs[0].properties);
    const TensorMeta uv_meta = project(inputs[1].properties);
    raw.inputs.push_back(dump_tensor_info("y", y_meta));
    raw.inputs.push_back(dump_tensor_info("uv", uv_meta));
    raw.input_tensors.push_back({"input0-y", dtype_name(y_meta.dtype), shape_of(y_meta),
                                  prepared.y_plane});
    raw.input_tensors.push_back({"input1-uv", dtype_name(uv_meta.dtype), shape_of(uv_meta),
                                  prepared.uv_plane});

    auto& outputs = guard.outputs;
    for (auto& output : outputs) {
      if (output.sysMem.virAddr != nullptr) {
        hbUCPFree(&output.sysMem);
        output.sysMem.virAddr = nullptr;
      }
    }
    if (prepare_output_tensor(outputs) != 0)
      throw std::runtime_error("S tensor allocation failed");
    // The dequantization gate proves the addressing contract of the buffers
    // that now exist.
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      long long count = 0;
      const Gate gate = check_s32_dequant(project(outputs[i].properties), &count);
      if (!gate) throw std::runtime_error("S output " + std::to_string(i) + ": " + gate.reason);
      // The dequantized element count must match what the helper will produce
      // for the same valid shape, padding notwithstanding.
      const auto& valid = outputs[i].properties.validShape.dimensionSize;
      const long long expected = static_cast<long long>(valid[1]) * valid[2] * valid[3];
      if (count != expected)
        throw std::runtime_error("S output " + std::to_string(i) +
                                 " has an inconsistent extent");
    }

    hbUCPTaskHandle_t task = nullptr;
    if (hbDNNInferV2(&task, outputs.data(), guard.inputs.data(), model) != 0)
      throw std::runtime_error("hbDNNInferV2 failed");
    guard.task = task;
    hbUCPSchedParam schedule{};
    HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
    // Declared difference from the fixed S source, which hard-codes priority 0
    // and HB_UCP_BPU_CORE_ANY: the unified CLI exposes scheduling, so silently
    // dropping the caller's value would make the documented parameters lie. The
    // backend is a bitmask (CORE_0..3 = 1ULL<<0..3, ANY = 1ULL<<7), so the CLI's
    // core index is converted explicitly by bpu_core_to_backend instead of
    // being assigned raw (0 would select no backend, 1 would select core 0).
    schedule.backend = backend;
    schedule.priority = config.priority;
    if (hbUCPSubmitTask(task, &schedule) != 0 || hbUCPWaitTaskDone(task, 0) != 0)
      throw std::runtime_error("S UCP task failed");
    for (auto& output : outputs) hbUCPMemFlush(&output.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);

    raw.shapes = heads;
    std::vector<std::vector<float>> dequantized;
    dequantized.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto& meta = output_meta[i];
      const auto& props = outputs[i].properties;
      raw.outputs.push_back(dump_tensor_info("output" + std::to_string(i), meta,
                                             props.scale.scaleData,
                                             props.scale.zeroPointData));
      // The raw file keeps the full allocated extent (alignedByteSize), so a
      // padded layout is dumped exactly as the runtime stored it; the manifest
      // records the strides needed to interpret it.
      const std::size_t raw_extent = static_cast<std::size_t>(meta.aligned_byte_size);
      std::vector<unsigned char> raw_bytes(raw_extent);
      if (raw_extent > 0) std::memcpy(raw_bytes.data(), outputs[i].sysMem.virAddr, raw_extent);
      raw.raw_tensors.push_back({"output" + std::to_string(i), dtype_name(meta.dtype),
                                 shape_of(meta), std::move(raw_bytes)});
      // Private broadcasting dequantizer (the shared c_utils helper indexes
      // scale_data[c] and would read out of bounds for a scalar descriptor);
      // the gate above proved the addressing contract it relies on.
      dequantized.push_back(dequant_s32_nhwc(
          static_cast<const unsigned char*>(outputs[i].sysMem.virAddr), meta,
          props.scale.scaleData, props.scale.scaleLen, props.scale.zeroPointData,
          props.scale.zeroPointLen));
      const auto& values = dequantized.back();
      std::vector<unsigned char> float_bytes(values.size() * sizeof(float));
      if (!values.empty()) std::memcpy(float_bytes.data(), values.data(), float_bytes.size());
      raw.transformed_tensors.push_back({"output" + std::to_string(i), "float32",
                                         shape_of(meta), std::move(float_bytes)});
    }
    raw.heads = std::move(dequantized);
    return raw;
  }

  Result postprocess(const RawResult& raw) {
    DecodePolicy policy;
    policy.score_threshold = config.score_threshold;
    policy.nms_threshold = config.nms_threshold;
    policy.top_k_per_class = -1;
    policy.strict_score_boundary = false;
    Result result;
    result.detections =
        decode_heads(raw.heads, raw.shapes, kInputSize, kClasses, policy, kAnchors);
    return result;
  }

  // Assembles the per-run evidence of exactly one call, mirroring the X5
  // backend: stage records from the raw value, config-derived identity, and
  // the detections mapped with this call's geometry.
  RunEvidence collect_evidence(const RawResult& raw, const Result& result) const {
    RunEvidence evidence;
    evidence.target = config.target;
    evidence.build_target = YOLOV5_TARGET_NAME;
    evidence.model_path = config.model_path;
    evidence.options = run_options;
    evidence.notes = notes;
    evidence.inputs = raw.inputs;
    evidence.outputs = raw.outputs;
    evidence.input_tensors = raw.input_tensors;
    evidence.raw_tensors = raw.raw_tensors;
    evidence.transformed_tensors = raw.transformed_tensors;
    evidence.detections = result.detections;
    evidence.detections_original =
        map_to_original(result.detections, raw.source_cols, raw.source_rows, kInputSize);
    return evidence;
  }

  Config config;
  SResourceGuard guard;
  hbDNNHandle_t model = nullptr;
  unsigned long long backend = 0;
  std::vector<HeadShape> heads;
  std::vector<TensorMeta> output_meta;
  std::vector<std::pair<std::string, std::string>> run_options;
  std::vector<std::string> notes;
};

// ====================================================== host (no target) ====

#else

// A build without a board target still compiles the SDK-free core above so it
// stays host-testable, but there is no runtime behind the stages: the
// build-identity gate rejects construction with a precise reason.
struct Yolov5::Impl {
  explicit Impl(const Config& config) {
    require_gate(check_target_matches_build(config.target, YOLOV5_TARGET_NAME));
    throw std::runtime_error("YOLOv5 detect requires a board target build "
                             "(YOLOV5_TARGET x5, s100, s100p or s600)");
  }
  int input_size() const { return 0; }
  Prepared preprocess(const Input&) {
    throw std::runtime_error("YOLOv5 preprocess requires a board target build");
  }
  RawResult infer(const Prepared&) {
    throw std::runtime_error("YOLOv5 infer requires a board target build");
  }
  Result postprocess(const RawResult&) {
    throw std::runtime_error("YOLOv5 postprocess requires a board target build");
  }
  RunEvidence collect_evidence(const RawResult&, const Result&) const {
    throw std::runtime_error("YOLOv5 evidence requires a board target build");
  }
};

#endif

// ------------------------------------------------------------ public API ----

Yolov5::Yolov5(const Config& config) : impl_(std::make_unique<Impl>(config)) {}

Yolov5::~Yolov5() = default;

int Yolov5::input_size() const { return impl_->input_size(); }

Yolov5::Prepared Yolov5::preprocess(const Input& input) {
  return impl_->preprocess(input);
}

Yolov5::RawResult Yolov5::infer(const Prepared& prepared) {
  return impl_->infer(prepared);
}

Yolov5::Result Yolov5::postprocess(const RawResult& raw) {
  return impl_->postprocess(raw);
}

// The visible chain. Every value is owned per call, so the returned Prediction
// stays valid no matter what a later predict does.
Yolov5::Prediction Yolov5::predict(const Input& input) {
  const Prepared prepared = preprocess(input);
  const RawResult raw = infer(prepared);
  Prediction run;
  run.result = postprocess(raw);
  run.evidence = impl_->collect_evidence(raw, run.result);
  return run;
}

}  // namespace yolov5
