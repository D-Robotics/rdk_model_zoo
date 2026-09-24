// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_gate.hpp"

#include <string>

namespace yolov5 {
namespace {

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

}  // namespace

Gate accept() { return Gate{true, {}}; }

Gate reject(std::string reason) { return Gate{false, std::move(reason)}; }

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
    if (meta.scale_len <= 0)
      return reject("S YOLOv5 scaled output has no scale descriptor");
    if (meta.scale_len != 1 && meta.scale_len < channels)
      return reject("S YOLOv5 output scale descriptor is shorter than its channels");
    if (meta.zero_point_len != 0 && meta.zero_point_len != 1 &&
        meta.zero_point_len < channels)
      return reject("S YOLOv5 output zero-point descriptor is shorter than its channels");
  } else if (meta.quanti_type == kQuantiNone) {
    if (meta.dtype != kDtypeF32)
      return reject("S unquantized YOLOv5 output must be native F32");
  } else {
    return reject("S YOLOv5 output has an unsupported quantization type");
  }

  // dequantizeTensorS32 addresses memory as contiguous NHWC; a strided layout
  // would be read with wrong offsets, so reject anything that is not contiguous.
  if (meta.stride[3] != element_bytes)
    return reject("S YOLOv5 output channel stride is not the element size");
  if (meta.stride[2] != channels * meta.stride[3])
    return reject("S YOLOv5 output is not contiguous across channels");
  if (meta.stride[1] != width * meta.stride[2])
    return reject("S YOLOv5 output is not contiguous across rows");
  const long long count = height * width * channels;
  const long long required = count * element_bytes;
  if (meta.aligned_byte_size < required)
    return reject("S YOLOv5 output alignedByteSize cannot hold its elements");
  if (meta.storage_bytes < required)
    return reject("S YOLOv5 output allocation cannot hold its elements");
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

}  // namespace yolov5
