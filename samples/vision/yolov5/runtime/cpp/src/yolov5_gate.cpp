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

// Checked arithmetic for the extent proofs: a hostile or corrupted stride must
// be rejected, not wrap around into an accepting comparison.
bool checked_mul(long long a, long long b, long long* out) {
  return !__builtin_mul_overflow(a, b, out);
}

bool checked_add(long long a, long long b, long long* out) {
  return !__builtin_add_overflow(a, b, out);
}

}  // namespace

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
    // A scalar descriptor (length 1) is accepted because the adapter's private
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

  // The addressing both the fixed-source helper and the adapter's private
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

}  // namespace yolov5
