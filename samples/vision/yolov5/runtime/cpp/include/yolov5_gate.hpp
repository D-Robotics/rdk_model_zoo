#ifndef RDK_MODEL_ZOO_YOLOV5_GATE_HPP_
#define RDK_MODEL_ZOO_YOLOV5_GATE_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// SDK-free tensor metadata gates shared by the X5 and S YOLOv5 native
// adapters. The adapters project hbDNNTensorProperties into TensorMeta and must
// pass the projection through these gates before any pointer cast, memcpy or
// dequantization read. Keeping the checks here (instead of inline in the
// adapters) makes the accept/reject behaviour compilable and testable on a host
// without the target SDK headers.

#include <string>

namespace yolov5 {

// Adapters translate their SDK enum values into these codes; unknown values
// translate to -1 and are always rejected. No SDK numeric value is assumed.
enum DtypeCode : int {
  kDtypeUnknown = -1,
  kDtypeF32 = 0,
  kDtypeS32 = 1,
  kDtypeS8 = 2,
  kDtypeU8 = 3,
  kDtypeS16 = 4,
};

enum QuantiCode : int {
  kQuantiUnknown = -1,
  kQuantiNone = 0,
  kQuantiScale = 1,
  kQuantiShift = 2,
};

enum ImageCode : int {
  kImageUnknown = -1,
  kImageNv12 = 0,
};

// Plain projection of the runtime tensor properties the gates inspect.
struct TensorMeta {
  int dtype = kDtypeUnknown;
  int quanti_type = kQuantiUnknown;
  int num_dimensions = 0;
  long long valid[4] = {0, 0, 0, 0};
  long long aligned[4] = {0, 0, 0, 0};
  long long aligned_byte_size = 0;
  long long storage_bytes = 0;
  long long stride[4] = {0, 0, 0, 0};
  long long scale_len = 0;
  long long zero_point_len = 0;
};

struct Gate {
  bool ok = true;
  std::string reason;

  explicit operator bool() const { return ok; }
};

Gate accept();
Gate reject(std::string reason);

// X5: exactly one packed NV12 input of expected_size x expected_size. The
// adapter copies a compact NV12 payload into the buffer, so a padded/aligned
// layout and an undersized allocation are both rejected instead of being
// silently reinterpreted.
Gate check_x5_nv12_input(int image_type, const TensorMeta& meta,
                         long long expected_size);

// X5: one native F32, NONE-quantized NHWC detection head. Requires the aligned
// layout to equal the valid layout (the reader does a flat float read) and the
// allocation to cover height * width * channels floats. On success writes the
// head stride level (8, 16 or 32).
Gate check_x5_head(const TensorMeta& meta, long long input_size, long long classes,
                   int* stride_level);

// S: one split NV12 plane (Y or UV). Validates the logical shape and requires
// byte strides and allocation that cover every row the preprocessing writer
// touches.
Gate check_s_nv12_plane(const TensorMeta& meta, long long rows, long long cols,
                        long long channels);

// S: one output tensor that dequantizeTensorS32 may read. Validates the native
// dtype, the quantization descriptor length and the byte strides against a
// contiguous NHWC layout, and requires the allocation to cover the whole
// element count. On success writes the element count.
Gate check_s32_dequant(const TensorMeta& meta, long long* element_count);

// A native binary is configured for exactly one target (S alignment differs
// between S600 and the rest), so it must refuse to run as another target.
Gate check_target_matches_build(const std::string& requested,
                                const std::string& build_target);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_GATE_HPP_
