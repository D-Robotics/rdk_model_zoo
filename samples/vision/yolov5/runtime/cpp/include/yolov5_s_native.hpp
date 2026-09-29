#ifndef RDK_MODEL_ZOO_YOLOV5_S_NATIVE_HPP_
#define RDK_MODEL_ZOO_YOLOV5_S_NATIVE_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// SDK-free numeric and scheduling helpers for the S YOLOv5 native adapter,
// extracted so their exact behaviour is compilable and testable on a host
// without the UCP SDK. The shared c_utils dequantizer is left untouched (other
// samples depend on it); the adapter uses these private helpers instead, with
// the addressing contract check_s32_dequant proves.

#include "yolov5_gate.hpp"

#include <cstdint>
#include <vector>

namespace yolov5 {

// Dequantizes one S output that passed check_s32_dequant. Element (h, w, c) is
// read at byte offset (h*W + w)*stride[2] + c*stride[3], exactly like the
// fixed-source dequantizeTensorS32, but with two deliberate differences that
// the shared helper lacks:
//   * a scalar descriptor (scale_len/zero_point_len == 1) broadcasts its single
//     value to every channel instead of reading scale_data[c] out of bounds;
//   * zero_point_len == 0 means no zero point (0).
// SCALE outputs are read as native int32, NONE outputs as native float32, and
// the result is the compact NHWC float vector of height*width*channels values.
// Throws std::invalid_argument for a quantization kind the gate rejects.
std::vector<float> dequant_s32_nhwc(const unsigned char* base, const TensorMeta& meta,
                                    const float* scale_data, long long scale_len,
                                    const std::int32_t* zero_point_data,
                                    long long zero_point_len);

// Real hb_ucp.h defines the scheduler backend as a bitmask: HB_UCP_BPU_CORE_0
// through _3 are 1ULL<<0..3 and HB_UCP_BPU_CORE_ANY is 1ULL<<7. The CLI's
// --bpu-core is a core *index* (-1 = any), so the conversion has to be
// explicit: assigning the index directly would send 0 as "no backend" and 1 as
// core 0. Returns false (leaving *backend untouched) for indices outside
// -1..3, which the caller must reject instead of silently scheduling.
bool bpu_core_to_backend(long long bpu_core, unsigned long long* backend);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_S_NATIVE_HPP_
