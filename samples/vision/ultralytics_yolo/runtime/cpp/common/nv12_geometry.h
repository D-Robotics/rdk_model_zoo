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

// Pure (host-testable) NV12 plane packing helpers. Both supported BPU input
// protocols consume the same colour conversion, only the memory layout
// differs:
//   * packed NV12 (X5 .bin): one contiguous buffer, Y plane followed by an
//     interleaved UV plane, no row padding beyond width alignment.
//   * split NV12 (S100/S100P/S600 .hbm): two UINT8 tensors, Y [1,H,W,1] and
//     UV [1,H/2,W/2,2], each written row-by-row honouring its byte stride.
// The split layout mirrors platforms/s/utils/c_utils/src/preprocess.cpp.

#ifndef RUNTIME_CPP_COMMON_NV12_GEOMETRY_H_
#define RUNTIME_CPP_COMMON_NV12_GEOMETRY_H_

#include <cstdint>
#include <cstring>

namespace yolo {

// Packs separate I420 planes (as produced by cv::cvtColor(...,
// COLOR_BGR2YUV_I420)) into one contiguous NV12 buffer of h*3/2 rows.
void i420_to_packed_nv12(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                         int h, int w, uint8_t* dst);

// Writes the same planes into two stride-padded tensors:
//   y_dst:  h rows of w bytes, advancing `y_stride` bytes per row.
//   uv_dst: h/2 rows of w bytes (u,v interleaved), advancing `uv_stride`
//           bytes per row.
void i420_to_split_nv12(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        int h, int w, uint8_t* y_dst, int y_stride,
                        uint8_t* uv_dst, int uv_stride);

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_NV12_GEOMETRY_H_
