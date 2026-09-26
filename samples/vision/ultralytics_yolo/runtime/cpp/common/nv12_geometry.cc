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

#include "nv12_geometry.h"
#include <cstddef>

namespace yolo {

void i420_to_packed_nv12(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                         int h, int w, uint8_t* dst) {
  const int y_size = h * w;
  const int uv_plane_size = y_size / 4;
  std::memcpy(dst, y, static_cast<size_t>(y_size));
  uint8_t* uv = dst + y_size;
  for (int i = 0; i < uv_plane_size; ++i) {
    uv[2 * i] = u[i];
    uv[2 * i + 1] = v[i];
  }
}

void i420_to_split_nv12(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        int h, int w, uint8_t* y_dst, int y_stride,
                        uint8_t* uv_dst, int uv_stride) {
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

}  // namespace yolo
