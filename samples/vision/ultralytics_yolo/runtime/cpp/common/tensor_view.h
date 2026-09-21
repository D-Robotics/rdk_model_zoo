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

// Pure (host-testable) helpers describing BPU output tensors and the shared
// YOLO head contracts. Nothing in this header may include board or OpenCV
// headers so the decode logic can be unit-tested on the host.

#ifndef RUNTIME_CPP_COMMON_TENSOR_VIEW_H_
#define RUNTIME_CPP_COMMON_TENSOR_VIEW_H_

#include <cstddef>
#include <string>
#include <vector>

namespace yolo {

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

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_TENSOR_VIEW_H_
