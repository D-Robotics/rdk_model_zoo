// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/task_outputs.h"
#include <array>
#include <stdexcept>
#include <vector>
namespace yoloe {
// Semantic order: (class, box, coefficients) at strides 8/16/32, prototype.
// Physical output order is deliberately not part of the contract. This only
// binds logical roles; use nhwc_float_plan/copy_float_output at the SDK
// boundary to require unquantized FLOAT32 and validate physical
// strides/allocation.
inline std::array<int, 10>
bind_heads(const std::vector<yolo::OutputShape> &shapes, int box_channels) {
  if (shapes.size() != 10 || (box_channels != 4 && box_channels != 64))
    throw std::invalid_argument(
        "YOLOE requires ten heads and explicit LTRB4 or DFL64 geometry.");
  std::array<int, 10> roles{};
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale;
    for (int role = 0; role < 3; ++role) {
      int channels = role == 0 ? 4585 : (role == 1 ? box_channels : 32);
      roles[scale * 3 + role] =
          yolo::find_output_by_shape(shapes, grid, grid, channels);
    }
  }
  roles[9] = yolo::find_output_by_shape(shapes, 160, 160, 32);
  for (int index : roles)
    if (index < 0)
      throw std::invalid_argument(
          "Missing, ambiguous or incompatible YOLOE output role.");
  return roles;
}
} // namespace yoloe
