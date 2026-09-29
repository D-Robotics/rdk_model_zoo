// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
namespace yoloe {
// Owned model-canvas candidate. Geometry/mask restoration is a separate stage.
struct RawDetection {
  std::array<float, 4> box{};
  float score = 0;
  int label = 0;
  std::array<float, 32> coefficients{};
};
} // namespace yoloe
