// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#pragma once
#include "gemma4_config.hpp"
#include "hobot/dnn/hb_dnn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
namespace gemma4 {
inline void LogVisionValues(const char *name,
                            const std::vector<float> &values) {
  if (!RuntimeDebugEnabled() || values.empty())
    return;
  double sum = 0, squares = 0;
  float low = values[0], high = values[0];
  for (float value : values) {
    sum += value;
    squares += static_cast<double>(value) * value;
    low = std::min(low, value);
    high = std::max(high, value);
  }
  const double mean = sum / values.size();
  const double variance = std::max(0.0, squares / values.size() - mean * mean);
  std::cerr << "[DEBUG] " << name << ": size=" << values.size()
            << " min=" << low << " max=" << high << " mean=" << mean
            << " std=" << std::sqrt(variance) << std::endl;
}
inline void LogVisionTensor(const char *name, const hbDNNTensorProperties &p) {
  if (!RuntimeDebugEnabled())
    return;
  std::cerr << "[DEBUG] " << name << ": type=" << p.tensorType
            << " ndim=" << p.validShape.numDimensions << " shape=[";
  for (int axis = 0; axis < p.validShape.numDimensions; ++axis) {
    if (axis)
      std::cerr << ",";
    std::cerr << p.validShape.dimensionSize[axis];
  }
  std::cerr << "] aligned_bytes=" << p.alignedByteSize << std::endl;
}
} // namespace gemma4
