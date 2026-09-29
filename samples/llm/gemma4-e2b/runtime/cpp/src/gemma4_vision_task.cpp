// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#include "gemma4_vision_task.hpp"
#include "gemma4_config.hpp"
#include <cmath>
#include <stdexcept>
namespace gemma4 {
std::vector<float> ForwardVision(const std::vector<float> &patches,
                                 const VisionRunner &runner) {
  if (!runner ||
      patches.size() != static_cast<size_t>(kVisionPatches) * kVisionPatchDim)
    throw std::invalid_argument(
        "Vision forward requires a runner and [2520,768] patches");
  for (float value : patches) {
    if (!std::isfinite(value) || value < 0.f || value > 1.f)
      throw std::invalid_argument(
          "Vision RGB patches must be finite values in [0,1]");
  }
  return runner(patches);
}
std::vector<float> PostprocessVision(const std::vector<float> &raw) {
  if (raw.size() != static_cast<size_t>(kVisionSoftTokens) * kHiddenSize)
    throw std::invalid_argument("Vision output requires [280,1536] features");
  for (float value : raw) {
    if (!std::isfinite(value))
      throw std::invalid_argument("Vision output must be finite");
  }
  return raw;
}
std::vector<float> PredictVision(const cv::Mat &bgr,
                                 const VisionRunner &runner) {
  return PostprocessVision(ForwardVision(PreprocessImage(bgr), runner));
}
} // namespace gemma4
