// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#pragma once
#include "gemma4_vision_preprocess.hpp"
#include <functional>
#include <vector>
namespace gemma4 {
using VisionRunner =
    std::function<std::vector<float>(const std::vector<float> &)>;
// Validate prepared RGB patches and call an explicitly provided raw runner
// once.
std::vector<float> ForwardVision(const std::vector<float> &patches,
                                 const VisionRunner &runner);
// Validate and own the [280,1536] features; no normalization or rescaling.
std::vector<float> PostprocessVision(const std::vector<float> &raw);
// Three-stage composition. Image IO and runner construction are caller-owned.
std::vector<float> PredictVision(const cv::Mat &bgr,
                                 const VisionRunner &runner);
} // namespace gemma4
