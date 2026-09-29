// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace gemma4 {
// Borrow CV_8UC3 BGR pixels; return an owned [2520,768] RGB float patch array.
// Bicubic resize to 960x672, scale 1/255, 16x16 patches in row-major order.
// No file IO, model load, SDK calls, shared state or mutation of the input.
std::vector<float> PreprocessImage(const cv::Mat &bgr);
} // namespace gemma4
