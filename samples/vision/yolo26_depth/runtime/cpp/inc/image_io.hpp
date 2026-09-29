// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>
namespace yolo26_depth {
std::vector<std::uint8_t> pack_nv12(const cv::Mat &image);
cv::Mat colorize_depth(const cv::Mat &depth);
} // namespace yolo26_depth
