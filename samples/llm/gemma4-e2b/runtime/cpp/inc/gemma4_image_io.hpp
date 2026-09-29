// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#pragma once
#include <opencv2/core.hpp>
#include <string>
namespace gemma4 {
// Application IO: decode an image into owned BGR pixels, or throw
// runtime_error.
cv::Mat LoadImage(const std::string &path);
} // namespace gemma4
