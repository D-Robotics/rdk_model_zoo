// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#include "gemma4_image_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
namespace gemma4 {
cv::Mat LoadImage(const std::string &path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::runtime_error("failed to read image: " + path);
  return image;
}
} // namespace gemma4
