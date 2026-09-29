// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "visualization.hpp"
#include <stdexcept>
namespace lanenet {
cv::Mat embedding_image(const LaneResult &result) {
  if (result.embedding.size() != 3 * 256 * 512)
    throw std::invalid_argument("Wrong embedding geometry");
  cv::Mat image(256, 512, CV_8UC3);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++)
      for (int c = 0; c < 3; c++)
        image.at<cv::Vec3b>(h, w)[c] =
            display_component(result.embedding[c * 256 * 512 + h * 512 + w]);
  return image; // preserved channel order; not lane-instance IDs
}
cv::Mat binary_image(const LaneResult &result) {
  if (result.binary.size() != 256 * 512)
    throw std::invalid_argument("Wrong binary geometry");
  cv::Mat image(256, 512, CV_8UC1);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++) {
      const auto value = result.binary[h * 512 + w];
      if (value > 1)
        throw std::invalid_argument("Invalid binary label");
      image.at<unsigned char>(h, w) = value * 255;
    }
  return image;
}
} // namespace lanenet
