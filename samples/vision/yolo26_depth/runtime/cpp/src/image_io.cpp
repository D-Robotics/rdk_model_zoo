// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "image_io.hpp"
#include "tensor_contract.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
namespace yolo26_depth {
std::vector<std::uint8_t> pack_nv12(const cv::Mat &image) {
  if (image.empty() || image.type() != CV_8UC3 || image.rows % 2 ||
      image.cols % 2)
    throw std::invalid_argument(
        "NV12 conversion needs even BGR uint8 dimensions");
  cv::Mat i420;
  cv::cvtColor(image, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous())
    i420 = i420.clone();
  const std::size_t area = static_cast<std::size_t>(image.rows) * image.cols;
  std::vector<std::uint8_t> packed(area * 3 / 2);
  std::memcpy(packed.data(), i420.data, area);
  const auto *u = i420.data + area;
  const auto *v = u + area / 4;
  for (std::size_t i = 0; i < area / 4; ++i) {
    packed[area + 2 * i] = u[i];
    packed[area + 2 * i + 1] = v[i];
  }
  return packed;
}
cv::Mat colorize_depth(const cv::Mat &depth) {
  if (depth.empty() || depth.type() != CV_32FC1 || !cv::checkRange(depth))
    throw std::invalid_argument("Rendering needs finite float32 HxW depth");
  std::vector<float> values;
  values.reserve(depth.total());
  for (int h = 0; h < depth.rows; ++h)
    values.insert(values.end(), depth.ptr<float>(h),
                  depth.ptr<float>(h) + depth.cols);
  const double low = percentile(values, .02), high = percentile(values, .98);
  const double range = std::max(high - low, 1e-6);
  cv::Mat gray(depth.rows, depth.cols, CV_8UC1), color;
  for (int h = 0; h < depth.rows; ++h)
    for (int w = 0; w < depth.cols; ++w) {
      const double normalized =
          std::clamp((depth.ptr<float>(h)[w] - low) / range, 0.0, 1.0);
      gray.ptr<std::uint8_t>(h)[w] =
          255 - static_cast<std::uint8_t>(normalized * 255);
    }
  cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
  return color;
}
} // namespace yolo26_depth
