// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/nv12_geometry.h"
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>
namespace yoloe {
struct Nv12Input {
  std::vector<uint8_t> y, uv;
};
inline Nv12Input to_nv12(const cv::Mat &pixels) {
  if (pixels.type() != CV_8UC3 || pixels.rows != 640 || pixels.cols != 640)
    throw std::invalid_argument("NV12 requires prepared uint8 BGR 640x640");
  cv::Mat i420;
  cv::cvtColor(pixels, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous() || i420.total() != 640 * 640 * 3 / 2)
    throw std::runtime_error("Unexpected OpenCV I420 storage");
  Nv12Input input;
  input.y.resize(640 * 640);
  input.uv.resize(320 * 640);
  const auto *y = i420.ptr<uint8_t>();
  yolo::i420_to_split_nv12(y, y + 640 * 640, y + 640 * 640 + 320 * 320, 640,
                           640, input.y.data(), 640, input.uv.data(), 640);
  return input;
}
} // namespace yoloe
