// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <cstdint>
#include <functional>
#include <opencv2/core.hpp>
#include <vector>
namespace yolo26_depth {
struct PreparedInput {
  std::vector<std::uint8_t> nv12;
  ImageContext context;
};
struct DepthResult {
  cv::Mat log_depth;
  cv::Mat depth_native;
  ImageContext context;
};
// Task stages only. Own per-call context; no SDK, mutable last-transform or IO.
class Yolo26DepthTask {
public:
  using RawRunner =
      std::function<std::vector<float>(const std::vector<std::uint8_t> &)>;
  explicit Yolo26DepthTask(RawRunner runner);
  PreparedInput pre_process(const cv::Mat &image) const;
  std::vector<float> forward(const std::vector<std::uint8_t> &nv12) const;
  DepthResult post_process(const std::vector<float> &raw,
                           const ImageContext &context) const;
  DepthResult predict(const cv::Mat &image) const;

private:
  RawRunner runner_;
};
} // namespace yolo26_depth
