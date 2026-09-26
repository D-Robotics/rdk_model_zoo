// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <functional>
#include <opencv2/core.hpp>
namespace lanenet {
class LaneNetTask {
public:
  using RawRunner =
      std::function<std::vector<RawTensor>(const std::vector<float> &)>;
  explicit LaneNetTask(RawRunner runner);
  std::vector<float> pre_process(const cv::Mat &image) const;
  std::vector<RawTensor> forward(const std::vector<float> &prepared) const;
  LaneResult post_process(const std::vector<RawTensor> &raw) const;
  LaneResult predict(const cv::Mat &image) const;

private:
  RawRunner runner_;
};
} // namespace lanenet
