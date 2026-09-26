// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "lanenet.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
namespace lanenet {
LaneNetTask::LaneNetTask(RawRunner runner) : runner_(std::move(runner)) {
  if (!runner_)
    throw std::invalid_argument("LaneNet requires a raw runner");
}
std::vector<float> LaneNetTask::pre_process(const cv::Mat &image) const {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR uint8 image");
  cv::Mat rgb, resized;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(512, 256), 0, 0, cv::INTER_AREA);
  const float mean[] = {.485f, .456f, .406f}, stddev[] = {.229f, .224f, .225f};
  std::vector<float> result(3 * 256 * 512);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++)
      for (int c = 0; c < 3; c++)
        result[c * 256 * 512 + h * 512 + w] =
            (resized.at<cv::Vec3b>(h, w)[c] / 255.0f - mean[c]) / stddev[c];
  return result;
}
std::vector<RawTensor>
LaneNetTask::forward(const std::vector<float> &prepared) const {
  return runner_(prepared);
}
LaneResult LaneNetTask::post_process(const std::vector<RawTensor> &raw) const {
  return decode_outputs(raw);
}
LaneResult LaneNetTask::predict(const cv::Mat &image) const {
  return post_process(forward(pre_process(image)));
}
} // namespace lanenet
