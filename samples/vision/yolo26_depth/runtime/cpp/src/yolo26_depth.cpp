// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "yolo26_depth.hpp"
#include "image_io.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
namespace yolo26_depth {
Yolo26DepthTask::Yolo26DepthTask(RawRunner runner)
    : runner_(std::move(runner)) {
  if (!runner_)
    throw std::invalid_argument("A raw inference runner is required");
}
PreparedInput Yolo26DepthTask::pre_process(const cv::Mat &image) const {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected a nonempty BGR uint8 HWC image");
  const auto context = letterbox_geometry(image.rows, image.cols);
  const int width = kInputSize - context.left - context.right,
            height = kInputSize - context.top - context.bottom;
  cv::Mat resized, padded;
  if (image.rows == height && image.cols == width)
    resized = image;
  else
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  cv::copyMakeBorder(resized, padded, context.top, context.bottom, context.left,
                     context.right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  return {pack_nv12(padded), context};
}
std::vector<float>
Yolo26DepthTask::forward(const std::vector<std::uint8_t> &nv12) const {
  return runner_(nv12);
}
DepthResult Yolo26DepthTask::post_process(const std::vector<float> &raw,
                                          const ImageContext &context) const {
  validate_context(context);
  if (raw.size() != kOutputSize * kOutputSize ||
      !std::all_of(raw.begin(), raw.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument(
        "Expected finite raw float32 192-square calibrated log-depth");
  cv::Mat log(kOutputSize, kOutputSize, CV_32FC1);
  std::copy(raw.begin(), raw.end(), log.ptr<float>());
  cv::Mat depth, square, restored;
  cv::exp(log, depth);
  if (!cv::checkRange(depth))
    throw std::invalid_argument("Depth exponential overflow");
  cv::resize(depth, square, cv::Size(kInputSize, kInputSize), 0, 0,
             cv::INTER_LINEAR);
  const cv::Rect area(context.left, context.top,
                      kInputSize - context.left - context.right,
                      kInputSize - context.top - context.bottom);
  cv::resize(square(area), restored,
             cv::Size(context.original_width, context.original_height), 0, 0,
             cv::INTER_LINEAR);
  return {log, restored, context};
}
DepthResult Yolo26DepthTask::predict(const cv::Mat &image) const {
  const auto prepared = pre_process(image);
  return post_process(forward(prepared.nv12), prepared.context);
}
} // namespace yolo26_depth
