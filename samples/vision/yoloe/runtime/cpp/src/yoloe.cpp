// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "yoloe.h"
#include "postprocess.h"
namespace yoloe {
YOLOE::YOLOE(Config config, std::unique_ptr<Runner> runner)
    : config_(config), runner_(std::move(runner)) {
  validate_config(config_);
  if (!runner_ || runner_->protocol() != config_.protocol)
    throw std::invalid_argument(
        "Provide a backend matching the YOLOE protocol");
}
Prepared YOLOE::pre_process(const cv::Mat &image) const {
  auto prepared = prepare_bgr(image, config_.protocol, config_.resize_type);
  return Prepared(to_nv12(prepared.pixels), prepared.geometry, identity_);
}
RawBatch YOLOE::infer(const Prepared &input) {
  if (input.owner_ != identity_)
    throw std::invalid_argument("Prepared input belongs to another task");
  return RawBatch(runner_->infer(input.input_), input.geometry_, identity_);
}
Result YOLOE::post_process(const RawBatch &raw) const {
  if (raw.owner_ != identity_)
    throw std::invalid_argument("Raw outputs belong to another task");
  return decode_result(raw.outputs_, raw.geometry_, config_);
}
Result YOLOE::predict(const cv::Mat &image) {
  return post_process(infer(pre_process(image)));
}
} // namespace yoloe
