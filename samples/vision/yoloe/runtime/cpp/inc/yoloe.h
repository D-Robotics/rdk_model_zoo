// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "config.h"
#include "pipeline_io.h"
namespace yoloe {
// One task owns one backend. No file I/O, rendering, implicit download or
// hardware fallback. The injected backend must implement the declared protocol.
class YOLOE {
public:
  YOLOE(Config config, std::unique_ptr<Runner> runner);
  YOLOE(const YOLOE &) = delete;
  YOLOE &operator=(const YOLOE &) = delete;
  Prepared pre_process(const cv::Mat &image) const;
  RawBatch infer(const Prepared &input);
  Result post_process(const RawBatch &raw) const;
  Result predict(const cv::Mat &image);

private:
  Config config_;
  std::unique_ptr<Runner> runner_;
  std::shared_ptr<const StageIdentity> identity_ =
      std::make_shared<StageIdentity>();
};
} // namespace yoloe
