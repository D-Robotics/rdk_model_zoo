// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "model_identity.h"
#include "runner.h"
#include <memory>
namespace yoloe {
class SdkRunner final : public Runner {
public:
  SdkRunner(SdkModel model, SdkPreflight preflight);
  ~SdkRunner() override;
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  Protocol protocol() const override;
  Heads infer(const Nv12Input &input) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace yoloe
