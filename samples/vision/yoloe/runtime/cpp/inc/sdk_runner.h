// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "runner.h"
#include <functional>
#include <memory>
#include <string>
namespace yoloe {
struct SdkModel {
  std::string path, target, variant;
};
// Required policy boundary: verify actual board identity, exact selected asset
// or custom float SHA-256 and vocabulary/conversion provenance. Runs before
// any SDK call. This low-level adapter does not supply a publication resolver.
using SdkPreflight = std::function<void(const SdkModel &)>;
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
