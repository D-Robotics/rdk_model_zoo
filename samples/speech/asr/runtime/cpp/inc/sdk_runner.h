// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace asr {
struct SdkModel {
  std::string path, target;
};
using SdkPreflight = std::function<void(const SdkModel &)>;
struct SdkMetadata {
  std::string model_name;
  size_t steps = 0;
  std::vector<int64_t> input_strides, output_strides;
  size_t input_bytes = 0, output_bytes = 0;
};
// UCP only. Callback is mandatory and runs before any SDK call.
class SdkRunner {
public:
  SdkRunner(SdkModel model, SdkPreflight preflight);
  ~SdkRunner();
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  const SdkMetadata &metadata() const;
  std::vector<float> infer(const std::vector<float> &prepared);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace asr
