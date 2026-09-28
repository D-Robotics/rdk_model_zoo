// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "policy.hpp"
#include <memory>
#include <string>

namespace himloco {
struct NativeConfig {
  std::string model_path; // Explicit local published BIN; never downloaded.
  int priority = -1;      // SDK default or [0,255].
};
struct TensorMetadata {
  std::string name;
  std::vector<int> valid_shape, aligned_shape;
  int tensor_layout = 0, tensor_type = 0, quanti_type = 0,
      aligned_byte_size = 0;
};
/// Require actual X5 identity and the exact published model SHA before SDK
/// load.
void verify_native_model(const std::string &model_path);

/// One packed model and reusable buffers, owned by RAII. Not thread-safe.
/// Constructor validates board/model and initializes the SDK; throws on
/// failure. No copying or moving: a task callback can safely reference a stable
/// runner.
class SdkRunner {
public:
  explicit SdkRunner(const NativeConfig &config);
  ~SdkRunner();
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  RawOutputs run(const std::vector<float> &input);
  const TensorMetadata &input_metadata() const;
  const TensorMetadata &output_metadata() const;
  const std::string &model_name() const;
  const std::string &runtime_version() const;
  int priority() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace himloco
