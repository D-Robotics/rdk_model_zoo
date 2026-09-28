// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
namespace paraformer {
enum class Stage { Encoder, Predictor, Decoder };
struct SdkModel {
  std::string path, target;
  Stage stage;
};
using SdkPreflight = std::function<void(const SdkModel &)>;
using RawTensor = std::variant<std::vector<float>, std::vector<int32_t>>;
using RawTensors = std::map<std::string, RawTensor>;
struct TensorMetadata {
  std::string name, role, dtype;
  std::vector<int> shape;
  std::vector<int64_t> strides;
  size_t allocation_bytes = 0;
};
struct SdkMetadata {
  std::string model_name;
  std::vector<TensorMetadata> inputs, outputs;
};
// Exactly one synchronous raw model call. The mandatory preflight callback must
// verify local identity and selected artifact before any SDK operation.
class SdkRunner {
public:
  SdkRunner(SdkModel, SdkPreflight);
  ~SdkRunner();
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  const SdkMetadata &metadata() const;
  RawTensors infer(const RawTensors &inputs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace paraformer
