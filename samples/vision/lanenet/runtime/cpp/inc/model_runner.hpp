// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <functional>
#include <memory>
#include <string>
namespace lanenet {
class ModelRunner {
public:
  using ExecutionGate = std::function<void()>; // host seam, never a CLI option
  explicit ModelRunner(const std::string &path, ExecutionGate gate = {});
  ~ModelRunner();
  ModelRunner(const ModelRunner &) = delete;
  ModelRunner &operator=(const ModelRunner &) = delete;
  std::vector<RawTensor> run(const std::vector<float> &prepared);
  const std::string &model_name() const;
  const TensorSpec &input_spec() const;
  const std::vector<TensorSpec> &output_specs() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace lanenet
