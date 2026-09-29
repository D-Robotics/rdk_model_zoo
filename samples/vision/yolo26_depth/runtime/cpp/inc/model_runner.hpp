// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace yolo26_depth {
// SDK ownership/transport only. No activation, geometry, timing or file
// outputs. Not thread-safe: concurrent tasks require independent owners.
class ModelRunner {
public:
  using ExecutionGate = std::function<void()>;
  explicit ModelRunner(const std::string &path, ExecutionGate gate = {});
  ~ModelRunner();
  ModelRunner(const ModelRunner &) = delete;
  ModelRunner &operator=(const ModelRunner &) = delete;
  std::vector<float> run(const std::vector<std::uint8_t> &nv12);
  const std::string &model_name() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace yolo26_depth
