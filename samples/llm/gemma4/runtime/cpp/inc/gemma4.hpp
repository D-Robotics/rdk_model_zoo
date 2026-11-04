// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "oellm_runtime_basic/oellm_runtime.h"

namespace gemma4 {

struct Config {
  std::string config_path;
  int max_new_tokens = 128;
};

// The SDK owns tokenization, embeddings, Gemma4 runners, KV state and sampling.
class Gemma4 {
 public:
  explicit Gemma4(const Config& config);
  nlohmann::json Generate(const std::string& prompt);
  void Reset();

 private:
  Config config_;
  oellm::OellmRuntime runtime_;
  std::vector<oellm::OellmPrompt::HistoryMessage> history_;
  int request_id_ = 0;
};

}  // namespace gemma4
