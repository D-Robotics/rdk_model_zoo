// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace himloco {
namespace {
void Validate(const std::vector<float> &values, std::size_t count,
              const char *name) {
  if (values.size() != count)
    throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                std::to_string(count) + " float32 values");
  if (!std::all_of(values.begin(), values.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) + " contains NaN/Inf");
}
void ValidateRaw(const RawOutputs &raw) {
  Validate(raw.actions, kOutputElements, "actions");
  if (!std::isfinite(raw.latency_ms) || raw.latency_ms < 0)
    throw std::invalid_argument("latency must be finite and non-negative");
}
}  // namespace

HimLoco::HimLoco(Runner runner) : runner_(std::move(runner)) {
  if (!runner_)
    throw std::invalid_argument("HimLoco requires a runner");
}

PreparedInput HimLoco::pre_process(const std::vector<float> &observation) const {
  Validate(observation, kInputElements, "obs_history");
  return {observation};
}

RawOutputs HimLoco::forward(const PreparedInput &input) const {
  Validate(input.values, kInputElements, "obs_history");
  RawOutputs raw = runner_(input.values);
  ValidateRaw(raw);
  return raw;
}

InferenceResult HimLoco::post_process(const RawOutputs &raw) const {
  ValidateRaw(raw);
  return {raw.actions, raw.latency_ms};
}

InferenceResult HimLoco::predict(const std::vector<float> &observation) const {
  return post_process(forward(pre_process(observation)));
}
}  // namespace himloco
