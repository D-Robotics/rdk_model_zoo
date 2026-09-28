// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <limits>
#include <vector>

namespace himloco {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "HIMLoco requires IEEE-754 float32");
constexpr int kInputElements = 270;
constexpr int kOutputElements = 12;

/// Owned, training-boundary observations: current 45 values then five past frames.
struct PreparedInput {
  std::vector<float> values;
};
/// Owned raw F32 actions with the same call's synchronous SDK duration.
struct RawOutputs {
  std::vector<float> actions;
  double latency_ms = 0;
};
/// Unscaled actions; application/controller owns any joint-target conversion.
struct InferenceResult {
  std::vector<float> actions;
  double latency_ms = 0;
};

/// Adapter returns 12 owned float values and its own inference timing.
using Runner = std::function<RawOutputs(const std::vector<float> &)>;

/// Pure policy stages. No SDK loading, file IO, history updates or robot control.
/// Stage values own their memory and can be interleaved. Thread safety depends
/// on the supplied Runner; a native SDK runner must not be used concurrently.
class HimLoco {
 public:
  explicit HimLoco(Runner runner);
  PreparedInput pre_process(const std::vector<float> &observation) const;
  RawOutputs forward(const PreparedInput &input) const;
  InferenceResult post_process(const RawOutputs &raw) const;
  InferenceResult predict(const std::vector<float> &observation) const;

 private:
  Runner runner_;
};
}  // namespace himloco
