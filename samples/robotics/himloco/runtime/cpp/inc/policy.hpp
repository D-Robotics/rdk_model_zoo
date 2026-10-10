// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

/// Native runtime configuration; the model path is an explicit local published
/// BIN, never downloaded.
struct NativeConfig {
  std::string model_path =
      "samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin";
  int priority = -1; // SDK default or [0,255].
};

struct TensorMetadata {
  std::string name;
  std::vector<int> valid_shape, aligned_shape;
  int tensor_layout = 0, tensor_type = 0, quanti_type = 0,
      aligned_byte_size = 0;
};

/// Model admission gate: require actual X5 identity and the exact published
/// model SHA before SDK load.
using ModelGate = std::function<void(const std::string &model_path)>;
void verify_native_model(const std::string &model_path);

/// Runtime facts for run reports; empty when a runner is injected.
struct RuntimeInfo {
  TensorMetadata input, output;
  std::string model_name, runtime_version;
  int priority = -1;
  bool valid = false;
};

/// HIMLoco locomotion policy and its X5 BPU runtime. The native constructor
/// admits the model through `gate` and loads the SDK runtime; the runner
/// constructor injects a deterministic substitute for host tests. Stage values
/// own their memory and can be interleaved. Thread safety depends on the
/// supplied Runner; the native SDK runtime must not be used concurrently.
class HimLoco {
 public:
  explicit HimLoco(const NativeConfig &config,
                   ModelGate gate = &verify_native_model);
  explicit HimLoco(Runner runner, RuntimeInfo info = {});

  PreparedInput preprocess(const std::vector<float> &observation) const;
  RawOutputs infer(const PreparedInput &input) const;
  InferenceResult postprocess(const RawOutputs &raw) const;
  InferenceResult predict(const std::vector<float> &observation) const;

  const TensorMetadata &input_metadata() const;
  const TensorMetadata &output_metadata() const;
  const std::string &model_name() const;
  const std::string &runtime_version() const;
  int priority() const;

 private:
  Runner runner_;
  RuntimeInfo info_;
};
}  // namespace himloco
