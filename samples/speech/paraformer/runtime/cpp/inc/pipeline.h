// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "contract.h"
#include <array>
#include <functional>
#include <optional>

namespace paraformer {
struct PredictorOutput {
  std::vector<float> weights;
  std::vector<float> hidden;
};
struct DecoderInput {
  const std::vector<float> &context;
  const std::vector<float> &acoustic;
  int32_t token_count;
  const std::array<float, 512> &bias;
};
struct Timings {
  double encoder_ms = 0;
  double predictor_ms = 0;
  double cif_ms = 0;
  std::optional<double> decoder_ms;
};
struct Prediction {
  std::string text;
  std::vector<int> token_ids;
  int32_t token_count = 0;
  bool decoder_executed = false;
  Timings timings;
};
using Encoder = std::function<std::vector<float>(const std::vector<float> &)>;
using Predictor = std::function<PredictorOutput(const std::vector<float> &)>;
using Decoder = std::function<std::vector<float>(const DecoderInput &)>;

// Application composition, not a single model forward method. Each callable
// must synchronously return owned raw arrays; SDK resources remain its concern.
class Pipeline {
public:
  Pipeline(Encoder encoder, Predictor predictor, Decoder decoder,
           std::vector<std::string> vocabulary);
  Prediction predict(const std::vector<float> &features,
                     int valid_frames) const;

private:
  Encoder encoder_;
  Predictor predictor_;
  Decoder decoder_;
  std::vector<std::string> vocabulary_;
  const std::array<float, 512> bias_{};
};
} // namespace paraformer
