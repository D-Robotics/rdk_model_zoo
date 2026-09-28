// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "frontend.h"
#include <functional>
#include <utility>
namespace asr {
// Transport returns owned raw FLOAT32 logits [1,steps,3503], no activation.
using Runner = std::function<std::vector<float>(const std::vector<float> &)>;
class ASR {
public:
  ASR(Runner runner, size_t steps, std::vector<std::string> vocabulary,
      DecodeMode mode = DecodeMode::Ctc)
      : runner_(std::move(runner)), steps_(steps),
        vocabulary_(std::move(vocabulary)), mode_(mode) {
    validate_vocabulary(vocabulary_);
    if (!runner_ || !steps_ || vocabulary_.size() != 3503 ||
        steps_ > std::numeric_limits<size_t>::max() / vocabulary_.size() ||
        (mode_ != DecodeMode::Ctc && mode_ != DecodeMode::Legacy))
      throw std::invalid_argument("ASR requires a runner, positive steps, 3503 "
                                  "tokens and a valid decoder mode");
  }
  PreparedChunk pre_process(const AudioChunk &audio) const {
    return prepare_chunk(audio);
  }
  std::vector<float> forward(const PreparedChunk &input) const {
    if (input.values.size() != 30000 || !input.valid_samples ||
        input.valid_samples > 30000 ||
        std::any_of(input.values.begin(), input.values.end(),
                    [](float v) { return !std::isfinite(v); }))
      throw std::invalid_argument(
          "Expected finite prepared audio [1,30000] with valid sample count");
    return runner_(input.values);
  }
  std::string post_process(const std::vector<float> &raw) const {
    return decode_logits(raw, steps_, vocabulary_, mode_);
  }
  std::string predict(const AudioChunk &audio) const {
    return post_process(forward(pre_process(audio)));
  }

private:
  Runner runner_;
  size_t steps_;
  std::vector<std::string> vocabulary_;
  DecodeMode mode_;
};
} // namespace asr
