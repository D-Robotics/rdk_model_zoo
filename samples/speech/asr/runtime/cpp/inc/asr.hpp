// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "platform_identity.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace asr {
// --- Decode contract -------------------------------------------------------
enum class DecodeMode { Ctc, Legacy };
inline void validate_vocabulary(const std::vector<std::string> &tokens) {
  if (tokens.empty() || tokens[0] != "<pad>" ||
      std::set<std::string>(tokens.begin(), tokens.end()).size() !=
          tokens.size() ||
      std::any_of(tokens.begin(), tokens.end(),
                  [](const auto &token) { return token.empty(); }))
    throw std::invalid_argument(
        "Expected unique ordered tokens with <pad> ID 0");
}
inline std::string decode_ids(const std::vector<int> &ids,
                              const std::vector<std::string> &tokens,
                              DecodeMode mode = DecodeMode::Legacy) {
  validate_vocabulary(tokens);
  if (mode != DecodeMode::Ctc && mode != DecodeMode::Legacy)
    throw std::invalid_argument("Invalid decode mode");
  std::string result;
  int previous = -1;
  for (int id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= tokens.size())
      throw std::invalid_argument("Token outside vocabulary");
    if (id != 0 && (mode == DecodeMode::Legacy || id != previous))
      result += tokens[id];
    previous = id;
  }
  if (mode == DecodeMode::Ctc) {
    // Wav2Vec2 uses | as the word delimiter; legacy keeps source tokens.
    std::replace(result.begin(), result.end(), '|', ' ');
    const auto first = result.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos)
      return "";
    const auto last = result.find_last_not_of(" \t\n\r\f\v");
    result = result.substr(first, last - first + 1);
  }
  return result;
}
inline std::string decode_logits(const std::vector<float> &logits, size_t steps,
                                 const std::vector<std::string> &tokens,
                                 DecodeMode mode = DecodeMode::Legacy) {
  validate_vocabulary(tokens);
  if (steps == 0 ||
      steps > std::numeric_limits<size_t>::max() / tokens.size() ||
      logits.size() != steps * tokens.size())
    throw std::invalid_argument("Invalid logits dimensions");
  std::vector<int> ids;
  ids.reserve(steps);
  for (size_t t = 0; t < steps; ++t) {
    float best = -std::numeric_limits<float>::infinity();
    int index = 0;
    for (size_t v = 0; v < tokens.size(); ++v) {
      const float value = logits[t * tokens.size() + v];
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite logits");
      if (value > best) {
        best = value;
        index = static_cast<int>(v);
      }
    }
    ids.push_back(index);
  }
  return decode_ids(ids, tokens, mode);
}
inline size_t source_chunk_size(int rate) {
  if (rate <= 0)
    throw std::invalid_argument("Sample rate must be positive");
  return static_cast<size_t>((int64_t(30000) * rate + 15999) / 16000);
}

// --- Chunk contract shared by the frontend, model and CLI -------------------
struct AudioChunk {
  std::vector<float> samples; // owned, interleaved [frames, channels]
  int sample_rate = 0;
  int channels = 0;
  size_t source_start = 0;
  size_t index = 0;
};
struct PreparedChunk {
  std::vector<float> values;
  size_t valid_samples;
};
inline PreparedChunk normalize_and_pad(const std::vector<float> &mono) {
  if (mono.empty())
    throw std::invalid_argument("Empty waveform");
  double sum = 0;
  for (float value : mono) {
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite audio");
    sum += value;
  }
  const double mean = sum / mono.size();
  double variance = 0;
  for (float value : mono) {
    const double d = value - mean;
    variance += d * d;
  }
  const double denom = std::sqrt(variance / mono.size() + 1e-5);
  PreparedChunk result{std::vector<float>(30000, 0.f),
                       std::min<size_t>(mono.size(), 30000)};
  for (size_t i = 0; i < result.valid_samples; ++i) {
    result.values[i] = static_cast<float>((mono[i] - mean) / denom);
    if (!std::isfinite(result.values[i]))
      throw std::invalid_argument("Nonfinite normalized audio");
  }
  return result;
}

// --- Native runtime and admission gate --------------------------------------
struct SdkModel {
  std::string path, target;
};
using SdkPreflight = std::function<void(const SdkModel &)>;
struct SdkMetadata {
  std::string model_name;
  size_t steps = 0;
  std::vector<int64_t> input_strides, output_strides;
  size_t input_bytes = 0, output_bytes = 0;
};
inline constexpr const char *kVocabularySha256 =
    "33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7";
void verify_preflight(const SdkModel &, const std::string &expected_sha256,
                      const std::string &vocabulary,
                      const rdk::NativeIdentity &actual);
SdkPreflight make_preflight(std::string expected_sha256,
                            std::string vocabulary);
// UCP only. Callback is mandatory and runs before any SDK call.
class SdkRunner {
public:
  SdkRunner(SdkModel model, SdkPreflight preflight);
  ~SdkRunner();
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  const SdkMetadata &metadata() const;
  std::vector<float> infer(const std::vector<float> &prepared);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// --- Model ------------------------------------------------------------------
// Transport returns owned raw FLOAT32 logits [1,steps,3503], no activation.
using Runner = std::function<std::vector<float>(const std::vector<float> &)>;
// Owned result of one predict call: the decoded chunk transcript plus the
// normalization/chunk facts the run report records.
struct Prediction {
  std::string text;
  size_t valid_samples = 0;
};
class ASR {
public:
  // Injectable stages: tests and alternative transports supply the runner.
  ASR(Runner runner, size_t steps, std::vector<std::string> vocabulary,
      DecodeMode mode = DecodeMode::Legacy);
  // Native runtime: owns the UCP SdkRunner and its observed metadata; the
  // mandatory preflight gate runs before any SDK call inside the runner.
  ASR(const SdkModel &model, const SdkPreflight &preflight,
      std::vector<std::string> vocabulary,
      DecodeMode mode = DecodeMode::Legacy);
  // The native construction wires runner_ to the runner owned by this object;
  // moving would leave it dereferencing the moved-from instance.
  ASR(const ASR &) = delete;
  ASR &operator=(const ASR &) = delete;
  ASR(ASR &&) = delete;
  ASR &operator=(ASR &&) = delete;
  PreparedChunk preprocess(const AudioChunk &audio) const;
  std::vector<float> infer(const PreparedChunk &input) const;
  std::string postprocess(const std::vector<float> &raw) const;
  Prediction predict(const AudioChunk &audio) const;
  // Observed tensor metadata; only the native constructor owns any.
  const SdkMetadata &metadata() const;

private:
  std::unique_ptr<SdkRunner> native_; // declared first: outlives runner_
  Runner runner_;
  size_t steps_;
  std::vector<std::string> vocabulary_;
  DecodeMode mode_;
};
} // namespace asr
