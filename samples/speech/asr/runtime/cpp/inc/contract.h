// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace asr {
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
                              DecodeMode mode = DecodeMode::Ctc) {
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
  return result;
}
inline std::string decode_logits(const std::vector<float> &logits, size_t steps,
                                 const std::vector<std::string> &tokens,
                                 DecodeMode mode = DecodeMode::Ctc) {
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
} // namespace asr
