// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace asr {
struct AudioChunk {
  std::vector<float> samples; // owned, interleaved [frames, channels]
  int sample_rate = 0;
  int channels = 0;
  size_t source_start = 0;
  size_t index = 0;
};
// File reading only: no channel mixing, resampling, normalization or SDK calls.
class AudioReader {
public:
  explicit AudioReader(const std::string &path);
  ~AudioReader();
  AudioReader(const AudioReader &) = delete;
  AudioReader &operator=(const AudioReader &) = delete;
  bool next(AudioChunk &chunk); // false at clean EOF; throws on read errors
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace asr
