// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "frontend.hpp"
#include <climits>
#include <samplerate.h>
namespace asr {
PreparedChunk prepare_chunk(const AudioChunk &chunk) {
  if (chunk.sample_rate <= 0 || chunk.channels <= 0 || chunk.samples.empty() ||
      chunk.samples.size() % size_t(chunk.channels))
    throw std::invalid_argument("Invalid interleaved audio geometry");
  const size_t frames = chunk.samples.size() / size_t(chunk.channels);
  if (frames > source_chunk_size(chunk.sample_rate) || frames > LONG_MAX)
    throw std::invalid_argument("Waveform exceeds one source chunk");
  std::vector<float> mono(frames);
  for (size_t i = 0; i < frames; ++i) {
    double sum = 0;
    for (int ch = 0; ch < chunk.channels; ++ch) {
      const float value =
          chunk.samples[i * size_t(chunk.channels) + size_t(ch)];
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite audio");
      sum += value;
    }
    mono[i] = static_cast<float>(sum / chunk.channels);
  }
  if (chunk.sample_rate != 16000) {
    const double ratio = 16000.0 / chunk.sample_rate;
    if (!src_is_valid_ratio(ratio))
      throw std::invalid_argument("Unsupported sample-rate ratio");
    const auto count = std::llround(frames * ratio);
    if (count < 1 || count > LONG_MAX)
      throw std::invalid_argument("Chunk too short for a target sample");
    std::vector<float> resampled(static_cast<size_t>(count));
    SRC_DATA data{};
    data.data_in = mono.data();
    data.input_frames = static_cast<long>(frames);
    data.data_out = resampled.data();
    data.output_frames = static_cast<long>(count);
    data.src_ratio = ratio;
    data.end_of_input = 1;
    const int error = src_simple(&data, SRC_SINC_BEST_QUALITY, 1);
    if (error)
      throw std::runtime_error("Resample failed: " +
                               std::string(src_strerror(error)));
    if (data.output_frames_gen <= 0 || data.output_frames_gen > count)
      throw std::runtime_error("Resampler produced invalid output length");
    resampled.resize(static_cast<size_t>(data.output_frames_gen));
    mono = std::move(resampled);
  }
  return normalize_and_pad(mono);
}
} // namespace asr
