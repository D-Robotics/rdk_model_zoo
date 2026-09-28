// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "audio_io.h"
#include "frontend.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sndfile.h>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

using namespace asr;
template <class F> void rejected(F action) {
  bool caught = false;
  try {
    action();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main() {
  static_assert(!std::is_copy_constructible_v<AudioReader>);
  char directory[] = "/tmp/rdk-asr-audio-XXXXXX";
  assert(mkdtemp(directory));
  const auto root = std::filesystem::path(directory);
  for (int rate : {16000, 44100, 8000}) {
    const size_t frames = source_chunk_size(rate) + 123;
    std::vector<float> stereo(frames * 2);
    for (size_t i = 0; i < frames; ++i) {
      stereo[i * 2] = float(.2 * std::sin(i * .07));
      stereo[i * 2 + 1] = float(.3 * std::cos(i * .09));
    }
    SF_INFO info{};
    info.channels = 2;
    info.samplerate = rate;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    const auto path = root / (std::to_string(rate) + ".wav");
    auto file = sf_open(path.c_str(), SFM_WRITE, &info);
    assert(file);
    assert(sf_writef_float(file, stereo.data(), frames) ==
           static_cast<sf_count_t>(frames));
    assert(sf_close(file) == 0);
    AudioReader reader(path.string());
    AudioChunk chunk;
    assert(reader.next(chunk));
    assert(chunk.index == 0 && chunk.source_start == 0 && chunk.channels == 2 &&
           chunk.sample_rate == rate);
    assert(chunk.samples.size() == source_chunk_size(rate) * 2);
    auto first = prepare_chunk(chunk);
    assert(first.values.size() == 30000 && first.valid_samples > 0);
    assert(reader.next(chunk));
    assert(chunk.index == 1 && chunk.source_start == source_chunk_size(rate));
    assert(chunk.samples.size() == 246);
    auto last = prepare_chunk(chunk);
    assert(last.valid_samples > 0 && last.valid_samples < 30000);
    for (size_t i = last.valid_samples; i < 30000; ++i)
      assert(last.values[i] == 0);
    assert(!reader.next(chunk));
    assert(chunk.samples.empty());
    assert(!reader.next(chunk));
  }
  AudioChunk constant{{.5f, .5f, .5f, .5f}, 16000, 2, 0, 0};
  auto zero = prepare_chunk(constant);
  assert(zero.valid_samples == 2);
  for (auto value : zero.values)
    assert(value == 0);
  rejected([] { prepare_chunk(AudioChunk{{}, 16000, 1, 0, 0}); });
  rejected([] { prepare_chunk(AudioChunk{{1}, 16000, 2, 0, 0}); });
  rejected([] { prepare_chunk(AudioChunk{{1}, 0, 1, 0, 0}); });
  rejected([] { prepare_chunk(AudioChunk{{1}, 16000, 0, 0, 0}); });
  rejected([] {
    prepare_chunk(
        AudioChunk{{std::numeric_limits<float>::infinity()}, 16000, 1, 0, 0});
  });
  rejected([] {
    prepare_chunk(AudioChunk{std::vector<float>(30001), 16000, 1, 0, 0});
  });
  rejected([] { prepare_chunk(AudioChunk{{1}, 192000, 1, 0, 0}); });
  rejected([&] { AudioReader reader((root / "missing.wav").string()); });
  SF_INFO empty_info{};
  empty_info.channels = 1;
  empty_info.samplerate = 16000;
  empty_info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  auto empty_file =
      sf_open((root / "empty.wav").c_str(), SFM_WRITE, &empty_info);
  assert(empty_file);
  assert(sf_close(empty_file) == 0);
  rejected([&] { AudioReader reader((root / "empty.wav").string()); });
  AudioChunk large{
      {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
      16000,
      2,
      0,
      0};
  const auto finite = prepare_chunk(large);
  assert(finite.valid_samples == 1 && finite.values[0] == 0);
  std::filesystem::remove_all(root);
}
