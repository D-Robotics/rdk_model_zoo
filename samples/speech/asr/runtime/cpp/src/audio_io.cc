// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "audio_io.h"
#include "contract.h"
#include <limits>
#include <sndfile.h>
#include <stdexcept>
namespace asr {
struct AudioReader::Impl {
  struct Close {
    void operator()(SNDFILE *p) const {
      if (p)
        sf_close(p);
    }
  };
  std::unique_ptr<SNDFILE, Close> file;
  SF_INFO info{};
  size_t block = 0, offset = 0, index = 0;
};
AudioReader::AudioReader(const std::string &path)
    : impl_(std::make_unique<Impl>()) {
  impl_->file.reset(sf_open(path.c_str(), SFM_READ, &impl_->info));
  if (!impl_->file)
    throw std::runtime_error("Cannot open audio: " + path);
  const auto &info = impl_->info;
  if (info.samplerate <= 0 || info.channels <= 0 || info.frames <= 0)
    throw std::invalid_argument(
        "Audio must contain frames with valid rate/channels");
  impl_->block = source_chunk_size(info.samplerate);
  if (impl_->block >
          std::numeric_limits<size_t>::max() / size_t(info.channels) ||
      impl_->block >
          static_cast<size_t>(std::numeric_limits<sf_count_t>::max()))
    throw std::invalid_argument("Audio chunk dimensions overflow");
}
AudioReader::~AudioReader() = default;
bool AudioReader::next(AudioChunk &chunk) {
  chunk = AudioChunk{};
  const auto &info = impl_->info;
  std::vector<float> samples(impl_->block * size_t(info.channels));
  const sf_count_t count = sf_readf_float(
      impl_->file.get(), samples.data(), static_cast<sf_count_t>(impl_->block));
  if (sf_error(impl_->file.get()) != SF_ERR_NO_ERROR || count < 0)
    throw std::runtime_error("Audio read failed: " +
                             std::string(sf_strerror(impl_->file.get())));
  if (!count)
    return false;
  if (impl_->offset > std::numeric_limits<size_t>::max() - size_t(count))
    throw std::overflow_error("Audio frame offset overflow");
  samples.resize(size_t(count) * size_t(info.channels));
  chunk = {std::move(samples), info.samplerate, info.channels, impl_->offset,
           impl_->index};
  impl_->offset += size_t(count);
  ++impl_->index;
  return true;
}
} // namespace asr
