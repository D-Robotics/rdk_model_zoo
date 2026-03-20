/*
 * Copyright (c) 2025 D-Robotics Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audio_chunk_reader.hpp"

/**
 * @brief Construct a new AudioChunkReader object.
 *
 * Opens the audio file, reads its metadata, and prepares internal buffers
 * sized to read roughly audio_maxlen samples per chunk before resampling.
 *
 * @param[in] path         Path to the input audio file.
 * @param[in] audio_maxlen Target fixed chunk length (in samples) after resampling.
 * @param[in] new_rate     Target sample rate for resampling (Hz).
 * @throw std::runtime_error If the audio file cannot be opened.
 */
AudioChunkReader::AudioChunkReader(const std::string& path, int audio_maxlen, int new_rate)
        : audio_maxlen_(audio_maxlen), new_rate_(new_rate)
{
    sfinfo_ = {};
    sndfile_ = sf_open(path.c_str(), SFM_READ, &sfinfo_);
    if (!sndfile_) throw std::runtime_error("Failed to open audio file: " + path);

    orig_sr_  = sfinfo_.samplerate;
    channels_ = sfinfo_.channels;

    // Compute frames to read to roughly match target length after resampling
    read_size_frames_ = static_cast<int>(std::ceil(audio_maxlen_ * (double)orig_sr_ / new_rate_));
    read_buf_.resize(static_cast<size_t>(read_size_frames_) * channels_);
    mono_.resize(static_cast<size_t>(read_size_frames_));
}

/**
 * @brief Destroy the AudioChunkReader object.
 *
 * Closes the open audio file handle if valid.
 */
AudioChunkReader::~AudioChunkReader()
{
    if (sndfile_) sf_close(sndfile_);
}

/**
 * @brief Read and process the next chunk of audio data.
 *
 * Steps: read frames -> mono conversion -> resample -> z-score normalize -> pad/truncate.
 *
 * @param[out] out_chunk Processed audio samples (size = audio_maxlen_).
 * @return bool True if a chunk was successfully read and processed, false at EOF.
 * @throw std::runtime_error If resampling fails.
 */
bool AudioChunkReader::next(std::vector<float>& out_chunk)
{
    out_chunk.clear();

    // Read a block of frames (multi-channel if needed)
    sf_count_t frames_read = sf_readf_float(sndfile_, read_buf_.data(), read_size_frames_);
    if (frames_read <= 0) return false;

    // Convert to mono
    mono_.resize(static_cast<size_t>(frames_read));
    if (channels_ > 1) {
        for (sf_count_t i = 0; i < frames_read; ++i) {
            float sum = 0.f;
            for (int ch = 0; ch < channels_; ++ch) {
                sum += read_buf_[i * channels_ + ch];
            }
            mono_[static_cast<size_t>(i)] = sum / channels_;
        }
    } else {
        std::copy(read_buf_.begin(), read_buf_.begin() + frames_read, mono_.begin());
    }

    // Resample if sample rate differs
    std::vector<float> resampled;
    if (orig_sr_ != new_rate_) {
        const double ratio = static_cast<double>(new_rate_) / orig_sr_;
        const int out_samples = static_cast<int>(std::llround(mono_.size() * ratio));
        resampled.resize(out_samples);

        SRC_DATA d{};
        d.data_in       = mono_.data();
        d.input_frames  = static_cast<long>(mono_.size());
        d.data_out      = resampled.data();
        d.output_frames = out_samples;
        d.src_ratio     = ratio;
        d.end_of_input  = 0;

        if (src_simple(&d, SRC_SINC_BEST_QUALITY, 1) != 0) {
            throw std::runtime_error("Resample failed");
        }
        resampled.resize(static_cast<size_t>(d.output_frames_gen));
    } else {
        resampled = mono_;
    }

    // Apply z-score normalization (mean=0, std=1)
    if (!resampled.empty()) {
        const float mean = std::accumulate(resampled.begin(), resampled.end(), 0.0f) / resampled.size();
        float sq_sum = 0.f;
        for (float v : resampled) {
            float dv = v - mean;
            sq_sum += dv * dv;
        }
        const float stddev = std::sqrt(sq_sum / std::max<size_t>(1, resampled.size()));
        if (stddev > 1e-9f) {
            for (float& v : resampled) {
                v = (v - mean) / stddev;
            }
        }
    }

    // Pad or truncate to fixed length
    out_chunk = std::move(resampled);
    if ((int)out_chunk.size() < audio_maxlen_) {
        out_chunk.resize(audio_maxlen_, 0.f);
    } else if ((int)out_chunk.size() > audio_maxlen_) {
        out_chunk.resize(audio_maxlen_);
    }

    return true;
}
