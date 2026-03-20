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

/**
 * @file audio_chunk_reader.hpp
 * @brief Streaming audio reader with mono conversion, resampling,
 *        z-score normalization, and fixed-length chunking.
 *
 * Reads an audio file chunk-by-chunk using libsndfile and libsamplerate.
 * Each call to next() yields exactly audio_maxlen samples at new_rate Hz,
 * ready to be passed to the ASR model input tensor.
 *
 * @note This model supports RDK S100 and RDK S600 platforms.
 */

#pragma once

#include <sndfile.h>
#include <samplerate.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @class AudioChunkReader
 * @brief Reads an audio file in fixed-size chunks, applies mono conversion, resampling,
 *        z-score normalization, and outputs fixed-length float samples.
 *
 * This utility is designed for audio preprocessing before feeding to ML models.
 * Each chunk is independent — no overlap or state is carried across calls to next().
 */
class AudioChunkReader {
private:
    SNDFILE*  sndfile_ = nullptr;  ///< libsndfile handle to the opened audio file
    SF_INFO   sfinfo_{};           ///< Audio file information (sample rate, channels, frames)
    int orig_sr_ = 0;              ///< Original sample rate of the audio file
    int channels_ = 0;             ///< Number of channels in the audio file
    int audio_maxlen_ = 0;         ///< Target fixed length for output chunks (in samples)
    int new_rate_ = 0;             ///< Target sample rate for resampling

    int read_size_frames_ = 0;     ///< Number of frames to read per chunk before resampling
    std::vector<float> read_buf_;  ///< Raw interleaved multi-channel audio buffer
    std::vector<float> mono_;      ///< Mono buffer after channel averaging

public:
    /**
     * @brief Construct a new AudioChunkReader object.
     *
     * Opens an audio file, initializes buffers, and computes the read size needed
     * to produce roughly @p audio_maxlen samples after resampling.
     *
     * @param[in] path         Path to the input audio file.
     * @param[in] audio_maxlen Target fixed chunk length (in samples) after resampling.
     * @param[in] new_rate     Target sample rate for resampling (Hz).
     * @throw std::runtime_error If the file cannot be opened.
     */
    AudioChunkReader(const std::string& path, int audio_maxlen, int new_rate);

    /**
     * @brief Destroy the AudioChunkReader object.
     *
     * Closes the audio file handle if open.
     */
    ~AudioChunkReader();

    /**
     * @brief Read and process the next chunk of audio data.
     *
     * This method:
     * 1. Reads a block of frames from the file.
     * 2. Converts to mono (averaging channels if > 1).
     * 3. Resamples to new_rate_ if necessary.
     * 4. Applies z-score normalization (mean = 0, stddev = 1).
     * 5. Pads or truncates to exactly audio_maxlen_ samples.
     *
     * @param[out] out_chunk Processed audio samples (size = audio_maxlen_).
     * @return bool True if a chunk was read and processed, false if end of file.
     * @throw std::runtime_error If resampling fails.
     */
    bool next(std::vector<float>& out_chunk);
};
