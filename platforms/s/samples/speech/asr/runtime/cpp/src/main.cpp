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
 * @file main.cpp
 * @brief ASR demo entry point: transcribe an audio file chunk-by-chunk.
 *
 * Pipeline:
 *  1) Parse CLI flags
 *  2) Load ASR model and vocabulary
 *  3) Stream audio in fixed-length chunks (mono, resampled, normalized)
 *  4) For each chunk: pre_process -> infer -> post_process (greedy decode)
 *  5) Concatenate chunk texts and print the full transcription
 *
 * @note This model supports RDK S100 and RDK S600 platforms.
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "asr.hpp"
#include "audio_chunk_reader.hpp"
#include "file_io.hpp"

// -------------------- Command-line flags --------------------
#ifdef SOC_S600
DEFINE_string(model_path, "/opt/hobot/model/s600/basic/asr.hbm",
              "Path to BPU quantized *.hbm model file.");
#else
DEFINE_string(model_path, "/opt/hobot/model/s100/basic/asr.hbm",
              "Path to BPU quantized *.hbm model file.");
#endif
DEFINE_string(test_sound, "../../../test_data/chi_sound.wav",
              "Path to input audio file (.wav or .flac).");
DEFINE_string(vocab_file, "../../../test_data/vocab.json",
              "Path to vocabulary JSON file (token -> id mapping).");
DEFINE_int32(audio_maxlen, 30000,
             "Number of audio samples per inference chunk at new_rate Hz.");
DEFINE_int32(new_rate, 16000,
             "Target audio sample rate in Hz.");

/**
 * @brief Program entry point.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Command-line argument values.
 * @return int Process exit code (0 on success).
 */
int main(int argc, char** argv)
{
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // Load ASR model
    ASR asr;
    if (asr.init(FLAGS_model_path.c_str()) != 0) {
        std::cerr << "Failed to initialize ASR model: " << FLAGS_model_path << std::endl;
        return 1;
    }

    // Load vocabulary: id2token lookup table
    auto id2token = load_id2token(FLAGS_vocab_file);

    // Create streaming audio reader
    AudioChunkReader reader(FLAGS_test_sound, FLAGS_audio_maxlen, FLAGS_new_rate);

    std::vector<float> chunk;
    std::string full_text;

    // Stream chunks until EOF
    while (reader.next(chunk)) {
        pre_process(asr.input_tensors, chunk);
        infer(asr.output_tensors, asr.input_tensors, asr.dnn_handle);
        full_text += post_process(asr.output_tensors, id2token, asr.seq_length, asr.vocab_size);
    }

    std::cout << "Transcription:\n" << full_text << std::endl;

    return 0;
}
