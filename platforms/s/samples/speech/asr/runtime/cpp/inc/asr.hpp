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
 * @file asr.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the ASR (Automatic Speech Recognition) model.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * ASR inference workflow on D-Robotics S100 platforms, including model
 * initialization, audio tensor preparation, BPU inference execution, and
 * CTC greedy decoding to produce transcribed text.
 *
 * @note This model supports RDK S100 and RDK S600 platforms.
 *
 * @see asr.cpp
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "preprocess.hpp"

/**
 * @brief Configuration parameters for ASR preprocessing.
 *
 * Stores audio processing hyperparameters required for the ASR inference
 * pipeline. Default values match the model's expected input format.
 *
 * @note This model only supports RDK S100.
 */
struct ASRConfig
{
    int audio_maxlen = 30000;  ///< Number of audio samples per inference chunk (at new_rate Hz)
    int new_rate     = 16000;  ///< Target audio sample rate in Hz
};

/**
 * @class ASR
 * @brief Wrapper class for ASR using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete inference pipeline, including:
 * - Model loading and initialization (via init())
 * - Float32 audio tensor preparation (copy mono audio into tensor memory)
 * - BPU inference execution
 * - CTC greedy decode to text string output
 *
 * @note This model only supports RDK S100 platform. Not thread-safe.
 */
class ASR
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int seq_length{0};                               ///< Output sequence length (time steps T)
    int vocab_size{0};                               ///< Vocabulary size V (number of tokens)

    /**
     * @brief Construct an ASR instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    ASR();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * Performs:
     * - Loading the packed model from disk
     * - Selecting the first model handle in the pack
     * - Querying input/output tensor counts and properties
     * - Allocating input/output tensor memory buffers
     * - Caching output dimensions (seq_length, vocab_size)
     *
     * @param[in] model_path Path to the quantized *.hbm model file (S100 only).
     * @retval 0        Success.
     * @retval non-zero DNN or UCP API error.
     *
     * @note Calling init() more than once is not allowed.
     */
    int32_t init(const char* model_path);

    /**
     * @brief Destructor.
     *
     * Releases all allocated tensor memory and DNN model resources.
     * Safe to call even if init() was never called or failed partially.
     */
    ~ASR();

private:
    int model_count_{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};  ///< Packed DNN handle
    int32_t input_count_{0};                          ///< Number of input tensors
    int32_t output_count_{0};                         ///< Number of output tensors
    bool inited_{false};                              ///< Whether init() has been called successfully
};


/**
 * @brief Prepare the model input tensor with raw mono audio samples.
 *
 * Copies the provided float32 audio samples into the first input tensor
 * and flushes CPU cache to DDR.
 *
 * @param[in,out] input_tensors Model input tensors to be filled with audio data.
 * @param[in]     audio_data    Mono audio samples (float32), length must match tensor shape.
 *
 * @retval 0   Success.
 * @retval -1  Shape mismatch (audio_data length != tensor length).
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    const std::vector<float>& audio_data);

/**
 * @brief Execute synchronous BPU inference on prepared input tensors.
 *
 * Creates an inference task, submits it to the UCP scheduler, waits for
 * completion, invalidates output caches for CPU access, and releases the
 * task handle.
 *
 * @param[in,out] output_tensors Output tensors filled by the runtime.
 * @param[in]     input_tensors  Prepared input tensors.
 * @param[in]     dnn_handle     DNN model handle used for inference.
 * @param[in]     sched_param    Optional UCP scheduling parameters (nullptr = default).
 *
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

/**
 * @brief Greedy CTC decode output logits into a text string.
 *
 * Iterates over time steps, selects the argmax token ID per step, maps IDs
 * to tokens via @p id2token, concatenates all tokens, and strips all
 * occurrences of "<pad>" from the result.
 *
 * @param[in] output_tensors Raw output tensors from inference.
 * @param[in] id2token       Vocabulary lookup table: index = token id, value = token string.
 * @param[in] seq_length     Number of time steps (T dimension).
 * @param[in] vocab_size     Vocabulary size (V dimension).
 *
 * @return std::string Decoded text with "<pad>" tokens removed.
 */
std::string post_process(std::vector<hbDNNTensor>& output_tensors,
                         const std::vector<std::string>& id2token,
                         int seq_length,
                         int vocab_size);
