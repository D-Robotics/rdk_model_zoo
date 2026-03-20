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

#include "asr.hpp"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <iostream>

// ============================================================
// ASR class methods
// ============================================================

/**
 * @brief Construct an ASR instance in an uninitialized state.
 *
 * No resources are allocated. Call init() to load the model.
 */
ASR::ASR() {}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * Loads the packed model, acquires the model handle, queries input/output
 * counts and tensor properties, caches output dimensions (seq_length, vocab_size),
 * and allocates memory for I/O tensors.
 *
 * @param[in] model_path Path to quantized *.hbm model file.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t ASR::init(const char* model_path)
{
    const char** model_name_list;

    // Load packed model and fetch model handle
    HBDNN_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
        "hbDNNInitializeFromFiles failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
        "hbDNNGetModelHandle failed");

    // Query I/O counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle),   "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle), "hbDNNGetOutputCount failed");

    // Query tensor properties
    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache output dims: output shape is (N=1, T, V)
    seq_length  = output_tensors[0].properties.validShape.dimensionSize[1];
    vocab_size  = output_tensors[0].properties.validShape.dimensionSize[2];

    // Allocate I/O tensor memory
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

/**
 * @brief Destructor: release tensor memory and model resources.
 */
ASR::~ASR()
{
    if (!inited_) return;

    for (int i = 0; i < input_count_;  i++) { hbUCPFree(&(input_tensors[i].sysMem)); }
    for (int i = 0; i < output_count_; i++) { hbUCPFree(&(output_tensors[i].sysMem)); }
    hbDNNRelease(packed_dnn_handle_);
}

// ============================================================
// Free pipeline functions
// ============================================================

/**
 * @brief Copy mono float audio into the model input tensor.
 *
 * Validates the tensor shape (N=1, L) and writes float32 samples into
 * tensor memory, then flushes CPU cache to DDR.
 *
 * @param[in,out] input_tensors Model input tensors.
 * @param[in]     audio_data    Mono float32 audio samples.
 * @retval 0   Success.
 * @retval -1  Shape mismatch.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    const std::vector<float>& audio_data)
{
    const hbDNNTensorShape& shape = input_tensors[0].properties.validShape;
    int N = shape.dimensionSize[0];
    int L = shape.dimensionSize[1];

    if (N != 1) {
        std::cerr << "[pre_process] Unexpected batch size: " << N << "\n";
        return -1;
    }
    if ((int)audio_data.size() != L) {
        std::cerr << "[pre_process] Audio length mismatch: got " << audio_data.size()
                  << ", expected " << L << "\n";
        return -1;
    }

    // Copy samples directly into tensor memory
    float* dst = reinterpret_cast<float*>(input_tensors[0].sysMem.virAddr);
    std::memcpy(dst, audio_data.data(), L * sizeof(float));

    // Flush CPU cache -> DDR so BPU sees the data
    hbUCPMemFlush(&input_tensors[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    return 0;
}

/**
 * @brief Execute synchronous BPU inference.
 *
 * Submits an hbDNNInferV2 task to the UCP scheduler, waits for completion,
 * invalidates output caches for CPU read, and releases the task handle.
 *
 * @param[in,out] output_tensors  Output tensors filled after inference.
 * @param[in]     input_tensors   Prepared input tensors.
 * @param[in]     dnn_handle      Model handle.
 * @param[in]     sched_param     Optional scheduling parameters (nullptr = default).
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param)
{
    hbUCPTaskHandle_t task_handle{nullptr};

    HBDNN_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
        "hbDNNInferV2 failed");

    hbUCPSchedParam default_param;
    if (sched_param == nullptr) {
        HB_UCP_INITIALIZE_SCHED_PARAM(&default_param);
        default_param.backend = HB_UCP_BPU_CORE_ANY;
        sched_param = &default_param;
    }

    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, sched_param), "hbUCPSubmitTask failed");
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),         "hbUCPWaitTaskDone failed");

    // Invalidate output caches (DDR -> CPU cache)
    for (auto& t : output_tensors) {
        hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");

    return 0;
}

/**
 * @brief Greedy CTC decode output logits to text.
 *
 * Iterates over T time steps, picks argmax over V vocab dimension,
 * maps token IDs to strings, concatenates, and removes "<pad>" tokens.
 *
 * @param[in] output_tensors Raw output tensors (shape N=1, T, V; dtype F32).
 * @param[in] id2token       id -> token string lookup table.
 * @param[in] seq_length     Number of time steps T.
 * @param[in] vocab_size     Vocabulary size V.
 * @return std::string Decoded transcription with <pad> removed.
 */
std::string post_process(std::vector<hbDNNTensor>& output_tensors,
                         const std::vector<std::string>& id2token,
                         int seq_length,
                         int vocab_size)
{
    const uint8_t*  base   = reinterpret_cast<const uint8_t*>(output_tensors[0].sysMem.virAddr);
    const int64_t*  stride = output_tensors[0].properties.stride;  // bytes: [N, T, V]

    std::ostringstream oss;

    for (int t = 0; t < seq_length; ++t) {
        int   best_id  = 0;
        float best_val = -1e30f;

        for (int v = 0; v < vocab_size; ++v) {
            const float val = *reinterpret_cast<const float*>(
                base + 0 * stride[0] + t * stride[1] + v * stride[2]);
            if (val > best_val) { best_val = val; best_id = v; }
        }

        if (best_id >= 0 && best_id < (int)id2token.size()) {
            oss << id2token[best_id];
        }
    }

    // Strip all "<pad>" tokens
    std::string text = oss.str();
    const std::string pad = "<pad>";
    size_t pos;
    while ((pos = text.find(pad)) != std::string::npos) {
        text.erase(pos, pad.length());
    }

    return text;
}
