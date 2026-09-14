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
 * @file lanenet.cpp
 * @brief Implement the LaneNet inference pipeline using HB-DNN / UCP runtime APIs.
 *
 * This file contains the complete implementation of the LaneNet wrapper and its
 * end-to-end inference flow on D-Robotics S100 platforms:
 * - Initialize and load a packed *.hbm model, query tensor properties,
 *   and allocate stride-aware tensor buffers.
 * - Preprocess input BGR images (direct resize + BGR->RGB + ImageNet normalization)
 *   and write float32 CHW data into input tensor memory.
 * - Execute synchronous BPU inference via hbDNNInferV2() and UCP scheduling APIs.
 * - Postprocess raw F32 instance embedding and S64 binary prediction outputs
 *   into visualizable uint8 images.
 *
 * @note This model only supports RDK S100 platform.
 *
 * @see lanenet.hpp
 */

#include "lanenet.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// LaneNet class
// ---------------------------------------------------------------------------

/**
 * @brief Construct a LaneNet instance in an uninitialized state.
 */
LaneNet::LaneNet()
{
    model_count_       = 0;
    packed_dnn_handle_ = nullptr;
    dnn_handle         = nullptr;
    input_count_       = 0;
    output_count_      = 0;
    input_h            = 0;
    input_w            = 0;
    inited_            = false;
}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * @param[in] model_path Path to the quantized *.hbm model file.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t LaneNet::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    if (inited_) {
        fprintf(stderr, "LaneNet::init() called twice\n");
        return -1;
    }

    // Load model from file
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
                        "hbDNNInitializeFromFiles failed");

    // Retrieve model names and select the first model
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");

    // Query input/output tensor counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle),
                        "hbDNNGetOutputCount failed");

    // Resize tensor storage and zero sysMem
    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);
    for (int i = 0; i < input_count_; ++i) {
        std::memset(&input_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }
    for (int i = 0; i < output_count_; ++i) {
        std::memset(&output_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }

    // Query tensor properties
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache model input resolution from NCHW layout: [N, C, H, W]
    input_h = input_tensors[0].properties.validShape.dimensionSize[2];
    input_w = input_tensors[0].properties.validShape.dimensionSize[3];

    // Allocate tensor memory
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

/**
 * @brief Destructor: release tensor memory and DNN model resources.
 */
LaneNet::~LaneNet()
{
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    if (packed_dnn_handle_) {
        hbDNNRelease(packed_dnn_handle_);
    }
}

// ---------------------------------------------------------------------------
// Free pipeline functions
// ---------------------------------------------------------------------------

/**
 * @brief Preprocess an input BGR image into float32 NCHW model input tensors.
 *
 * @param[in,out] input_tensors Model input tensors to be filled.
 * @param[in]     img           Input image in BGR format.
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
 * @param[in]     image_format  Input format string (only "BGR" supported).
 * @retval 0        Success.
 * @retval -1       Unsupported image format.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    cv::Mat& img,
                    const int input_w, const int input_h,
                    const std::string& image_format)
{
    if (image_format != "BGR") {
        fprintf(stderr, "Unsupported image_format: %s\n", image_format.c_str());
        return -1;
    }

    // Direct resize (no letterbox) with INTER_AREA
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_AREA);

    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float32 and normalize to [0, 1]
    cv::Mat rgb_f32;
    rgb.convertTo(rgb_f32, CV_32F, 1.0 / 255.0);

    // Split into individual channel planes
    std::vector<cv::Mat> ch(3);
    cv::split(rgb_f32, ch);

    // Apply ImageNet mean/std normalization per channel
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std_v[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - mean[c]) / std_v[c];
    }

    // Write CHW float32 data into input tensor (with stride awareness)
    return write_chw32_to_tensor(ch, input_tensors);
}

/**
 * @brief Execute synchronous BPU inference on prepared input tensors.
 *
 * @param[in,out] output_tensors Output tensors to be filled by runtime.
 * @param[in]     input_tensors  Prepared input tensors.
 * @param[in]     dnn_handle     DNN model handle.
 * @param[in]     sched_param    Optional UCP scheduling parameters.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param)
{
    hbUCPTaskHandle_t task_handle{nullptr};

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
                        "hbDNNInferV2 failed");

    // Configure and submit to BPU scheduler
    hbUCPSchedParam ctrl_param;
    if (sched_param) {
        ctrl_param = *sched_param;
    } else {
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    }
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "hbUCPSubmitTask failed");

    // Wait for inference to complete (blocking)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "hbUCPWaitTaskDone failed");

    // Invalidate CPU cache for output tensors
    for (auto& t : output_tensors) {
        hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
    return 0;
}

/**
 * @brief Postprocess LaneNet outputs into instance and binary segmentation masks.
 *
 * @param[out] instance_pred  Instance segmentation mask (CV_8UC3, H x W).
 * @param[out] binary_pred    Binary lane mask (CV_8UC1, H x W).
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 */
void post_process(cv::Mat& instance_pred,
                  cv::Mat& binary_pred,
                  std::vector<hbDNNTensor>& output_tensors,
                  int input_w, int input_h)
{
    // --- Instance segmentation: output_tensors[0], F32 NCHW [1, 3, H, W] ---
    {
        const hbDNNTensor& tensor = output_tensors[0];
        const int64_t* stride = tensor.properties.stride;  // bytes: [N, C, H, W]
        const uint8_t* base   = static_cast<const uint8_t*>(tensor.sysMem.virAddr);

        // Build 3 float channels then convert to uint8
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            channels[c] = cv::Mat(input_h, input_w, CV_32F);
            for (int h = 0; h < input_h; ++h) {
                const float* src = reinterpret_cast<const float*>(
                    base + c * stride[1] + h * stride[2]);
                float* dst = channels[c].ptr<float>(h);
                for (int w = 0; w < input_w; ++w) {
                    // Clamp to [0, 1] before scaling; values are logits in ~[0, 1]
                    dst[w] = std::min(std::max(src[w], 0.0f), 1.0f) * 255.0f;
                }
            }
        }

        // Merge 3 channels into a 3-channel float Mat, then convert to uint8
        cv::Mat merged_f32;
        cv::merge(channels, merged_f32);
        merged_f32.convertTo(instance_pred, CV_8UC3);
    }

    // --- Binary segmentation: output_tensors[1], S64 NCHW [1, 1, H, W] ---
    {
        const hbDNNTensor& tensor = output_tensors[1];
        const int64_t* stride = tensor.properties.stride;  // bytes: [N, C, H, W]
        const uint8_t* base   = static_cast<const uint8_t*>(tensor.sysMem.virAddr);

        binary_pred = cv::Mat(input_h, input_w, CV_8UC1);
        for (int h = 0; h < input_h; ++h) {
            uint8_t* dst = binary_pred.ptr<uint8_t>(h);
            for (int w = 0; w < input_w; ++w) {
                // Binary prediction: int64 value 0 or 1, scale to 0 or 255
                const int64_t val = *reinterpret_cast<const int64_t*>(
                    base + h * stride[2] + w * stride[3]);
                dst[w] = static_cast<uint8_t>(val != 0 ? 255 : 0);
            }
        }
    }
}
