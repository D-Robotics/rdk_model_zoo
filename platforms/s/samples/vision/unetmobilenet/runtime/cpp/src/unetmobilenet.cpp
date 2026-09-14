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
 * @file unetmobilenet.cpp
 * @brief Implement the UnetMobileNet inference pipeline using HB-DNN / UCP runtime APIs.
 *
 * This file contains the complete implementation of the UnetMobileNet wrapper
 * and its end-to-end inference flow on D-Robotics platforms:
 * - Initialize and load a packed *.hbm model, query tensor properties,
 *   and allocate stride-aware tensor buffers.
 * - Preprocess input images (direct resize with INTER_AREA + BGR->NV12).
 * - Execute synchronous BPU inference via hbDNNInferV2() and UCP APIs.
 * - Postprocess raw NHWC int32 output via argmax and resize to original size.
 *
 * @see unetmobilenet.hpp
 */

#include "unetmobilenet.hpp"
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------------------------
// UnetMobileNet class
// ---------------------------------------------------------------------------

/**
 * @brief Construct a UnetMobileNet instance in an uninitialized state.
 */
UnetMobileNet::UnetMobileNet()
{
    model_count_       = 0;
    packed_dnn_handle_ = nullptr;
    dnn_handle        = nullptr;
    input_count_       = 0;
    output_count_      = 0;
    input_h           = 0;
    input_w           = 0;
    inited_            = false;
}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * @param[in] model_path Path to the quantized *.hbm model file.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t UnetMobileNet::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    if (inited_) {
        fprintf(stderr, "UnetMobileNet::init() called twice\n");
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

    // Resize tensor storage
    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);

    // Query tensor properties
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache model input resolution from the first input tensor (Y plane)
    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    // Allocate tensor memory
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

/**
 * @brief Destructor: release tensor memory and DNN model resources.
 */
UnetMobileNet::~UnetMobileNet()
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
 * @brief Preprocess an input BGR image into NV12 model input tensors.
 *
 * Uses direct resize (INTER_AREA) instead of letterbox to match the training
 * preprocessing of the UnetMobileNet model.
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

    // Direct resize (no letterbox) with INTER_AREA to reduce aliasing
    cv::Mat resized_mat;
    resized_mat.create(input_h, input_w, img.type());
    cv::resize(img, resized_mat, resized_mat.size(), 0, 0, cv::INTER_AREA);

    return bgr_to_nv12_tensor(resized_mat, input_tensors, input_h, input_w);
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
 * @brief Postprocess UnetMobileNet outputs into a per-pixel class ID mask.
 *
 * Applies argmax over the channel axis of the int32 output tensor, then
 * resizes the result from model output resolution to the original image size.
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  orig_img_w     Original image width (pixels).
 * @param[in]  orig_img_h     Original image height (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 * @return SegmentationMask   Segmentation result with class_ids of shape
 *                            (orig_img_h × orig_img_w), type CV_32S.
 */
SegmentationMask post_process(std::vector<hbDNNTensor>& output_tensors,
                              int orig_img_w, int orig_img_h,
                              int input_w, int input_h)
{
    // Step 1: Argmax over channel axis → (H, W) class ID map
    cv::Mat raw_mask = argmax_nhwc_s32(output_tensors[0]);

    // Step 2: Resize to model input size (if output is spatially smaller)
    cv::resize(raw_mask, raw_mask, cv::Size(input_w, input_h), 0, 0, cv::INTER_NEAREST);

    // Step 3: Resize to original image size
    SegmentationMask result;
    cv::resize(raw_mask, result.class_ids, cv::Size(orig_img_w, orig_img_h), 0, 0, cv::INTER_NEAREST);
    return result;
}
