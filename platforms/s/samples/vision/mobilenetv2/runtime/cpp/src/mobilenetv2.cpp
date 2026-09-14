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
 * @file mobilenetv2.cpp
 * @brief Implement MobileNetV2 sample runtime and helper functions.
 *
 * Note: This file provides the implementation of MobileNetV2 runtime management
 * and the standalone preprocessing / inference / postprocessing helpers.
 */

#include "mobilenetv2.hpp"
#include "nn_math.hpp"


/**
 * @brief Construct a MobileNetV2 instance.
 *
 * The object is created in an uninitialized state.
 * Call init() to load the model, query tensor properties,
 * and allocate input/output tensor buffers.
 *
 * @note The constructor does not allocate any device or host resources.
 */
MobileNetV2::MobileNetV2()
{
    model_count_ = 0;                             ///< Number of models loaded in the packed handle

    packed_dnn_handle_ = nullptr;                 ///< Packed DNN handle for managing multiple models

    dnn_handle = nullptr;                         ///< DNN handle for the specific MobileNetV2 model

    input_count_ = 0;                             ///< Number of input tensors for the model

    output_count_ = 0;                            ///< Number of output tensors for the model

    input_h = 0;                                  ///< Model input image height

    input_w = 0;                                  ///< Model input image width

    inited_ = false;                              ///< Whether init() has been successfully called.
}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * This function performs the actual initialization, including:
 * - Loading the packed model from disk
 * - Selecting the first model handle in the pack
 * - Querying input/output tensor properties
 * - Allocating input/output tensor buffers
 * - Caching model input resolution (H/W)
 *
 * Calling init() more than once is not allowed.
 *
 * @param model_path [in] Path to the quantized *.hbm model file.
 * @return int32_t 0 on success, non-zero on failure.
 */
int32_t MobileNetV2::init(const char* model_path)
{
    const char **model_name_list = nullptr;

    if (inited_) {
        fprintf(stderr, "MobileNetV2::init() called twice\n");
        return -1;
    }

    // Initialize DNN from file(s)
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
                        "hbDNNInitializeFromFiles failed");

    // Query available model names in the pack
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");

    // Use the first model in the pack
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");

    // Query input and output tensor counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle),
                        "hbDNNGetOutputCount failed");

    // Prepare tensor descriptor arrays
    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);

    // Clear sysMem fields so destructor can safely free memory
    for (int i = 0; i < input_count_; ++i) {
        std::memset(&input_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }

    for (int i = 0; i < output_count_; ++i) {
        std::memset(&output_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }

    // Fetch input tensor properties
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    }

    // Fetch output tensor properties
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache model input resolution
    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    // Allocate memory for all tensors
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;

    return 0;
}

/**
 * @brief Destructor that releases allocated resources and model handles.
 */
MobileNetV2::~MobileNetV2()
{
    // Free input tensor memory
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }

    // Free output tensor memory
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }

    // Release packed model handle
    if (packed_dnn_handle_ != nullptr) {
        hbDNNRelease(packed_dnn_handle_);
        packed_dnn_handle_ = nullptr;
    }
}

/**
 * @brief Preprocess an image and fill model input tensor memory.
 *
 * Letterbox-resizes the image to (input_w, input_h), converts BGR to NV12,
 * and writes the result into input_tensors memory.
 *
 * @param[in,out] input_tensors Model input tensors to be filled (tensor[0].sysMem must be allocated).
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
 * @param[in]     img           Input image (cv::Mat).
 * @param[in]     image_format  Input format string (only "BGR" supported).
 * @retval 0   Success.
 * @retval -1  Failure.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    const int input_w, const int input_h,
                    cv::Mat& img,
                    const std::string& image_format)
{
    if (input_tensors[0].sysMem.virAddr == nullptr) {
        fprintf(stderr, "MobileNetV2 not initialized.\n");
        return -1;
    }

    if (image_format == "BGR")
    {
        // Letterbox resize to model input resolution
        cv::Mat resized_mat;
        resized_mat.create(input_h, input_w, img.type());
        letterbox_resize(img, resized_mat);

        // Convert BGR to NV12 and write into input tensors
        bgr_to_nv12_tensor(resized_mat, input_tensors, input_h, input_w);
    }
    else
    {
        std::cerr << "[MobileNetV2::pre_process] Unsupported image_format: "
                  << image_format
                  << ". Supported formats: BGR."
                  << std::endl;
        return -1;
    }

    return 0;
}

/**
 * @brief Execute model inference on BPU.
 *
 * This function:
 * - Creates inference tasks via hbDNNInferV2()
 * - Submits it to the scheduler
 * - Waits for task to complete (blocking)
 * - Invalidates output tensor cache for CPU access
 * - Releases both task handle
 *
 * @param output_tensors [in,out] Output tensors to be filled by runtime.
 * @param input_tensors [in] Prepared model input tensors.
 * @param dnn_handle [in] DNN model handle used for inference.
 * @param sched_param [in] Optional scheduling parameters (nullptr uses default HB_UCP_BPU_CORE_ANY).
 *
 * @return int32_t 0 on success, non-zero on failure.
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

    // Prepare scheduling parameters
    hbUCPSchedParam local_param;
    hbUCPSchedParam* param_to_use = nullptr;

    if (sched_param == nullptr) {
        HB_UCP_INITIALIZE_SCHED_PARAM(&local_param);
        local_param.backend = HB_UCP_BPU_CORE_ANY;
        param_to_use = &local_param;
    } else {
        param_to_use = sched_param;
    }

    param_to_use->priority = 0;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, param_to_use),
                        "hbUCPSubmitTask failed");

    // Wait until task completion (blocking)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // Invalidate output tensor cache for CPU access
    for (int i = 0; i < (int)output_tensors.size(); i++) {
        hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release task handle
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");

    return 0;
}

/**
 * @brief Postprocess model outputs into final classification results.
 *
 * Reads probabilities directly from the output tensor without applying
 * softmax, since the MobileNetV2 model output node already contains
 * post-softmax probability values.
 *
 * @param[out] topk_results  Top-K classification results sorted by probability.
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Postprocess configuration (reserved, unused currently).
 * @param[in]  top_k          Number of top classes to keep.
 *
 * @note MobileNetV2 "prob" output is already a probability distribution;
 *       applying softmax again would incorrectly flatten the scores.
 */
void post_process(std::vector<Classification>& topk_results,
                  std::vector<hbDNNTensor>& output_tensors,
                  const MobileNetV2Config& config, int top_k)
{
    std::vector<Classification> cls_result;

    // Read probabilities directly without softmax via utils helper
    tensor_to_cls_results(output_tensors[0], cls_result);

    // Select top-K classes by probability
    get_topk_result(cls_result, topk_results, top_k);
}
