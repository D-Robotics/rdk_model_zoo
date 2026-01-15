/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
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
 * @file yolov5.cc
 * @brief Implement the YOLOv5x inference pipeline using HB-DNN / UCP runtime APIs.
 *
 *        This file contains the concrete implementation of the YOLOv5x wrapper
 *        and its end-to-end inference flow on Horizon Robotics platforms:
 *        - Initialize and load a packed *.hbm model, query tensor properties,
 *          and allocate stride-aware tensor buffers.
 *        - Preprocess input images (letterbox resize and BGR->NV12 conversion)
 *          and write data into input tensor memory with proper cache handling.
 *        - Execute synchronous inference tasks on BPU via hbDNNInferV2() and UCP
 *          scheduling APIs, then invalidate output caches for CPU-side access.
 *        - Postprocess raw outputs by dequantization, decoding, thresholding,
 *          NMS, and coordinate scaling back to the original image space.
 *
 *        The implementation is designed as a reference pipeline that separates
 *        preprocessing, runtime execution, and postprocessing into reusable steps
 *        while keeping resource ownership and cleanup explicit.
 */


#include "yolov5.hpp"
#include <chrono>

/**
 * @brief Construct a YOLOv5x instance.
 *
 * The object is created in an uninitialized state. Call init() to load the
 * model, query tensor properties, and allocate tensor buffers.
 */
YOLOv5x::YOLOv5x()
{
    // Number of models contained in the packed DNN handle
    model_count = 0;

    // Packed DNN handle (may contain multiple models)
    packed_dnn_handle = nullptr;

    // Handle of the YOLOv5x model used for inference
    dnn_handle = nullptr;

    // Number of input tensors required by the model
    input_count = 0;

    // Number of output tensors produced by the model
    output_count = 0;

    // Model input height (pixels)
    input_h = 0;

    // Model input width (pixels)
    input_w = 0;

    inited = false;
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
int32_t YOLOv5x::init(const char* model_path)
{
    const char **model_name_list = nullptr;

    if (inited) {
        fprintf(stderr, "YOLOv5x::init() called twice\n");
        return -1;
    }

    // Initialize DNN from file(s)
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path, 1),
                        "hbDNNInitializeFromFiles failed");

    // Query available model names in the pack
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
                        "hbDNNGetModelNameList failed");

    // Use the first model in the pack
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
                        "hbDNNGetModelHandle failed");

    // Query input and output tensor counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                        "hbDNNGetOutputCount failed");

    // Prepare tensor descriptor arrays
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    // Clear sysMem fields so destructor can safely free memory
    for (int i = 0; i < input_count; ++i) {
        std::memset(&input_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }

    for (int i = 0; i < output_count; ++i) {
        std::memset(&output_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    }

    // Fetch input tensor properties
    for (int i = 0; i < input_count; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    }

    // Fetch output tensor properties
    for (int i = 0; i < output_count; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache model input resolution
    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    // Allocate memory for all tensors
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited = true;

    return 0;
}

/**
* @brief Destructor.
*
* Releases allocated tensor memory and DNN resources.
* Safe to call even if init() failed partially.
*/
YOLOv5x::~YOLOv5x()
{
    // Free input tensor memory
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }

    // Free output tensor memory
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }

    // Release packed model handle
    if (packed_dnn_handle != nullptr) {
        hbDNNRelease(packed_dnn_handle);
        packed_dnn_handle = nullptr;
    }
}

/**
 * @brief Preprocess an input image into model input tensor buffers.
 *
 * Current implementation:
 * - Letterbox resize to model input resolution (input_w/input_h)
 * - Convert BGR image to NV12 as required by the compiled model
 * - Write the converted data into input_tensors[0].sysMem
 *
 * @param input_tensors [in,out] Model input tensors to be filled.
 * @param input_w [in] Model input width in pixels.
 * @param input_h [in] Model input height in pixels.
 * @param img [in] Input image (OpenCV Mat).
 * @param image_format [in] Input image format string (only "BGR" is supported).
 *
 * @return int32_t 0 on success, non-zero on failure.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    cv::Mat& img,
                    const int input_w, const int input_h,
                    const std::string& image_format)
{
    if (input_tensors[0].sysMem.virAddr == nullptr) {
        fprintf(stderr, "YOLOv5x not initialized.\n");
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
        std::cerr << "[YOLOv5x::pre_process] Unsupported image_format: "
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
    for (int i = 0; i < output_tensors.size(); i++) {
        hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release task handle
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");

    return 0;
}

/**
 * @brief Postprocess model outputs into final detection results.
 *
 * Steps:
 * - Dequantize each output tensor to float (current implementation uses S32 dequantization)
 * - Decode bounding boxes and class scores
 * - Apply confidence threshold filtering (config.score_thresh)
 * - Perform class-wise Non-Maximum Suppression (NMS) (config.nms_thresh)
 * - Map bounding boxes back to the original image space (letterbox inverse)
 *
 * @param results [out] Final detection results after postprocessing.
 * @param output_tensors [in] Raw output tensors from inference.
 * @param config [in] Postprocess configuration (anchors/strides/hw_list/thresholds/classes, etc.).
 * @param orig_img_w [in] Width of the original input image.
 * @param orig_img_h [in] Height of the original input image.
 * @param input_w [in] Model input width.
 * @param input_h [in] Model input height.
 */
void post_process(std::vector<Detection>& results,
                  std::vector<hbDNNTensor>& output_tensors,
                  const Yolov5Config& config,
                  int orig_img_w, int orig_img_h,
                  int input_w, int input_h)
{
    std::vector<std::vector<float>> float_outputs;

    // Dequantize each output tensor to float
    for (int i = 0; i < output_tensors.size(); ++i) {
        const auto& tensor = output_tensors[i];
        auto deq_result = dequantizeTensorS32(tensor);
        float_outputs.push_back(std::move(deq_result));
    }

    // Decode all detection layers
    auto decode_all = yolov5_decode_all_layers(
        float_outputs,
        config.hw_list,
        config.strides,
        config.anchors,
        config.score_thresh);

    // Apply class-wise NMS
    results = nms_bboxes(decode_all, config.nms_thresh);

    // Map bounding boxes back to original image coordinates
    scale_letterbox_bboxes_back(results, orig_img_w, orig_img_h, input_w, input_h);
}
