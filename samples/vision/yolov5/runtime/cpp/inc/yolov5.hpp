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
 * @file yolov5x.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the YOLOv5x object detection model.
 *
 *        This file provides a structured C++ interface that encapsulates the
 *        complete YOLOv5x inference workflow on Horizon Robotics platforms,
 *        including model configuration, runtime resource management,
 *        preprocessing, inference execution, and postprocessing. It serves
 *        as an integration layer that connects low-level runtime utilities
 *        with application-level usage in a clear and reusable manner.
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "visualize.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

/**
 * @class YOLOv5x
 * @brief Runtime resource wrapper for YOLOv5x model using HB-DNN / UCP APIs.
 *
 * Responsibilities:
 * - Load a packed *.hbm model and select a model handle
 * - Query input/output tensor properties
 * - Allocate and own tensor buffers (hbUCPSysMem)
 */
struct Yolov5Config
{
    // Feature map strides for each detection scale
    // Default values correspond to YOLOv5 (P3, P4, P5)
    std::vector<int> strides{8, 16, 32};

    // Anchor definitions per scale:
    // anchors[scale][anchor_id] = {anchor_width, anchor_height}
    std::vector<std::vector<std::array<float, 2>>> anchors{
        { {10, 13}, {16, 30}, {33, 23} },
        { {30, 61}, {62, 45}, {59, 119} },
        { {116, 90}, {156, 198}, {373, 326} }
    };

    // Feature map spatial resolution (height, width) for each detection scale
    // Must match the output tensor layout of the compiled model
    std::vector<std::pair<int, int>> hw_list = {
        {84, 84},
        {42, 42},
        {21, 21}
    };

    // Confidence threshold for filtering candidate detections
    float score_thresh{0.25f};

    // IoU threshold used during Non-Maximum Suppression (NMS)
    float nms_thresh{0.45f};

    // Number of object categories predicted by the model
    int num_classes{80};

    // Image resize mode used during preprocessing
    // 0 = stretch resize
    // 1 = keep aspect ratio with padding (letterbox)
    int resize_mode{1};
};

/**
 * @class YOLOv5x
 * @brief Wrapper class for YOLOv5x inference using Horizon Robotics DNN APIs.
 *
 * This class encapsulates the complete inference pipeline, including:
 * - Model loading and initialization
 * - Input preprocessing
 * - BPU inference execution
 * - Output postprocessing (decode, thresholding, NMS)
 */
class YOLOv5x
{
public:
    // Number of models contained in the packed DNN handle
    int model_count{0};

    // Packed DNN handle (may contain multiple models)
    hbDNNPackedHandle_t packed_dnn_handle{nullptr};

    // Handle of the YOLOv5x model used for inference
    hbDNNHandle_t dnn_handle{nullptr};

    // Number of input tensors required by the model
    int32_t input_count{0};

    // Number of output tensors produced by the model
    int32_t output_count{0};

    // Input tensor descriptors and memory
    std::vector<hbDNNTensor> input_tensors;

    // Output tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;

    // Model input height (pixels)
    int input_h{0};

    // Model input width (pixels)
    int input_w{0};

    bool inited{false};

    /**
    * @brief Construct a YOLOv5x instance.
    *
    * The object is created in an uninitialized state. Call init() to load the
    * model, query tensor properties, and allocate tensor buffers.
    */
    YOLOv5x();

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
    int32_t init(const char* model_path);

    /**
    * @brief Destructor.
    *
    * Releases allocated tensor memory and DNN resources.
    * Safe to call even if init() failed partially.
    */
    ~YOLOv5x();
};


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
                    const std::string& image_format = "BGR");

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
int32_t infer(std::vector<hbDNNTensor> & output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

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
                  int input_w, int input_h);
