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
 * @file yolo11.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the YOLO11 anchor-free object detection model.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * YOLO11 inference workflow on D-Robotics platforms, including model
 * configuration, runtime resource management, preprocessing, BPU inference
 * execution, and DFL-based postprocessing.
 *
 * @see yolo11.cpp
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "nn_math.hpp"
#include "visualize.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

/**
 * @brief Configuration parameters for YOLO11 preprocessing and postprocessing.
 *
 * Stores all parameters required for the YOLO11 DFL inference pipeline,
 * including feature map strides, grid sizes, DFL bins, and detection thresholds.
 * Default values cover the recommended typical usage scenario and allow the
 * sample to run without modification.
 */
struct Yolo11Config
{
    std::vector<int> strides{8, 16, 32};        ///< Feature map downsampling strides for P3/P4/P5 heads
    std::vector<int> anchor_sizes{80, 40, 20};  ///< Feature map grid sizes for P3/P4/P5 heads
    int reg{16};                                 ///< Number of DFL regression bins per bounding-box side
    float score_thresh{0.25f};                   ///< Confidence threshold for filtering candidate detections
    float nms_thresh{0.45f};                     ///< IoU threshold used during Non-Maximum Suppression (NMS)
    int num_classes{80};                         ///< Number of object categories predicted by the model
};

/**
 * @class YOLO11
 * @brief Wrapper class for YOLO11 inference using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete inference pipeline, including:
 * - Model loading and initialization (via init())
 * - Input tensor preparation
 * - BPU inference execution
 * - Anchor-free DFL output postprocessing
 *
 * @note Not thread-safe. Create one instance per thread for concurrent use.
 */
class YOLO11
{
public:
    int model_count{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle{nullptr};  ///< Packed DNN handle
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded YOLO11 model
    int32_t input_count{0};                          ///< Number of input tensors required by the model
    int32_t output_count{0};                         ///< Number of output tensors produced by the model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)
    bool inited{false};                              ///< Whether init() has been successfully called

    /**
     * @brief Construct a YOLO11 instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    YOLO11();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * Performs:
     * - Loading the packed model from disk
     * - Selecting the first model handle in the pack
     * - Querying input/output tensor counts and properties
     * - Allocating input/output tensor memory buffers
     * - Caching model input resolution (H, W)
     *
     * @param[in] model_path Path to the quantized *.hbm model file.
     * @retval 0 Success.
     * @retval non-zero Failure (DNN or UCP API error).
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
    ~YOLO11();
};


/**
 * @brief Preprocess an input BGR image into NV12 model input tensors.
 *
 * Performs letterbox resize to the model input dimensions and converts the
 * image from BGR to NV12 format as required by the compiled model.
 *
 * @param[in,out] input_tensors Model input tensors to be filled with NV12 data.
 * @param[in]     img           Input image in BGR format (OpenCV Mat).
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
 * @param[in]     image_format  Input image format string. Only "BGR" is supported.
 *
 * @retval 0        Success.
 * @retval non-zero Unsupported format or conversion failure.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    cv::Mat& img,
                    const int input_w, const int input_h,
                    const std::string& image_format = "BGR");

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
 * @brief Postprocess YOLO11 DFL outputs into final detection results.
 *
 * Steps:
 * 1) Convert probability threshold to logit threshold.
 * 2) For each of the 3 detection scales, decode classification and DFL box tensors.
 * 3) Concatenate all detections, apply NMS, and rescale boxes to original image space.
 *
 * @param[out] results        Final detection list after NMS and coordinate rescaling.
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration (strides, anchor_sizes, thresholds).
 * @param[in]  orig_img_w     Width of the original input image (pixels).
 * @param[in]  orig_img_h     Height of the original input image (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 */
void post_process(std::vector<Detection>& results,
                  std::vector<hbDNNTensor>& output_tensors,
                  const Yolo11Config& config,
                  int orig_img_w, int orig_img_h,
                  int input_w, int input_h);
