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
 * @file lanenet.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the LaneNet lane detection model.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * LaneNet inference workflow on D-Robotics S100 platforms, including model
 * configuration, runtime resource management, float32 preprocessing,
 * BPU inference execution, and instance/binary segmentation postprocessing.
 *
 * @note This model only supports RDK S100 platform.
 *       RDK S600 is NOT supported; the model was compiled for S100 BPU (nash-e).
 *
 * @see lanenet.cpp
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "preprocess.hpp"

/**
 * @brief Configuration parameters for LaneNet preprocessing.
 *
 * Stores parameters required for the LaneNet inference pipeline.
 * Default values cover the recommended typical usage and allow the sample
 * to run without modification.
 *
 * @note This model only supports RDK S100.
 */
struct LaneNetConfig
{
    // No model-specific hyperparameters required beyond model path.
    // All preprocessing constants (ImageNet mean/std) are fixed in the pipeline.
};

/**
 * @class LaneNet
 * @brief Wrapper class for LaneNet lane detection using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete inference pipeline, including:
 * - Model loading and initialization (via init())
 * - Float32 NCHW input tensor preparation (resize + ImageNet normalization)
 * - BPU inference execution
 * - Instance segmentation and binary segmentation mask output
 *
 * @note This model only supports RDK S100 platform. Not thread-safe.
 */
class LaneNet
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)

    /**
     * @brief Construct a LaneNet instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    LaneNet();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * Performs:
     * - Loading the packed model from disk
     * - Selecting the first model handle in the pack
     * - Querying input/output tensor counts and properties
     * - Allocating input/output tensor memory buffers
     * - Caching model input resolution (H, W) from NCHW layout
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
    ~LaneNet();

private:
    int model_count_{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};  ///< Packed DNN handle
    int32_t input_count_{0};                          ///< Number of input tensors
    int32_t output_count_{0};                         ///< Number of output tensors
    bool inited_{false};                              ///< Whether init() has been called
};


/**
 * @brief Preprocess an input BGR image into float32 NCHW model input tensors.
 *
 * Performs:
 * 1. Direct resize (INTER_AREA) to model input dimensions (H x W)
 * 2. BGR -> RGB conversion
 * 3. Per-channel normalization with ImageNet mean/std ([0,1] -> standardized float32)
 * 4. CHW layout write into model input tensor memory
 *
 * @param[in,out] input_tensors Model input tensors to be filled with float32 CHW data.
 * @param[in]     img           Input image in BGR format (OpenCV Mat).
 * @param[in]     input_w       Model input width in pixels (512).
 * @param[in]     input_h       Model input height in pixels (256).
 * @param[in]     image_format  Input image format string. Only "BGR" is supported.
 *
 * @retval 0        Success.
 * @retval -1       Unsupported format or conversion failure.
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
 * @brief Postprocess LaneNet output tensors into instance and binary segmentation masks.
 *
 * Decodes two model outputs:
 * - instance_seg_logits (output index 0): F32 NCHW [1, 3, H, W], each channel scaled to [0, 255]
 *   and packed into a 3-channel uint8 image (HWC).
 * - binary_seg_pred     (output index 1): S64 NCHW [1, 1, H, W], binary values {0, 1}
 *   multiplied by 255 to produce a grayscale uint8 image.
 *
 * @param[out] instance_pred  Instance segmentation result (CV_8UC3, H x W).
 * @param[out] binary_pred    Binary lane segmentation result (CV_8UC1, H x W).
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 */
void post_process(cv::Mat& instance_pred,
                  cv::Mat& binary_pred,
                  std::vector<hbDNNTensor>& output_tensors,
                  int input_w, int input_h);
