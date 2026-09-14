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
 * @file unetmobilenet.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the UnetMobileNet semantic segmentation model.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * UnetMobileNet inference workflow on D-Robotics platforms, including model
 * configuration, runtime resource management, preprocessing, BPU inference
 * execution, and argmax-based postprocessing.
 *
 * @see unetmobilenet.cpp
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "visualize.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

/**
 * @brief Configuration parameters for UnetMobileNet preprocessing and postprocessing.
 *
 * Stores parameters required for the UnetMobileNet inference pipeline.
 * Default values cover the recommended typical usage and allow the sample
 * to run without modification.
 */
struct UnetMobileNetConfig
{
    int    num_classes{19};    ///< Number of Cityscapes semantic classes
    double alpha_f{0.75};      ///< Alpha blending factor: 0.0 = mask only, 1.0 = original only
};

/**
 * @class UnetMobileNet
 * @brief Wrapper class for UnetMobileNet inference using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete inference pipeline, including:
 * - Model loading and initialization (via init())
 * - Input tensor preparation (direct resize + BGR->NV12)
 * - BPU inference execution
 * - Per-pixel argmax postprocessing with segmentation mask output
 *
 * Public data members (dnn_handle, input_tensors, output_tensors, input_h,
 * input_w) are exposed for use by the free pipeline functions pre_process(),
 * infer(), and post_process(). All internal lifecycle state is private.
 *
 * @note Not thread-safe. Create one instance per thread for concurrent use.
 */
class UnetMobileNet
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)

    /**
     * @brief Construct a UnetMobileNet instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    UnetMobileNet();

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
    ~UnetMobileNet();

private:
    int model_count_{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};  ///< Packed DNN handle
    int32_t input_count_{0};                          ///< Number of input tensors
    int32_t output_count_{0};                         ///< Number of output tensors
    bool inited_{false};                              ///< Whether init() has been called
};


/**
 * @brief Preprocess an input BGR image into NV12 model input tensors.
 *
 * Performs direct resize (INTER_AREA) to the model input dimensions and
 * converts the image from BGR to NV12 format.
 *
 * @param[in,out] input_tensors Model input tensors to be filled with NV12 data.
 * @param[in]     img           Input image in BGR format (OpenCV Mat).
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
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
 * @brief Postprocess UnetMobileNet output tensors into a per-pixel class ID mask.
 *
 * Applies argmax over the channel axis of the int32 output tensor, then
 * resizes the result back to the original image dimensions.
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  orig_img_w     Width of the original input image (pixels).
 * @param[in]  orig_img_h     Height of the original input image (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 * @return SegmentationMask   Segmentation result with class_ids of shape
 *                            (orig_img_h × orig_img_w), type CV_32S.
 */
SegmentationMask post_process(std::vector<hbDNNTensor>& output_tensors,
                              int orig_img_w, int orig_img_h,
                              int input_w, int input_h);
