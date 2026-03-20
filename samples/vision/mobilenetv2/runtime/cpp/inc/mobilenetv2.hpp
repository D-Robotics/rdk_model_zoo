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

#pragma once

/**
 * @file mobilenetv2.hpp
 * @brief Public interfaces for MobileNetV2 classification sample.
 *
 * This header declares the runtime wrapper, configuration structure,
 * and standalone preprocessing, inference, and postprocessing helpers
 * for running a MobileNetV2 classification model.
 *
 * @note This file only defines interfaces. Implementations are provided
 *       in corresponding source files.
 */

#include "file_io.hpp"
#include "runtime.hpp"
#include "visualize.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

/**
 * @brief Configuration parameters for MobileNetV2 preprocessing.
 *
 * Defines image resize behavior applied during input preprocessing.
 */
struct MobileNetV2Config
{
    /**
     * @brief Image resize mode used during preprocessing.
     *
     * - 0: Stretch resize to model input resolution.
     * - 1: Keep aspect ratio with padding (letterbox).
     */
    int resize_mode{1};  ///< Resize mode selector (default: letterbox).
};

/**
 * @brief MobileNetV2 inference runtime wrapper.
 *
 * Provides model loading and runtime management for a MobileNetV2 HBM model using
 * the D-Robotics DNN API. The class owns the model handles and manages the
 * input/output tensor descriptors and buffers required for inference.
 *
 * @note One instance represents one independent runtime context.
 */
class MobileNetV2
{
    private:
        int model_count_;                             ///< Number of models loaded in the packed handle
        hbDNNPackedHandle_t packed_dnn_handle_;       ///< Packed DNN handle for managing multiple models
        int32_t input_count_;                         ///< Number of input tensors for the model
        int32_t output_count_;                        ///< Number of output tensors for the model
        bool inited_;                                 ///< init flag

    public:
        hbDNNHandle_t dnn_handle;                    ///< DNN handle for the specific MobileNetV2 model
        std::vector<hbDNNTensor> input_tensors;      ///< Vector storing input tensors
        std::vector<hbDNNTensor> output_tensors;     ///< Vector storing output tensors
        int input_h;                                 ///< Model input image height
        int input_w;                                 ///< Model input image width

        /**
        * @brief Construct a MobileNetV2 instance.
        *
        * The object is created in an uninitialized state.
        * Call init() to load the model, query tensor properties,
        * and allocate input/output tensor buffers.
        *
        * @note The constructor does not allocate any device or host resources.
        */
        MobileNetV2();

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
        * @brief Destructor that releases allocated resources and model handles.
        */
        ~MobileNetV2();
};

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
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

/**
 * @brief Postprocess model outputs into final classification results.
 *
 * Reads probabilities directly from the output tensor via
 * `tensor_to_cls_results()` without applying softmax, since the MobileNetV2
 * model output node already contains post-softmax probability values.
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
                  const MobileNetV2Config& config, int top_k);
