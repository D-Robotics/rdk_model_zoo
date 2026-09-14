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
 * @file yoloe11seg.hpp
 * @brief Define interfaces for YOLOe11-Seg open-vocabulary instance segmentation.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * YOLOe11-Seg inference workflow on D-Robotics platforms, including model
 * configuration, runtime resource management, preprocessing, BPU inference
 * execution, and DFL + prototype-based mask postprocessing.
 *
 * @note S600 platform is NOT supported. A compile-time check is enforced in main.cpp.
 *
 * @see yoloe11seg.cpp
 */

#pragma once

#include <utility>
#include <unordered_map>
#include "file_io.hpp"
#include "runtime.hpp"
#include "nn_math.hpp"
#include "visualize.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

/**
 * @brief Configuration parameters for YOLOe11-Seg preprocessing and postprocessing.
 *
 * Default values cover the recommended typical usage scenario for the open-vocabulary
 * model (4585 classes) and allow the sample to run without modification.
 */
struct YoloE11SegConfig
{
    std::vector<int> strides{8, 16, 32};        ///< Feature map strides for P3/P4/P5 heads
    std::vector<int> anchor_sizes{80, 40, 20};  ///< Feature map grid sizes for P3/P4/P5 heads
    int  reg{16};                                ///< Number of DFL regression bins per bbox side
    int  mces_num{32};                           ///< Dimension of the MCES mask coefficient vector
    int  proto_size{160};                        ///< Prototype feature map spatial size (W and H)
    float score_thresh{0.25f};                   ///< Confidence threshold for filtering detections
    float nms_thresh{0.7f};                      ///< IoU threshold for Non-Maximum Suppression
    float mask_thresh{0.5f};                     ///< Sigmoid threshold for binary mask generation
    bool  do_morph{false};                       ///< Apply morphological opening to mask edges (off by default for open-vocab)
};

/**
 * @class YOLO11ESeg
 * @brief Wrapper class for YOLOe11-Seg open-vocabulary inference using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete instance segmentation pipeline, including:
 * - Model loading and initialization (via init())
 * - Input tensor preparation (letterbox + BGR->NV12)
 * - BPU inference execution
 * - Anchor-free DFL decoding, MCES extraction, NMS, mask generation and resizing
 *
 * @note Not thread-safe. Create one instance per thread for concurrent use.
 * @note S600 platform is NOT supported.
 */
class YOLO11ESeg
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)

    /**
     * @brief Construct a YOLO11ESeg instance in an uninitialized state.
     */
    YOLO11ESeg();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * @param[in] model_path Path to the quantized *.hbm model file.
     * @retval 0        Success.
     * @retval non-zero DNN or UCP API error.
     *
     * @note Calling init() more than once is not allowed.
     */
    int32_t init(const char* model_path);

    /**
     * @brief Destructor. Releases all tensor memory and DNN model resources.
     */
    ~YOLO11ESeg();

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
 * Performs letterbox resize to preserve aspect ratio and converts the image
 * from BGR to NV12 format.
 *
 * @param[in,out] input_tensors Model input tensors to be filled.
 * @param[in]     img           Input image in BGR format.
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
 * @param[in]     image_format  Input format string (only "BGR" supported).
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
 * @param[in,out] output_tensors Output tensors filled by the runtime.
 * @param[in]     input_tensors  Prepared input tensors.
 * @param[in]     dnn_handle     DNN model handle.
 * @param[in]     sched_param    Optional UCP scheduling parameters.
 *
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

/**
 * @brief Postprocess YOLOe11-Seg outputs into final detections and instance masks.
 *
 * Steps:
 * 1) For each of the 3 detection heads, filter class logits by threshold (using
 *    OpenMP parallelism to handle 4585 classes efficiently), decode DFL bounding
 *    boxes, and extract MCES mask coefficients.
 * 2) Concatenate all head results and apply class-wise NMS (keeping MCES aligned).
 * 3) Dequantize the prototype feature tensor (int16, per-N scale).
 * 4) Decode per-instance binary masks via linear combination of protos and MCES.
 * 5) Rescale boxes to original image coordinates (undo letterbox).
 * 6) Resize each mask to its bounding box in original image space.
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration.
 * @param[in]  orig_img_w     Original image width (pixels).
 * @param[in]  orig_img_h     Original image height (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 * @return InstanceSegResult  Detections and per-instance masks (index-aligned).
 */
InstanceSegResult post_process(std::vector<hbDNNTensor>& output_tensors,
                                const YoloE11SegConfig& config,
                                int orig_img_w, int orig_img_h,
                                int input_w, int input_h);
