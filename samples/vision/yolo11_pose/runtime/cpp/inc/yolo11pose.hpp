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
 * @file yolo11pose.hpp
 * @brief Define a high-level inference wrapper and pipeline interfaces for
 *        the YOLO11-Pose anchor-free pose estimation model.
 *
 * This file provides a structured C++ interface encapsulating the complete
 * YOLO11-Pose inference workflow on D-Robotics platforms, including model
 * configuration, runtime resource management, preprocessing, BPU inference
 * execution, and DFL box decoding with keypoint postprocessing.
 *
 * @see yolo11pose.cpp
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
 * @brief Configuration parameters for YOLO11-Pose preprocessing and postprocessing.
 *
 * Default values cover the recommended typical usage scenario and allow the
 * sample to run without modification.
 */
struct YOLO11PoseConfig
{
    std::vector<int> strides{8, 16, 32};        ///< Feature map strides for P3/P4/P5 heads
    std::vector<int> anchor_sizes{80, 40, 20};  ///< Feature map grid sizes for P3/P4/P5 heads
    int  reg{16};                                ///< Number of DFL regression bins per bbox side
    float score_thresh{0.25f};                   ///< Confidence threshold for filtering detections
    float nms_thresh{0.7f};                      ///< IoU threshold for Non-Maximum Suppression
    float kpt_conf_thresh{0.5f};                 ///< Keypoint visibility confidence threshold
};

/**
 * @class YOLO11Pose
 * @brief Wrapper class for YOLO11-Pose inference using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates the complete pose estimation pipeline, including:
 * - Model loading and initialization (via init())
 * - Input tensor preparation (letterbox + BGR->NV12)
 * - BPU inference execution
 * - Anchor-free DFL decoding and keypoint extraction with NMS
 *
 * @note Not thread-safe. Create one instance per thread for concurrent use.
 */
class YOLO11Pose
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)

    /**
     * @brief Construct a YOLO11Pose instance in an uninitialized state.
     */
    YOLO11Pose();

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
    ~YOLO11Pose();

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
 * @brief Postprocess YOLO11-Pose outputs into final detections and keypoints.
 *
 * Steps:
 * 1) For each of the 3 detection heads, filter class logits by threshold,
 *    decode DFL bounding boxes, and extract keypoint predictions.
 * 2) Concatenate all head results and apply class-wise NMS (keeping keypoints aligned).
 * 3) Rescale boxes and keypoints to original image coordinates (undo letterbox).
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration.
 * @param[in]  orig_img_w     Original image width (pixels).
 * @param[in]  orig_img_h     Original image height (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 * @return Pair of (detections, keypoints_per_detection).
 */
PoseResult post_process(std::vector<hbDNNTensor>& output_tensors,
                        const YOLO11PoseConfig& config,
                        int orig_img_w, int orig_img_h,
                        int input_w, int input_h);
