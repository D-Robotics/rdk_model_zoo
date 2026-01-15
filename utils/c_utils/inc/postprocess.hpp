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
 * @file postprocess.hpp
 * @brief Declare postprocessing interfaces for model inference results.
 *
 * This file provides common interfaces for converting raw model outputs
 * into application-level inference results.
 */

#pragma once

#include <cstdint>
#include <array>
#include <utility>
#include <vector>
#include <opencv2/core/mat.hpp>

#include "model_types.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/dnn/hb_dnn.h"

/**
 * @brief Extract top-k classification results from a tensor.
 *
 * @param[in] tensor Input tensor containing classification logits or scores.
 * @param[out] top_k_cls Vector to store the top-k classification results.
 * @param[in] top_k Number of top results to keep.
 */
void get_topk_result(hbDNNTensor& tensor, std::vector<Classification> &top_k_cls, int top_k);

/**
 * @brief Dequantize a quantized value to floating-point.
 *
 * @tparam T Quantized value type (e.g., int8_t, int16_t, int32_t).
 * @param[in] qval Quantized input value.
 * @param[in] channel Channel index for per-channel quantization.
 * @param[in] prop Tensor properties containing quantization parameters.
 * @return Dequantized floating-point value.
 */
template<typename T>
inline float dequant_value(T qval, int channel,
                           const hbDNNTensorProperties& prop)
{
    const float scale = prop.scale.scaleData[channel];
    const int zp = (prop.scale.zeroPointData && prop.scale.zeroPointLen > 0)
                   ? prop.scale.zeroPointData[channel]
                   : 0;
    return (static_cast<int>(qval) - zp) * scale;
}

/**
 * @brief Fully dequantize an S32 quantized tensor into floating-point values.
 *
 * @param[in] tensor Input tensor with quantized S32 data.
 * @return Vector of dequantized float values.
 */
std::vector<float> dequantizeTensorS32(const hbDNNTensor& tensor);

/**
 * @brief Decode YOLOv5 multi-level, anchor-based outputs to detections.
 *
 * @param[in] all_results  Per-level flat outputs; each level is na*h*w*(5+num_classes).
 * @param[in] hw_list      (h,w) for each level.
 * @param[in] strides      Stride for each level.
 * @param[in] all_anchors  Anchors per level: {{aw, ah}, ...}.
 * @param[in] score_thresh Keep boxes with sigmoid(obj)*sigmoid(max_class) >= threshold.
 * @param[in] num_classes  Number of classes.
 * @return std::vector<Detection> Detections as {x1,y1,x2,y2,score,class_id}.
 *
 * @note NMS is not applied here.
 */
std::vector<Detection> yolov5_decode_all_layers(
    const std::vector<std::vector<float>>& all_results,
    const std::vector<std::pair<int, int>>& hw_list,
    const std::vector<int>& strides,
    const std::vector<std::vector<std::array<float, 2>>>& all_anchors,
    float score_thresh,
    int num_classes = 80);

/**
 * @brief Compute Intersection-over-Union (IoU) between two bounding boxes.
 *
 * @param[in] a First detection bounding box.
 * @param[in] b Second detection bounding box.
 * @return IoU value in [0,1].
 */
float iou(const Detection& a, const Detection& b);

/**
 * @brief Perform Non-Maximum Suppression (NMS) on bounding boxes.
 *
 * @param[in] detections Input bounding boxes.
 * @param[in] iou_thresh IoU threshold for suppression (default: 0.45).
 * @return Filtered list of detections after NMS.
 */
std::vector<Detection> nms_bboxes(const std::vector<Detection>& detections,
                                  float iou_thresh = 0.45f);

/**
 * @brief Scale letterbox-adjusted bounding boxes back to original image coordinates.
 *
 * @param[in,out] dets Vector of detections with bbox coordinates to be scaled.
 * @param[in] img_w Original image width.
 * @param[in] img_h Original image height.
 * @param[in] input_w Model input width.
 * @param[in] input_h Model input height.
 */
void scale_letterbox_bboxes_back(std::vector<Detection>& dets,
                                 int img_w, int img_h,
                                 int input_w, int input_h);

/**
 * @brief Scale letterbox-adjusted keypoints back to original image coordinates.
 *
 * @param[in,out] kpts Keypoints for all detections to be scaled.
 * @param[in] img_w Original image width.
 * @param[in] img_h Original image height.
 * @param[in] input_w Model input width.
 * @param[in] input_h Model input height.
 */
void scale_keypoints_back_letterbox(std::vector<std::vector<Keypoint>>& kpts,
                                    int img_w, int img_h,
                                    int input_w, int input_h);
