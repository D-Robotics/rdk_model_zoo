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

/**
 * @brief Select top-K classification results by probability.
 *
 * @param[in]  results
 *     Full classification results.
 *
 * @param[out] topk_results
 *     Output top-K results, sorted in descending order by probability.
 *     The vector will be cleared before use.
 *
 * @param[in]  k
 *     Number of top results to select.
 *
 */
void get_topk_result(const std::vector<Classification>& results,
                     std::vector<Classification>& topk_results,
                     size_t k);

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
    if(prop.quantiType==SCALE) {
        const float scale = prop.scale.scaleData[channel];
        const int zp = (prop.scale.zeroPointData && prop.scale.zeroPointLen > 0)
                    ? prop.scale.zeroPointData[channel] : 0;
        return (static_cast<int>(qval) - zp) * scale;
    }else{
        return qval;
    }
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

/**
 * @brief Compute per-pixel argmax over the channel axis of an NHWC int32 tensor.
 *
 * Iterates over all spatial positions (H × W) and finds the channel index
 * with the maximum int32 value. No dequantization is required since argmax
 * is order-preserving under per-channel scale quantization.
 *
 * Typical use case: semantic segmentation models that output NHWC S32 logits
 * (e.g., UnetMobileNet), where the argmax directly yields per-pixel class IDs.
 *
 * @param[in] tensor Input BPU tensor with NHWC layout and S32 data type (N=1).
 * @return cv::Mat   Single-channel class ID map of type CV_32S, shape (H, W).
 *
 * @note Only N=1 batches are supported.
 */
cv::Mat argmax_nhwc_s32(const hbDNNTensor& tensor);

/**
 * @brief Dequantize an NHWC int16 tensor with per-N (axis-0) scale to float32.
 *
 * The tensor uses int16 data with a per-batch-item (axis-0) scale and optional
 * zero-point. Output is a flattened float vector in NHWC order.
 *
 * Typical use case: prototype feature tensors in instance segmentation models
 * (e.g., YOLO11-Seg protos).
 *
 * @param[in] tensor Input BPU tensor with NHWC layout and S16 data type.
 * @return std::vector<float> Flattened NHWC array of size N*H*W*C.
 */
std::vector<float> dequantize_s16_axis0(const hbDNNTensor& tensor);

/**
 * @brief Decode per-instance segmentation masks from prototype features and MCES coefficients.
 *
 * For each detection, crops the prototype feature map to the scaled bounding box
 * region, computes the linear combination proto × mces, applies sigmoid, and
 * binarizes the result.
 *
 * @param[in] detections  Detections whose bounding boxes define the crop regions
 *                        (coordinates in model input scale).
 * @param[in] mces        MCES coefficient vectors, one per detection; shape (N, mces_num).
 * @param[in] protos      Prototype feature map, flattened NHWC (mask_h × mask_w × mces_num).
 * @param[in] input_w     Model input width (pixels).
 * @param[in] input_h     Model input height (pixels).
 * @param[in] mask_w      Prototype feature map width.
 * @param[in] mask_h      Prototype feature map height.
 * @param[in] mask_thresh Threshold applied after sigmoid for binary mask generation.
 * @return std::vector<cv::Mat> Binary masks (CV_8UC1, 0/1), one per detection,
 *         each sized to its proto-scale crop region.
 */
std::vector<cv::Mat> decode_masks(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& mces,
    const std::vector<float>& protos,
    int input_w, int input_h,
    int mask_w, int mask_h,
    float mask_thresh = 0.5f);

/**
 * @brief Resize instance segmentation masks to their bounding boxes in original image space.
 *
 * Each mask (cropped at proto scale) is resized to match the pixel dimensions of
 * its corresponding bounding box in the original image. Optionally applies
 * morphological opening to smooth mask edges.
 *
 * @param[in] masks       Per-instance masks at proto scale (CV_8UC1), aligned with @p detections.
 * @param[in] detections  Detections in original image coordinates; bounding boxes define
 *                        the target size for each mask.
 * @param[in] img_w       Original image width (pixels).
 * @param[in] img_h       Original image height (pixels).
 * @param[in] do_morph    Whether to apply morphological opening to smooth mask edges.
 * @return std::vector<cv::Mat> Resized binary masks (CV_8UC1), one per detection,
 *         each sized (box_h × box_w).
 */
std::vector<cv::Mat> resize_masks_to_boxes(
    const std::vector<cv::Mat>& masks,
    const std::vector<Detection>& detections,
    int img_w, int img_h,
    bool do_morph = true);

/**
 * @brief Convert dilated polygons to minimum-area bounding boxes with area filtering.
 *
 * For each polygon, computes the minimum-area enclosing rectangle and converts
 * it to a 4-point box. Polygons whose area is below @p min_area are discarded.
 *
 * @param[in] dilated_polys  Input polygon list (each a vector of cv::Point).
 * @param[in] min_area       Minimum contour area threshold (pixels²); smaller contours are skipped.
 * @return std::vector<std::vector<cv::Point>>  4-point clockwise bounding boxes.
 */
std::vector<std::vector<cv::Point>> get_bounding_boxes(
    const std::vector<std::vector<cv::Point>>& dilated_polys,
    float min_area);

/**
 * @brief Crop and rectify a rotated text region from an image via perspective warp.
 *
 * Computes the minimum-area rectangle of the input polygon, applies a perspective
 * transform to produce an axis-aligned crop, and rotates 90° clockwise when
 * the detected angle indicates a tall (portrait) orientation (angle >= 45°).
 *
 * @param[in] img  Source image (BGR, CV_8UC3).
 * @param[in] box  Polygon enclosing the text region (>= 4 points, pixel coordinates).
 * @return cv::Mat Rectified crop (CV_8UC3), or an empty Mat if @p box has fewer than 4 points
 *         or the computed rectangle is degenerate.
 */
cv::Mat crop_and_rotate_image(const cv::Mat& img,
                              const std::vector<cv::Point>& box);
