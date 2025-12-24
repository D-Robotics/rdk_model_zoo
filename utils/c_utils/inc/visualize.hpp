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

#pragma once

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "model_types.hpp"

/**
 * @brief Small, visually distinct color palette (BGR).
 * @note Colors repeat cyclically when class_id exceeds the palette length.
 */
inline std::vector<cv::Scalar> rdk_colors = {
    {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255},
    {49, 210, 207}, {10, 249, 72},  {23, 204, 146}, {134, 219, 61},
    {52, 147, 26},  {187, 212, 0},  {168, 153, 44}, {255, 194, 0},
    {147, 69, 52},  {255, 115, 100},{236, 24, 0},   {255, 56, 132},
    {133, 0, 82},   {255, 56, 203}, {200, 149, 255},{199, 55, 255}
};

/**
 * @brief Pack 8-bit RGB into ARGB8888 (alpha set to 0xFF).
 * @param[in] r  Red channel   [0..255].
 * @param[in] g  Green channel [0..255].
 * @param[in] b  Blue channel  [0..255].
 * @return uint32_t Packed color as 0xAARRGGBB with AA=0xFF.
 */
uint32_t rgb_to_argb8888(uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief Draw detection boxes and class labels on the hardware display overlay.
 *
 * This renders each detection as a rectangle with a text label above it using the
 * sp_display_* overlay APIs. Colors are chosen per-class from the provided palette.
 *
 * @param[in] display_obj  Opaque handle returned by display init/start functions.
 * @param[in] detections   List of detections with xyxy boxes, score, and class_id.
 * @param[in] class_names  List of class names; indexed by detection.class_id.
 * @param[in] colors       BGR color palette; cycled by class_id.
 * @param[in] chn          Display channel/layer index to draw on.
 */
void draw_detections_on_disp(void* display_obj,
                              const std::vector<Detection>& detections,
                              const std::vector<std::string>& class_names,
                              const std::vector<cv::Scalar>& colors,
                              int chn = 2);

/**
 * @brief Draw axis-aligned bounding boxes with class label and score.
 *
 * @param[in,out] image  BGR image to draw on.
 * @param[in]     detections Vector of detections (xyxy, score, class_id).
 * @param[in]     class_names Vector of class names; indexed by class_id.
 * @param[in]     colors Color palette for classes; used cyclically.
 */
void draw_boxes(cv::Mat& image,
                const std::vector<Detection>& detections,
                const std::vector<std::string>& class_names,
                const std::vector<cv::Scalar>& colors);

/**
 * @brief Blend segmentation masks into the image inside each detection's bbox.
 *
 * @param[in,out] image       BGR image to draw on.
 * @param[in]     detections  Detections with xyxy boxes (used as ROI).
 * @param[in]     masks       Binary masks aligned to detections (1=foreground).
 * @param[in]     colors      Color palette for classes.
 * @param[in]     alpha       Alpha blending factor (0=orig, 1=color).
 */
void draw_masks(cv::Mat& image,
                const std::vector<Detection>& detections,
                const std::vector<cv::Mat>& masks,
                const std::vector<cv::Scalar>& colors,
                float alpha = 0.3f);

/**
 * @brief Draw contour lines (polygon outlines) for segmentation masks.
 *
 * @param[in,out] img     Image to draw on.
 * @param[in]     detections Detected boxes; used to shift local contours to image coords.
 * @param[in]     masks   Binary masks (per detection) in local bbox coordinates.
 * @param[in]     colors  Color palette used cyclically by class id.
 * @param[in]     thickness Line thickness in pixels.
 */
void draw_contours(cv::Mat& img,
                   const std::vector<Detection>& detections,
                   const std::vector<cv::Mat>& masks,
                   const std::vector<cv::Scalar>& colors,
                   int thickness = 2);

/**
 * @brief Draw keypoints for pose estimation results.
 *
 * A keypoint is drawn if its score (pre-sigmoid) passes the given threshold after sigmoid.
 * Two concentric circles are used for a simple highlight.
 *
 * @param[in,out] image           Image to draw on.
 * @param[in]     kpts            Keypoints per instance; shape: N x K.
 * @param[in]     kpt_conf_thresh Display threshold applied to sigmoid(score).
 * @param[in]     radius_outer    Radius of the outer filled circle.
 * @param[in]     radius_inner    Radius of the inner filled circle.
 */
void draw_keypoints(cv::Mat& image,
                    const std::vector<std::vector<Keypoint>>& kpts,
                    float kpt_conf_thresh = 0.5f,
                    int radius_outer = 5,
                    int radius_inner = 2);

/**
 * @brief Draw multiple text strings onto an image using FreeType (TrueType) fonts.
 *
 * Text origin for each string is taken from the first vertex of the corresponding box.
 * Coordinates are clamped to the image canvas to avoid out-of-bound drawing.
 *
 * @param[in] img        Background image (BGR).
 * @param[in] texts      Texts to draw (aligned with boxes).
 * @param[in] boxes      Polygon boxes; only boxes[i][0] is used as the anchor point.
 * @param[in] font_path  Path to a .ttf font file.
 * @param[in] font_size  Font size in FreeType units.
 * @param[in] color      Text color (BGR).
 * @param[in] thickness  Stroke thickness.
 * @return cv::Mat A copy of @p img with texts rendered (original is unchanged).
 */
cv::Mat draw_text(cv::Mat img,
                  const std::vector<std::string>& texts,
                  const std::vector<std::vector<cv::Point>>& boxes,
                  const std::string& font_path,
                  int font_size,
                  cv::Scalar color,
                  int thickness);

/**
 * @brief Draw closed polygon boxes on an image.
 *
 * @param[in] img        Input image (BGR).
 * @param[in] bboxes     List of polygons (each is a vector of points).
 * @param[in] color      Line color (BGR).
 * @param[in] thickness  Line thickness.
 * @return cv::Mat A copy of the input image with polygons drawn.
 */
cv::Mat draw_polygon_boxes(const cv::Mat& img,
                           const std::vector<std::vector<cv::Point>>& bboxes,
                           const cv::Scalar& color = cv::Scalar(128, 240, 128),
                           int thickness = 3);

/**
 * @brief Convert a label mask (H x W, int32 class ids) to a color image.
 *
 * @param[in] seg_mask  Per-pixel class ids (CV_32S).
 * @param[in] colors    Color palette; index is the class id.
 * @return cv::Mat BGR colorized image (CV_8UC3).
 */
cv::Mat colorize_mask(const cv::Mat& seg_mask, const std::vector<cv::Scalar>& colors);

/**
 * @brief Print top-k classification results in a friendly format.
 *
 * @param[in] top_k_cls Vector of top-k (id, score) pairs.
 * @param[in] label_map Optional mapping from id to human-readable label.
 */
void print_topk_results(const std::vector<Classification>& top_k_cls,
                               const std::map<int, std::string>& label_map);
