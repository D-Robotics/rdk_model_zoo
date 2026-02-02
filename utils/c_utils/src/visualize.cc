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
 * @file visualize.cpp
 * @brief Provide visualization utilities for rendering model inference results.
 *
 * This file implements common drawing and formatting helpers used to present
 * inference outputs (e.g., classification results, detections, masks, keypoints,
 * and text overlays) on images and, when enabled, on hardware display overlays.
 */

#include "visualize.hpp"
#include "nn_math.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>

#include "sp_display.h"


/**
 * @brief Print top-k classification results in a friendly format.
 *
 * @param[in] top_k_cls Vector of top-k (id, score) pairs.
 * @param[in] label_map Optional mapping from id to human-readable label.
 */
void print_topk_results(const std::vector<Classification>& top_k_cls,
                               const std::map<int, std::string>& label_map)
{
    for (size_t i = 0; i < top_k_cls.size(); ++i) {
        const auto& item = top_k_cls[i];
        auto it = label_map.find(item.id);
        const char* label = (it != label_map.end()) ? it->second.c_str() : "Unknown";
        std::cout << "TOP " << i
           << ": label=" << label
           << ", prob=" << item.score
           << '\n';
    }
}

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
                const std::vector<cv::Scalar>& colors)
{
    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.bbox[0]); // top-left x
        int y1 = static_cast<int>(det.bbox[1]); // top-left y
        int x2 = static_cast<int>(det.bbox[2]); // bottom-right x
        int y2 = static_cast<int>(det.bbox[3]); // bottom-right y
        int class_id = det.class_id;

        // Pick color by class id
        cv::Scalar color = colors[class_id % colors.size()];
        // Compose label "name score"
        std::string label = class_names[class_id] + " " + cv::format("%.2f", det.score);

        // Draw rectangle
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Draw label text above (or at) the top edge
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int y_text = std::max(y1 - 5, label_size.height);

        cv::putText(image, label, cv::Point(x1, y_text),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

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
                float alpha)
{
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        // Guard: valid bbox and corresponding mask
        if (x2 <= x1 || y2 <= y1) continue;
        if (i >= masks.size()) continue;

        const cv::Mat& mask = masks[i];
        if (mask.empty()) continue;

        // Clamp bbox to image region
        x1 = std::clamp(x1, 0, image.cols);
        x2 = std::clamp(x2, 0, image.cols);
        y1 = std::clamp(y1, 0, image.rows);
        y2 = std::clamp(y2, 0, image.rows);
        if (x2 <= x1 || y2 <= y1) continue;

        // ROI view over the image for in-place blending
        cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // Ensure mask matches ROI size
        cv::Mat mask_resized;
        if (mask.size() != roi.size()) {
            cv::resize(mask, mask_resized, roi.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            mask_resized = mask;
        }

        // Convert mask to 8-bit for countNonZero / indexing
        cv::Mat mask_bool;
        mask_resized.convertTo(mask_bool, CV_8U);
        if (cv::countNonZero(mask_bool) == 0) continue;

        // Single-color patch for blending
        cv::Mat color_patch(roi.size(), roi.type(),
                            colors[det.class_id % colors.size()]);

        // Manual alpha blending on masked pixels
        for (int y = 0; y < roi.rows; ++y) {
            for (int x = 0; x < roi.cols; ++x) {
                if (mask_bool.at<uint8_t>(y, x)) {
                    cv::Vec3b& pixel = roi.at<cv::Vec3b>(y, x);
                    const cv::Vec3b color_pixel = color_patch.at<cv::Vec3b>(y, x);
                    // Blend each channel
                    for (int c = 0; c < 3; ++c) {
                        pixel[c] = static_cast<uint8_t>(
                            (1 - alpha) * pixel[c] + alpha * color_pixel[c]);
                    }
                }
            }
        }
    }
}

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
                   int thickness)
{
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        // Ensure mask exists
        if (i >= masks.size()) continue;
        const cv::Mat& mask = masks[i];
        if (mask.empty()) continue;

        // Extract contours from the local mask (ROI space)
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        // Shift local contour points by top-left ROI corner to image coords
        for (auto& contour : contours) {
            for (auto& pt : contour) {
                pt.x += x1;
                pt.y += y1;
            }
        }

        // Draw the polygon outlines
        cv::Scalar color = colors[det.class_id % colors.size()];
        cv::polylines(img, contours, true, color, thickness);
    }
}

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
void draw_keypoints(
    cv::Mat& image,
    const std::vector<std::vector<Keypoint>>& kpts,
    float kpt_conf_thresh,
    int radius_outer,
    int radius_inner)
{
    // Convert threshold on sigmoid(score) to raw-logit space for a cheap compare:
    // sigmoid(s) > t  <=>  s > -log(1/t - 1)
    const float kpt_conf_inverse = -std::log(1.0f / kpt_conf_thresh - 1.0f);

    for (size_t i = 0; i < kpts.size(); ++i) {
        const auto& instance = kpts[i];
        for (size_t j = 0; j < instance.size(); ++j) {
            const Keypoint& kp = instance[j];
            // Skip low-confidence points
            if (kp.score < kpt_conf_inverse) continue;

            // Integer pixel coordinates
            const int x = static_cast<int>(kp.x);
            const int y = static_cast<int>(kp.y);

            // Draw two concentric circles for better visibility
            cv::circle(image, {x, y}, radius_outer, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            cv::circle(image, {x, y}, radius_inner, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

            // Overlay keypoint index (shadow + bright)
            const std::string id_str = std::to_string(static_cast<int>(j));
            cv::putText(image, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            cv::putText(image, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }
}

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
                  int thickness)
{
    // Work on a clone to keep the original image untouched
    cv::Mat output = img.clone();

    // Create FreeType renderer
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    if (ft2.empty()) {
        std::cerr << "[Error] FreeType module not available in this OpenCV build.\n";
        return output;
    }
    ft2->loadFontData(font_path, 0);  // load TTF face

    for (size_t i = 0; i < texts.size() && i < boxes.size(); ++i) {
        const std::string& text = texts[i];

        // Skip empty strings
        if (text.empty()) {
            continue;
        }

        // Use first vertex of the polygon as text origin
        cv::Point org = boxes[i][0];

        // Clamp origin into the canvas (leave room for ascent)
        if (org.x < 0)            org.x = 0;
        if (org.y < font_size)    org.y = font_size;
        if (org.x >= img.cols)    org.x = img.cols - 1;
        if (org.y >= img.rows)    org.y = img.rows - 1;

        // Render UTF-8 text with anti-aliasing
        ft2->putText(output, text, org, font_size, color, thickness, cv::LINE_AA, true);
    }
    return output;
}

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
                           const cv::Scalar& color,
                           int thickness)
{
    // Work on a clone to keep the original image intact
    cv::Mat img_copy = img.clone();

    for (const auto& bbox : bboxes) {
        if (bbox.empty()) continue;

        // Draw a closed polygon (last point connected to first)
        cv::polylines(img_copy, bbox, true, color, thickness);
    }

    return img_copy;
}

/**
 * @brief Convert a label mask (H x W, int32 class ids) to a color image.
 *
 * @param[in] seg_mask  Per-pixel class ids (CV_32S).
 * @param[in] colors    Color palette; index is the class id.
 * @return cv::Mat BGR colorized image (CV_8UC3).
 */
cv::Mat colorize_mask(const cv::Mat& seg_mask,
                      const std::vector<cv::Scalar>& colors)
{
    CV_Assert(seg_mask.type() == CV_32S);      // ensure integer ids
    CV_Assert(!colors.empty());

    const int H = seg_mask.rows;
    const int W = seg_mask.cols;

    cv::Mat parsing_img(H, W, CV_8UC3);        // output BGR image

    // Map each class id to a palette color (out-of-range → black)
    for (int y = 0; y < H; ++y) {
        const int*    mask_row = seg_mask.ptr<int>(y);
        cv::Vec3b*    out_row  = parsing_img.ptr<cv::Vec3b>(y);

        for (int x = 0; x < W; ++x) {
            const int cls = mask_row[x];

            if (cls >= 0 && cls < static_cast<int>(colors.size())) {
                const cv::Scalar& c = colors[cls];  // B,G,R,(A)
                out_row[x] = cv::Vec3b(
                    static_cast<uchar>(c[0]),
                    static_cast<uchar>(c[1]),
                    static_cast<uchar>(c[2])
                );
            } else {
                out_row[x] = cv::Vec3b(0, 0, 0);    // unknown → black
            }
        }
    }
    return parsing_img;
}

/**
 * @brief Pack 8-bit RGB into ARGB8888 (alpha set to 0xFF).
 * @param[in] r  Red channel   [0..255].
 * @param[in] g  Green channel [0..255].
 * @param[in] b  Blue channel  [0..255].
 * @return uint32_t Packed color as 0xAARRGGBB with AA=0xFF.
 */
uint32_t rgb_to_argb8888(uint8_t r, uint8_t g, uint8_t b)
{
    // 0xFF for full opacity, then R,G,B in big-endian byte order
    return (0xFFu << 24) | (static_cast<uint32_t>(r) << 16)
                         | (static_cast<uint32_t>(g) << 8)
                         | (static_cast<uint32_t>(b));
}

#ifdef ENABLE_SP_DISPLAY

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
                             int chn)
{
    // Clear canvas
    sp_display_draw_rect(display_obj, 0, 0, 0, 0, chn, 1, 1, 3);
    sp_display_draw_string(display_obj, 0, 0,  const_cast<char*>(""), chn, 1, 1, 16);

    for (const auto& det : detections) {
        // Clamp/convert bbox coordinates to int (display API expects ints)
        const int x1 = static_cast<int>(det.bbox[0]);   // left
        const int y1 = static_cast<int>(det.bbox[1]);   // top
        const int x2 = static_cast<int>(det.bbox[2]);   // right
        const int y2 = static_cast<int>(det.bbox[3]);   // bottom
        const int class_id = det.class_id;

        // Compose label text, e.g. "person 0.94"
        std::ostringstream oss;
        oss << class_names[class_id] << ' ' << std::fixed << std::setprecision(2) << det.score;
        const std::string label = oss.str();

        // Convert BGR (cv::Scalar) to ARGB8888 expected by display (with full alpha)
        const cv::Scalar bgr = colors[class_id % colors.size()]; // color per class
        const uint32_t argb8888_color = rgb_to_argb8888(
            static_cast<uint8_t>(bgr[2]),  // R
            static_cast<uint8_t>(bgr[1]),  // G
            static_cast<uint8_t>(bgr[0])   // B
        );

        // Draw rectangle; flush=0 (accumulate on overlay), line_width=3
        sp_display_draw_rect(display_obj, x1, y1, x2, y2, chn, 0, argb8888_color, 3);

        // Draw text slightly above the top-left of the box
        const int y_text = std::max(y1 - 20, 0);
        sp_display_draw_string(display_obj, x1, y_text,
                               const_cast<char*>(label.c_str()),
                               chn, 0, argb8888_color, 16);
    }
}

#endif
