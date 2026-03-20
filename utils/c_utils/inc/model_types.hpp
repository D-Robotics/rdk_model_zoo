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
 * @file model_types.hpp
 * @brief Define common data structures for representing model inference results.
 *
 * This file provides shared result data types used across inference,
 * postprocessing, and visualization modules.
 */

#pragma once

#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Classification item.
 */
typedef struct {
    float probability;  /**< Classification probability */
    int   class_id;      /**< Class index */
} Classification;


/**
 * @brief Generic detection with an axis-aligned bounding box.
 */
struct Detection {
    float bbox[4];           ///< Bounding box coordinates: x1, y1, x2, y2 (pixels, inclusive)
    float score;             ///< Confidence score (e.g., obj * cls)
    int   class_id;          ///< Argmax class index
};

/**
 * @brief A single keypoint.
 */
struct Keypoint {
    float x;                 ///< X coordinate (pixels)
    float y;                 ///< Y coordinate (pixels)
    float score;             ///< Raw score/logit (apply sigmoid before thresholding)
};

/**
 * @brief Semantic segmentation result.
 *
 * Represents the output of a semantic segmentation model as a per-pixel
 * class ID map at the original image resolution. Visualization (colorization,
 * alpha blending) should be performed in the calling code after receiving
 * this result.
 *
 * @note class_ids has type CV_32S (int32_t per pixel).
 */
struct SegmentationMask {
    cv::Mat class_ids;  ///< Per-pixel class ID map, CV_32S, shape (orig_h × orig_w)
};

/**
 * @brief Instance segmentation result.
 *
 * Represents the output of an instance segmentation model as a collection
 * of detected objects each paired with a per-instance binary mask.
 * The detections and masks vectors are index-aligned: detections[i] and
 * masks[i] describe the same instance. Visualization (mask overlay, contour
 * drawing) should be performed in the calling code after receiving this result.
 *
 * @note Each mask has type CV_8UC1 with values 0 or 1, sized to its
 *       corresponding bounding box (box_h × box_w).
 */
struct InstanceSegResult {
    std::vector<Detection> detections;  ///< Detected objects (bbox, score, class_id), aligned with masks
    std::vector<cv::Mat>   masks;       ///< Per-instance binary masks (CV_8UC1, 0/1), aligned with detections
};

/**
 * @brief Pose estimation result.
 *
 * Represents the output of a pose estimation model as a collection of detected
 * persons, each paired with a set of keypoints. The detections and keypoints
 * vectors are index-aligned: detections[i] and keypoints[i] describe the same
 * instance. Visualization (skeleton drawing, keypoint markers) should be
 * performed in the calling code after receiving this result.
 *
 * @note Each keypoints[i] is a vector of 17 COCO keypoints (x, y, score).
 *       Keypoint scores are raw logits; apply sigmoid before thresholding.
 */
struct PoseResult {
    std::vector<Detection>              detections;  ///< Detected persons (bbox, score, class_id), aligned with keypoints
    std::vector<std::vector<Keypoint>>  keypoints;   ///< Per-instance keypoints (17 COCO kpts each), aligned with detections
};

/**
 * @brief OCR (text detection + recognition) result.
 *
 * Represents the combined output of a complete OCR pipeline: detected text
 * region bounding boxes and their recognized text strings. The boxes and texts
 * vectors are index-aligned: boxes[i] and texts[i] describe the same region.
 * Visualization (polygon overlay, text rendering) should be performed in the
 * calling code after receiving this result.
 *
 * @note Each element of boxes is a 4-point clockwise polygon in pixel space.
 *       texts contains UTF-8 encoded strings.
 */
struct OCRResult {
    std::vector<std::vector<cv::Point>> boxes;  ///< Detected text region polygons (4-point boxes), aligned with texts
    std::vector<std::string>            texts;  ///< Recognized UTF-8 text strings, aligned with boxes
};
