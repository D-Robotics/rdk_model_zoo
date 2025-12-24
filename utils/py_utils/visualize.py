# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa: E501

import cv2
import numpy as np

# List of predefined RGB color tuples used for bounding box visualization.
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def print_topk_predictions(output: np.ndarray,
                           idx2label: dict,
                           topk: int = 5) -> None:
    """
    @brief Print top-k classification predictions.
    @details Uses softmax to compute probability and selects top-k.
    @param output Raw logits as NumPy array (shape: [num_classes]).
    @param idx2label Dictionary mapping class indices to labels.
    @param topk Number of top predictions to display.
    @return None
    """
    # Softmax with stability adjustment
    exp_logits = np.exp(output - np.max(output))
    probabilities = exp_logits / np.sum(exp_logits)

    # Top-k indices
    topk_idx = np.argsort(probabilities)[-topk:][::-1]
    topk_prob = probabilities[topk_idx]

    print(f"Top-{topk} Predictions:")
    for i in range(topk):
        idx = topk_idx[i]
        prob = topk_prob[i]
        label = idx2label[idx] if idx2label and idx in idx2label else f"Class {idx}"
        print(f"{label}: {prob:.4f}")


def draw_boxes(image: np.ndarray, boxes: np.ndarray, cls_ids: np.ndarray,
               scores: np.ndarray, class_names: list, colors: list) -> np.ndarray:
    """
    @brief Draw bounding boxes with class names and scores on the image.
    @param image Input image as a NumPy array.
    @param boxes Bounding boxes as a NumPy array of shape (N, 4), format: [x1, y1, x2, y2].
    @param cls_ids List or array of class indices corresponding to boxes.
    @param scores List or array of confidence scores for each detection.
    @param class_names List of class name strings.
    @param colors List of RGB color tuples for each class.
    @return Image with drawn boxes and labels.
    """
    for box, cls_id, score in zip(boxes, cls_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        label = f"{class_names[cls_id]} {score:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # Draw class label and score
        cv2.putText(image, label, (x1, max(y1 - 5, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=1)

    return image


def draw_masks(image: np.ndarray, boxes: np.ndarray, masks: list,
               cls_ids: list, colors: list, alpha: float = 0.3) -> None:
    """
    @brief Overlay semi-transparent instance masks on the image.
    @param image Input image to draw on (modified in-place).
    @param boxes Bounding boxes corresponding to masks, shape: (N, 4).
    @param masks List of binary masks, each with shape matching box region.
    @param cls_ids List of class indices for each instance.
    @param colors List of RGB color tuples for each class.
    @param alpha Transparency level for the masks (0: transparent, 1: opaque).
    @return None
    """
    for class_id, box, mask in zip(cls_ids, boxes, masks):
        x1, y1, x2, y2 = map(int, box)
        if mask.size == 0 or x2 <= x1 or y2 <= y1:
            continue

        region = image[y1:y2, x1:x2]  # Crop region from image
        mask_area = mask.astype(bool)  # Convert to boolean mask

        if not np.any(mask_area):
            continue

        # Generate a solid color patch
        color = colors[(class_id - 1) % len(colors)]
        color_patch = np.empty(region.shape, dtype=np.uint8)
        color_patch[:, :] = color

        # Blend mask with image
        region[mask_area] = (
            (1 - alpha) * region[mask_area] + alpha * color_patch[mask_area]
        ).astype(np.uint8)


def draw_contours(img: np.ndarray, boxes: np.ndarray, masks: list,
                  cls_ids: list, colors: list, thickness: int = 2) -> None:
    """
    @brief Draw contour outlines of instance masks on the image.
    @param img Input image to draw on (modified in-place).
    @param boxes Bounding boxes for each mask, shape: (N, 4).
    @param masks List of binary masks for each instance.
    @param cls_ids List of class indices for each instance.
    @param colors List of RGB color tuples.
    @param thickness Thickness of contour lines.
    @return None
    """
    for class_id, box, mask in zip(cls_ids, boxes, masks):
        x1, y1, x2, y2 = map(int, box)
        if mask.size == 0:
            continue

        # Extract external contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Merge all contours and shift to global coordinates
        merged_points = np.vstack([c for c in contours])
        merged_points[:, 0, 0] += x1
        merged_points[:, 0, 1] += y1

        # Draw the contour line on the image
        cv2.polylines(img, [merged_points], isClosed=True,
                      color=colors[(class_id - 1) % len(colors)],
                      thickness=thickness)


def rgb_to_disp_color(rgb_tuple: tuple) -> int:
    """
    @brief Convert RGB tuple to 32-bit ARGB display color format.
    @details Format is ARGB: alpha in high 8 bits, followed by R, G, B.
    @param rgb_tuple Tuple of (R, G, B) values.
    @return 32-bit ARGB integer color value.
    """
    r, g, b = rgb_tuple
    alpha = 255
    return (alpha << 24) | (r << 16) | (g << 8) | b


def draw_detections_on_disp(disp, boxes: np.ndarray, cls_ids: list,
                            scores: list, class_names: list,
                            colors: list, chn: int = 2) -> None:
    """
    @brief Draw detection boxes and labels on a hardware display.
    @param disp Display device object with `set_graph_rect` and `set_graph_word` methods.
    @param boxes Array of bounding boxes (N, 4).
    @param cls_ids List of class indices.
    @param scores List of detection confidence scores.
    @param class_names List of class name strings.
    @param colors List of RGB color tuples.
    @param chn Display channel index.
    @return None
    """
    # Clear canvas
    disp.set_graph_rect(0, 0, 0, 0, 2, 1, 0, 3)
    disp.set_graph_word(0, 0, "", chn, 1, 0, 16)

    for box, cls_id, score in zip(boxes, cls_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]} {score:.2f}"
        color = rgb_to_disp_color(colors[cls_id % len(colors)])

        # Draw bounding box on display
        disp.set_graph_rect(x1, y1, x2, y2, 2, 0, color, 3)
        # Draw class name and confidence
        disp.set_graph_word(x1, max(y1 - 20, 0), label, chn, 0, color, 16)


def draw_keypoints(image: np.ndarray, kpts_xy: np.ndarray,
                   kpts_score: np.ndarray, kpt_conf_thresh: float = 0.5,
                   radius_outer: int = 5, radius_inner: int = 2) -> None:
    """
    @brief Draw keypoints with confidence scores on an image.
    @param image Input/output image in-place modification.
    @param kpts_xy Keypoints coordinates, shape (N, K, 2).
    @param kpts_score Keypoints confidence scores, shape (N, K, 1).
    @param kpt_conf_thresh Confidence threshold to show keypoints.
    @param radius_outer Outer circle radius.
    @param radius_inner Inner circle radius.
    @return None
    """
    # Convert threshold to logit space (same as sigmoid(score) > threshold)
    kpt_conf_inverse = -np.log(1 / kpt_conf_thresh - 1)

    for instance_xy, instance_score in zip(kpts_xy, kpts_score):
        for j in range(instance_xy.shape[0]):
            if instance_score[j, 0] < kpt_conf_inverse:
                continue

            x, y = int(instance_xy[j, 0]), int(instance_xy[j, 1])

            # Draw outer and inner circles
            cv2.circle(image, (x, y), radius_outer, (0, 0, 255), -1)
            cv2.circle(image, (x, y), radius_inner, (0, 255, 255), -1)

            # Draw index number twice for bold outline effect
            cv2.putText(image, f"{j}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"{j}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_polygon_boxes(img: np.ndarray, bboxes: list,
                       color: tuple = (128, 240, 128),
                       thickness: int = 3) -> np.ndarray:
    """
    @brief Draw polygon-style bounding boxes on a copy of the image.
    @param img Input image (BGR format).
    @param bboxes List of polygon boxes, each is an ndarray of shape (N, 2).
    @param color Polygon color (B, G, R).
    @param thickness Line thickness.
    @return Image with drawn polygons.
    """
    img_copy = img.copy()
    for bbox in bboxes:
        bbox = bbox.astype(int)
        # Draw closed polygon on image
        cv2.polylines(img_copy, [bbox], isClosed=True, color=color, thickness=thickness)
    return img_copy
