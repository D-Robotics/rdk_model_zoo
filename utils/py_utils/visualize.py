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


"""
visualize: Visualization utilities for model results.

This module provides reusable helpers for rendering model outputs onto images
or display devices, including drawing bounding boxes, masks, contours,
keypoints, and text annotations. It is intended to support debugging,
verification, and demo visualization across multiple samples and runtimes.

Key Features:
    - Draw detection results (boxes, labels, scores) on images or displays.
    - Overlay segmentation results (masks, contours) with configurable styles.
    - Render keypoints and simple geometric annotations for visualization.

Notes:
    - The module focuses on generic visualization building blocks; task- or
      product-specific UI policies should be implemented at the sample level.
    - Visual styles and color mappings may evolve as new tasks and models are
      added.
"""


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
    """Print the top-k classification predictions.

    This function applies softmax to the raw classification logits,
    selects the top-k classes with the highest probabilities, and prints
    their labels and confidence scores.

    Args:
        output: Raw classification logits as a NumPy array with shape
            `(num_classes,)`.
        idx2label: Dictionary mapping class indices to human-readable
            label strings.
        topk: Number of top predictions to display.

    Returns:
        None
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
    """Draw bounding boxes with class names and confidence scores on an image.

    This function draws rectangular bounding boxes on the input image and
    overlays the corresponding class name and confidence score for each
    detected object.

    Args:
        image: Input image as a NumPy array.
        boxes: Bounding boxes with shape `(N, 4)` in `(x1, y1, x2, y2)` format.
        cls_ids: Class indices corresponding to each bounding box.
        scores: Confidence scores for each detection.
        class_names: List of class name strings indexed by class ID.
        colors: List of RGB color tuples used for visualization. Colors are
            cycled if the number of classes exceeds the list length.

    Returns:
        The input image with bounding boxes and labels drawn on it.
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
    """Overlay semi-transparent instance masks on an image.

    This function blends instance segmentation masks onto the input image
    using alpha compositing. Each mask is rendered with a class-specific
    color inside its corresponding bounding box. The input image is modified
    in place.

    Args:
        image: Input image array to draw on. The image is modified in place.
        boxes: Bounding boxes corresponding to each mask with shape `(N, 4)`
            in `(x1, y1, x2, y2)` format.
        masks: List of binary mask arrays, each corresponding to a bounding
            box region.
        cls_ids: List of class indices for each detected instance.
        colors: List of RGB color tuples used for visualization. Colors are
            cycled if the number of classes exceeds the list length.
        alpha: Transparency factor for mask blending. A value of `0` means
            fully transparent, while `1` means fully opaque.

    Returns:
        None
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
    """Draw contour outlines of instance masks on an image.

    This function extracts the external contours from each instance mask
    and draws their outlines on the input image. The contours are shifted
    from local (box-relative) coordinates to global image coordinates.
    The input image is modified in place.

    Args:
        img: Input image array to draw on. The image is modified in place.
        boxes: Bounding boxes corresponding to each mask with shape `(N, 4)`
            in `(x1, y1, x2, y2)` format.
        masks: List of binary mask arrays for each detected instance.
        cls_ids: List of class indices for each instance.
        colors: List of RGB color tuples used for visualization. Colors are
            cycled if the number of classes exceeds the list length.
        thickness: Thickness of the contour lines.

    Returns:
        None
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
    """Convert an RGB tuple to a 32-bit ARGB display color value.

    The output color format is ARGB, where the highest 8 bits represent
    the alpha channel, followed by red, green, and blue channels.

    Args:
        rgb_tuple: A tuple of `(R, G, B)` values, each in the range `[0, 255]`.

    Returns:
        A 32-bit integer representing the color in ARGB format.
    """
    r, g, b = rgb_tuple
    alpha = 255
    return (alpha << 24) | (r << 16) | (g << 8) | b


def draw_detections_on_disp(disp, boxes: np.ndarray, cls_ids: list,
                            scores: list, class_names: list,
                            colors: list, chn: int = 2) -> None:
    """Draw detection boxes and labels on a hardware display.

    This function renders detection results directly onto a hardware display
    device by drawing bounding boxes and overlaying class names with
    confidence scores. The display canvas is cleared before drawing.

    Args:
        disp: Display device object that provides `set_graph_rect` and
            `set_graph_word` methods for drawing graphics and text.
        boxes: Bounding boxes with shape `(N, 4)` in `(x1, y1, x2, y2)` format.
        cls_ids: List of class indices corresponding to each bounding box.
        scores: List of detection confidence scores.
        class_names: List of class name strings indexed by class ID.
        colors: List of RGB color tuples used for visualization. Colors are
            cycled if the number of classes exceeds the list length.
        chn: Display channel index.

    Returns:
        None
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
    """Draw keypoints with confidence scores on an image.

    This function visualizes keypoints by drawing concentric circles at
    each keypoint location and annotating them with their index. Only
    keypoints whose confidence exceeds the given threshold are rendered.
    The input image is modified in place.

    Args:
        image: Input image array to draw on. The image is modified in place.
        kpts_xy: Keypoint coordinates with shape `(N, K, 2)`, where `N` is
            the number of instances and `K` is the number of keypoints.
        kpts_score: Keypoint confidence scores with shape `(N, K, 1)`.
        kpt_conf_thresh: Confidence threshold for displaying keypoints.
        radius_outer: Radius of the outer circle drawn for each keypoint.
        radius_inner: Radius of the inner circle drawn for each keypoint.

    Returns:
        None
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
    """Draw polygon-style bounding boxes on a copy of the image.

    This function draws closed polygon outlines on a copy of the input image.
    Each polygon is defined by a sequence of vertex coordinates.

    Args:
        img: Input image in BGR format.
        bboxes: List of polygon bounding boxes. Each element is a NumPy array
            with shape `(N, 2)` representing polygon vertices.
        color: Polygon color in `(B, G, R)` format.
        thickness: Line thickness used to draw polygon edges.

    Returns:
        A copy of the input image with polygon bounding boxes drawn on it.
    """
    img_copy = img.copy()
    for bbox in bboxes:
        bbox = bbox.astype(int)
        # Draw closed polygon on image
        cv2.polylines(img_copy, [bbox], isClosed=True, color=color, thickness=thickness)
    return img_copy
