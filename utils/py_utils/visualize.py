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

"""
visualize: Visualization utilities for model results.

This module provides reusable helpers for rendering model outputs onto images
and logging detailed results. It includes generic drawing functions and 
specific wrappers for the YOLO26 model series.
"""

import cv2
import numpy as np
import logging
import math

# Use a consistent logger
logger = logging.getLogger("YOLO26")

# List of predefined RGB color tuples used for bounding box visualization.
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

# Standard human pose skeleton structure (COCO format)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]


# ==============================================================================
# 1. Generic Drawing Functions (Numpy Array Based)
# ==============================================================================

def draw_boxes(image: np.ndarray, boxes: np.ndarray, cls_ids: np.ndarray,
               scores: np.ndarray, class_names: list, colors: list) -> np.ndarray:
    """Draw bounding boxes with class names and scores."""
    for box, cls_id, score in zip(boxes, cls_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name} {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1, max(y1 - 5, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=1)
    return image


def draw_masks(image: np.ndarray, boxes: np.ndarray, masks: list,
               cls_ids: list, colors: list, alpha: float = 0.3) -> None:
    """Overlay semi-transparent instance masks."""
    for class_id, box, mask in zip(cls_ids, boxes, masks):
        x1, y1, x2, y2 = map(int, box)
        if mask.size == 0 or x2 <= x1 or y2 <= y1:
            continue

        region = image[y1:y2, x1:x2]
        mask_area = mask.astype(bool)
        if not np.any(mask_area):
            continue

        color = colors[class_id % len(colors)]
        color_patch = np.zeros(region.shape, dtype=np.uint8)
        color_patch[:] = color

        region[mask_area] = (
            (1 - alpha) * region[mask_area] + alpha * color_patch[mask_area]
        ).astype(np.uint8)


def draw_rotated_boxes(img: np.ndarray, rrects: list, ids: list, scores: list,
                       class_names: list, colors: list, thickness: int = 2) -> np.ndarray:
    """Draw rotated bounding boxes (OBB)."""
    for rrect, cid, score in zip(rrects, ids, scores):
        cx, cy, w, h, a = rrect
        pts = cv2.boxPoints(((cx, cy), (w, h), a * 180 / math.pi)).astype(np.int32)
        color = colors[cid % len(colors)]
        name = class_names[cid] if cid < len(class_names) else str(cid)
        label = f"{name}: {score:.2f}"
        
        cv2.drawContours(img, [pts], 0, color, thickness)
        lx, ly = pts[0]
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (lx, ly - lh - 5), (lx + lw, ly), color, cv2.FILLED)
        cv2.putText(img, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def draw_pose(img: np.ndarray, boxes: np.ndarray, kpts: np.ndarray,
              skeleton: list = COCO_SKELETON, kpt_conf_thres: float = 0.5,
              scores: np.ndarray = None, class_ids: np.ndarray = None,
              colors: list = rdk_colors) -> np.ndarray:
    """Draw pose estimation results."""
    if scores is None: scores = np.ones(len(boxes))
    if class_ids is None: class_ids = np.zeros(len(boxes), dtype=int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = colors[class_ids[i] % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Pose boxes usually green
        
        label = f"person: {scores[i]:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        for x, y, conf in kpts[i]:
            if conf >= kpt_conf_thres:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        for sk in skeleton:
            idx1, idx2 = sk[0] - 1, sk[1] - 1
            if idx1 < len(kpts[i]) and idx2 < len(kpts[i]):
                p1, p2 = kpts[i][idx1], kpts[i][idx2]
                if p1[2] >= kpt_conf_thres and p2[2] >= kpt_conf_thres:
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)
    return img


def draw_classification(img: np.ndarray, results: list, labels: dict,
                        pos: tuple = (10, 30), scale: float = 0.8,
                        color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw Top-K results on image."""
    for i, (cid, score) in enumerate(results):
        label_str = labels.get(cid, str(cid)) if isinstance(labels, dict) else labels[cid] if cid < len(labels) else str(cid)
        text = f"Rank {i+1}: Class {cid} ({label_str}) | Score: {score:.4f}"
        cv2.putText(img, text, (pos[0], pos[1] + i * 30), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img


# ==============================================================================
# 2. YOLO26 Specific Wrappers (Result Dict/Tuple Based + Logging)
# ==============================================================================

def draw_detect_yolo26(img: np.ndarray, results: list, class_names: list,
                       colors: list = rdk_colors) -> np.ndarray:
    """Draw YOLO26 detection results and log details."""
    if not results: return img
    
    # Unpack and Log
    boxes, ids, scores = [], [], []
    for r in results:
        cid, score, x1, y1, x2, y2 = int(r[0]), r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])
        name = class_names[cid] if cid < len(class_names) else str(cid)
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {score:.2f}")
        ids.append(cid); scores.append(score); boxes.append([x1, y1, x2, y2])
    
    return draw_boxes(img, np.array(boxes), np.array(ids), np.array(scores), class_names, colors)


def draw_seg_yolo26(img: np.ndarray, results: list, class_names: list,
                    colors: list = rdk_colors) -> np.ndarray:
    """Draw YOLO26 segmentation results and log details."""
    if not results: return img

    boxes, ids, scores, masks = [], [], [], []
    for r in results:
        x1, y1, x2, y2 = map(int, r['box'])
        cid, score = int(r['id']), r['score']
        name = class_names[cid] if cid < len(class_names) else str(cid)
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {score:.2f}")
        
        boxes.append([x1, y1, x2, y2]); ids.append(cid); scores.append(score)
        
        # Binary mask preparation
        if r['mask'].size > 0:
            prob_mask = cv2.resize(r['mask'], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
            masks.append((prob_mask > 0.5).astype(np.uint8))
        else:
            masks.append(np.zeros((0, 0), dtype=np.uint8))

    draw_boxes(img, np.array(boxes), np.array(ids), np.array(scores), class_names, colors)
    draw_masks(img, np.array(boxes), masks, ids, colors)
    return img


def draw_obb_yolo26(img: np.ndarray, results: list, class_names: list,
                    colors: list = rdk_colors) -> np.ndarray:
    """Draw YOLO26 OBB results and log details."""
    if not results: return img

    rrects, ids, scores = [], [], []
    for r in results:
        cid, score = int(r['id']), r['score']
        name = class_names[cid] if cid < len(class_names) else str(cid)
        logger.info(f"{name}: {score:.2f} | Angle: {r['rrect'][4] * 180 / math.pi:.1f} deg")
        rrects.append(r['rrect']); ids.append(cid); scores.append(score)

    return draw_rotated_boxes(img, rrects, ids, scores, class_names, colors)


def draw_pose_yolo26(img: np.ndarray, results: list, skeleton: list = COCO_SKELETON,
                     kpt_conf_thres: float = 0.5) -> np.ndarray:
    """Draw YOLO26 pose results and log details."""
    if not results: return img

    boxes, kpts, scores = [], [], []
    for r in results:
        x1, y1, x2, y2 = map(int, r['box'])
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> person: {r['score']:.2f}")
        boxes.append(r['box']); kpts.append(r['kpts']); scores.append(r['score'])
    
    return draw_pose(img, np.array(boxes), np.array(kpts), skeleton, kpt_conf_thres, scores=np.array(scores))


def draw_cls_yolo26(img: np.ndarray, results: list, labels: dict) -> np.ndarray:
    """Log YOLO26 classification results (returns image unchanged)."""
    for i, (cid, score) in enumerate(results):
        label_str = labels.get(cid, str(cid)) if isinstance(labels, dict) else labels[cid] if cid < len(labels) else str(cid)
        logger.info(f"Rank {i+1}: Class {cid} ({label_str}) | Score: {score:.4f}")
    return img