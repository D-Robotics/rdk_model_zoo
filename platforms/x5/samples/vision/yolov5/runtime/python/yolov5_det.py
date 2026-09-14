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
YOLOv5 Detection Inference Module.

This module implements the YOLOv5 object detection pipeline on BPU,
including pre-processing, forward execution, and post-processing.

Key Features:
    - Optimized for RDK X5 single-input NV12 `.bin` models.
    - Keeps the original YOLOv5 anchor-based detection protocol.
    - Provides a complete `predict()` pipeline for Python samples.

Typical Usage:
    >>> from yolov5_det import YOLOv5Config, YOLOv5Detect
    >>> config = YOLOv5Config(model_path="path/to/yolov5.bin")
    >>> model = YOLOv5Detect(config)
    >>> results = model.predict(image)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


logger = logging.getLogger("YOLOv5")


@dataclass
class YOLOv5Config:
    """
    Configuration for YOLOv5 detection inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        classes_num (int): Number of detection classes.
        score_thres (float): Confidence threshold.
        nms_thres (float): IoU threshold for NMS.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        anchors (List[int]): Anchor list used by YOLOv5 heads.
        strides (List[int]): Detection head strides.
    """

    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.45
    resize_type: int = 0
    anchors: List[int] = field(
        default_factory=lambda: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    )
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLOv5Detect:
    """
    YOLOv5 detection wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLOv5Config):
        """
        Initialize the model, load metadata, and precompute grids and anchors.

        Args:
            config (YOLOv5Config): Configuration object containing model path and params.
        """

        self.cfg = config

        t0 = time.time()
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]

        anchors = np.array(self.cfg.anchors, dtype=np.float32).reshape(3, -1, 2)
        self.grids: Dict[int, np.ndarray] = {}
        self.anchors: Dict[int, np.ndarray] = {}

        for i, stride in enumerate(self.cfg.strides):
            grid_h = self.input_h // stride
            grid_w = self.input_w // stride
            grid = np.stack(
                [
                    np.tile(np.linspace(0.5, grid_w - 0.5, grid_w), reps=grid_h),
                    np.repeat(np.arange(0.5, grid_h + 0.5, 1), grid_w),
                ],
                axis=0,
            ).transpose(1, 0)
            self.grids[stride] = np.hstack([grid, grid, grid]).reshape(-1, 2).astype(np.float32)
            self.anchors[stride] = np.tile(anchors[i], (grid_h * grid_w, 1)).astype(np.float32)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """
        Set BPU scheduling parameters like priority and core affinity.

        Args:
            priority (Optional[int]): Scheduling priority (0-255).
            bpu_cores (Optional[List[int]]): BPU core indexes to run inference.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
        image_format: str = "BGR",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert a BGR image to packed NV12 input for hbm_runtime.

        Args:
            image (np.ndarray): Input image in BGR format.
            resize_type (Optional[int]): Override default resize strategy.
            image_format (str): Input image format.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared input tensors for hbm_runtime.run().
        """

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type
        resized = pre_utils.resized_image(image, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)
        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return {
            self.model_name: {
                self.input_names[0]: packed_nv12,
            }
        }

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Execute inference on BPU using hbm_runtime.

        Args:
            inputs (Dict[str, Dict[str, np.ndarray]]): Prepared input tensors.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Raw output tensors from the runtime.
        """

        t0 = time.time()
        outputs = self.model.run(inputs)
        logger.info(f"\033[1;31mForward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        ori_img_w: int,
        ori_img_h: int,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> List[Tuple[int, float, int, int, int, int]]:
        """
        Convert raw model outputs to final detection results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.
            score_thres (Optional[float]): Override confidence threshold.
            nms_thres (Optional[float]): Override NMS threshold.

        Returns:
            List[Tuple[int, float, int, int, int, int]]: Detection results in
            `(class_id, score, x1, y1, x2, y2)` format.
        """

        t0 = time.time()
        score_thres = self.cfg.score_thres if score_thres is None else score_thres
        nms_thres = self.cfg.nms_thres if nms_thres is None else nms_thres
        raw_outputs = outputs[self.model_name]

        boxes_all = []
        scores_all = []
        cls_ids_all = []
        for i, stride in enumerate(self.cfg.strides):
            pred = raw_outputs[self.output_names[i]].reshape(-1, 5 + self.cfg.classes_num)
            obj_raw = pred[:, 4]
            cls_raw = pred[:, 5:]
            cls_max_raw = np.max(cls_raw, axis=1)
            scores = post_utils.sigmoid(obj_raw) * post_utils.sigmoid(cls_max_raw)
            valid_indices = np.flatnonzero(scores >= score_thres)
            if valid_indices.size == 0:
                continue

            cls_ids = np.argmax(cls_raw[valid_indices], axis=1)
            score = scores[valid_indices]
            dxywh = post_utils.sigmoid(pred[valid_indices, :4])
            grid = self.grids[stride][valid_indices]
            anchors = self.anchors[stride][valid_indices]

            xy = (dxywh[:, 0:2] * 2.0 + grid - 1.0) * stride
            wh = (dxywh[:, 2:4] * 2.0) ** 2 * anchors
            xyxy = np.hstack([xy - wh * 0.5, xy + wh * 0.5])
            boxes_all.append(xyxy)
            scores_all.append(score)
            cls_ids_all.append(cls_ids)

        final_res = []
        if boxes_all:
            boxes = np.concatenate(boxes_all, axis=0).astype(np.float32)
            scores = np.concatenate(scores_all, axis=0).astype(np.float32)
            cls_ids = np.concatenate(cls_ids_all, axis=0).astype(np.int32)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_thres, nms_thres)
            if len(indices) > 0:
                indices = np.array(indices).reshape(-1)
                kept_boxes = post_utils.scale_coords_back(
                    boxes[indices].copy(),
                    ori_img_w,
                    ori_img_h,
                    self.input_w,
                    self.input_h,
                    self.cfg.resize_type,
                )
                kept_scores = scores[indices]
                kept_cls_ids = cls_ids[indices]
                for box, score, cls_id in zip(kept_boxes, kept_scores, kept_cls_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    final_res.append((int(cls_id), float(score), x1, y1, x2, y2))

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res

    def predict(self, image: np.ndarray) -> List[Tuple[int, float, int, int, int, int]]:
        """
        High-level interface for one-click inference.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            List[Tuple[int, float, int, int, int, int]]: Detection results.
        """

        ori_img_h, ori_img_w = image.shape[:2]
        input_tensors = self.pre_process(image)
        outputs = self.forward(input_tensors)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, image: np.ndarray) -> List[Tuple[int, float, int, int, int, int]]:
        """
        Provide functional-style calling capability.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            List[Tuple[int, float, int, int, int, int]]: Detection results from `predict()`.
        """

        return self.predict(image)
