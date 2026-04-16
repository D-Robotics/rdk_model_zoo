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
YOLO26 Detection Inference Module.

This module implements the YOLO26 object detection pipeline on BPU,
including pre-processing, forward execution, and post-processing.

Key Features:
    - Optimized for RDK X5 single-input NV12 models.
    - Supports standard YOLO26 anchor-free box decoding.
    - Provides a complete `predict()` pipeline for Python samples.

Typical Usage:
    >>> from yolo26_det import YOLO26Config, YOLO26Detect
    >>> config = YOLO26Config(model_path="path/to/yolo26_detect.bin")
    >>> model = YOLO26Detect(config)
    >>> results = model.predict(image)
"""

import os
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path to import shared utilities.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26")


@dataclass
class YOLO26Config:
    """
    Configuration for YOLO26 detection inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        classes_num (int): Number of detection classes.
        score_thres (float): Confidence threshold.
        nms_thres (float): IoU threshold for NMS.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        strides (List[int]): Detection head strides.
    """
    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.7
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26Detect:
    """
    YOLO26 detection wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLO26Config):
        """
        Initialize the model, load metadata, and precompute decoding grids.

        Args:
            config (YOLO26Config): Configuration object containing model path and params.
        """
        self.cfg = config
        self.conf_raw = -np.log(1.0 / self.cfg.score_thres - 1.0)

        t0 = time.time()
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        if input_shape[1] == 3:
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        else:
            self.input_h = input_shape[1]
            self.input_w = input_shape[2]

        self.grids = {}
        for stride in self.cfg.strides:
            grid_h, grid_w = self.input_h // stride, self.input_w // stride
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            self.grids[stride] = grid.reshape(-1, 2).astype(np.float32) + 0.5

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        Set BPU scheduling parameters like priority and core affinity.

        Args:
            priority (Optional[int]): Scheduling priority (0-255).
            bpu_cores (Optional[list]): BPU core indexes to run inference.
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                    img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert a BGR image to packed NV12 input for hbm_runtime.

        Args:
            img (np.ndarray): Input image in BGR format.
            resize_type (Optional[int]): Override default resize strategy.
            image_format (str): Input image format.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared input tensors for hbm_runtime.run().
        """
        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)

        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)
        return {
            self.model_name: {
                self.input_names[0]: packed_nv12,
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Execute inference on BPU using hbm_runtime.

        Args:
            input_tensor (Dict[str, Dict[str, np.ndarray]]): Prepared input tensors.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Raw output tensors from the runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info(f"\033[1;31mForward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert raw model outputs to final detection results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.
            score_thres (Optional[float]): Override confidence threshold.
            nms_thres (Optional[float]): Override NMS threshold.

        Returns:
            A tuple containing:
                - bounding boxes in `(x1, y1, x2, y2)` format
                - confidence scores
                - class ids
        """
        t0 = time.time()
        score_thres = self.cfg.score_thres if score_thres is None else score_thres
        nms_thres = self.cfg.nms_thres if nms_thres is None else nms_thres
        conf_raw = -np.log(1.0 / score_thres - 1.0)

        raw_outputs = outputs[self.model_name]
        dets = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 2
            cls_data = raw_outputs[self.output_names[base_idx]].reshape(-1, self.cfg.classes_num)
            box_data = raw_outputs[self.output_names[base_idx + 1]].reshape(-1, 4)

            valid_score, valid_cls_id, valid_indices = post_utils.filter_classification(cls_data, conf_raw)
            if valid_indices.size == 0:
                continue

            grid = self.grids[stride][valid_indices]
            valid_box = box_data[valid_indices]
            xyxy = post_utils.decode_ltrb_boxes(grid, valid_box, stride)

            dets.extend(np.hstack([xyxy, valid_score[:, None], valid_cls_id[:, None]]))

        final_boxes = []
        final_scores = []
        final_cls_ids = []
        if dets:
            dets = np.array(dets, dtype=np.float32)
            for cls_id in np.unique(dets[:, 5]):
                cls_dets = dets[dets[:, 5] == cls_id]
                indices = post_utils.NMS(cls_dets[:, :4], cls_dets[:, 4], cls_dets[:, 5], nms_thres)
                if not indices:
                    continue

                kept = cls_dets[indices]
                boxes = post_utils.scale_coords_back(
                    kept[:, :4].copy(),
                    ori_img_w,
                    ori_img_h,
                    self.input_w,
                    self.input_h,
                    self.cfg.resize_type,
                )
                final_boxes.append(boxes.astype(np.float32))
                final_scores.append(kept[:, 4].astype(np.float32))
                final_cls_ids.append(kept[:, 5].astype(np.int32))

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        if not final_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.concatenate(final_boxes, axis=0),
            np.concatenate(final_scores, axis=0),
            np.concatenate(final_cls_ids, axis=0),
        )

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        High-level interface for one-click inference.

        Args:
            img (np.ndarray): Input image.

        Returns:
            Detection results in `(boxes, scores, cls_ids)` format.
        """
        ori_img_h, ori_img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide functional-style calling capability.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            Detection results from `predict()` in `(boxes, scores, cls_ids)` format.
        """
        return self.predict(img)
