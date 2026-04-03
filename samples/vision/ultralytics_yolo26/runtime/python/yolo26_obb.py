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
YOLO26 Oriented Bounding Box Inference Module.

This module implements the YOLO26 OBB pipeline on BPU,
including pre-processing, forward execution, rotated box decoding,
and rotated NMS.

Key Features:
    - Optimized for RDK X5 single-input NV12 models.
    - Supports YOLO26 rotated box decoding and angle regularization.
    - Maps OBB results back to the original image coordinate system.

Typical Usage:
    >>> from yolo26_obb import YOLO26OBBConfig, YOLO26OBB
    >>> config = YOLO26OBBConfig(model_path="path/to/yolo26_obb.bin")
    >>> model = YOLO26OBB(config)
    >>> results = model.predict(image)
"""

import os
import sys
import time
import math
import logging
import cv2
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add project root to sys.path to import shared utilities.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_OBB")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid activation element-wise.

    Args:
        x (np.ndarray): Input tensor in logit space.

    Returns:
        np.ndarray: Tensor converted to probability space.
    """
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class YOLO26OBBConfig:
    """
    Configuration for YOLO26 OBB inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        classes_num (int): Number of OBB classes.
        score_thres (float): Confidence threshold.
        nms_thres (float): IoU threshold for rotated NMS.
        angle_sign (float): Angle sign used during decoding.
        angle_offset (float): Angle offset in degrees.
        regularize (bool): Whether to regularize the width-height ordering.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        strides (List[int]): Detection head strides.
    """
    model_path: str
    classes_num: int = 15
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26OBB:
    """
    YOLO26 OBB wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLO26OBBConfig):
        """
        Initialize the model, load metadata, and precompute decoding grids.

        Args:
            config (YOLO26OBBConfig): Configuration object containing model path and params.
        """
        self.cfg = config
        self.conf_raw = -np.log(1.0 / self.cfg.score_thres - 1.0)
        self.angle_offset_rad = self.cfg.angle_offset * math.pi / 180.0

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
            priority (Optional[int]): Scheduling priority in range [0, 255].
            bpu_cores (Optional[list]): BPU core indexes used for inference.
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
            image_format (str): Input image format expected by the wrapper.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared runtime input tensors.
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

    def _rotated_iou(self, a, b) -> float:
        """
        Compute IoU between two rotated rectangles.

        Args:
            a: First rotated rectangle in `(cx, cy, w, h, angle_rad)` format.
            b: Second rotated rectangle in `(cx, cy, w, h, angle_rad)` format.

        Returns:
            float: IoU value between the two rotated rectangles.
        """
        rect1 = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), float(a[4] * 180.0 / math.pi))
        rect2 = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4] * 180.0 / math.pi))
        try:
            int_ret, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
        except Exception:
            return 0.0
        if int_ret <= 0 or inter_pts is None:
            return 0.0
        inter_area = cv2.contourArea(inter_pts)
        union = a[2] * a[3] + b[2] * b[3] - inter_area
        return 0.0 if union <= 0 else inter_area / union

    def _nms_rotated(self, rrects, scores, cids, iou_thresh):
        """
        Perform class-wise rotated NMS.

        Args:
            rrects: Rotated rectangles in `(cx, cy, w, h, angle_rad)` format.
            scores: Confidence scores for each rotated rectangle.
            cids: Class IDs corresponding to each rotated rectangle.
            iou_thresh: IoU threshold used to suppress overlapping boxes.

        Returns:
            List[int]: Indices of the rotated rectangles kept after NMS.
        """
        keep = []
        scores = np.asarray(scores, dtype=np.float32)
        cids = np.asarray(cids, dtype=np.int32)
        for cid in np.unique(cids):
            idx = np.where(cids == cid)[0]
            order = idx[np.argsort(scores[idx])[::-1]]
            while order.size > 0:
                current = order[0]
                keep.append(current)
                remaining = []
                for other in order[1:]:
                    if self._rotated_iou(rrects[current], rrects[other]) < iou_thresh:
                        remaining.append(other)
                order = np.array(remaining, dtype=np.int32)
        return keep

    def _scale_rrect_to_original(self,
                                 rrect: List[float],
                                 ori_img_w: int,
                                 ori_img_h: int) -> List[float]:
        """
        Map a rotated rectangle from input scale back to original image scale.

        Args:
            rrect (List[float]): Rotated rectangle on model input scale.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.

        Returns:
            List[float]: Rotated rectangle mapped to original image coordinates.
        """
        cx, cy, w, h, a = rrect
        if self.cfg.resize_type == 0:
            scale_x = ori_img_w / self.input_w
            scale_y = ori_img_h / self.input_h
            cx *= scale_x
            cy *= scale_y
            w *= scale_x
            h *= scale_y
        else:
            scale = min(self.input_w / ori_img_w, self.input_h / ori_img_h)
            pad_w = (self.input_w - ori_img_w * scale) / 2
            pad_h = (self.input_h - ori_img_h * scale) / 2
            cx = (cx - pad_w) / scale
            cy = (cy - pad_h) / scale
            w /= scale
            h /= scale

        cx = float(np.clip(cx, 0, ori_img_w))
        cy = float(np.clip(cy, 0, ori_img_h))
        w = float(np.clip(w, 0, ori_img_w))
        h = float(np.clip(h, 0, ori_img_h))
        return [cx, cy, w, h, a]

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int) -> List[Dict]:
        """
        Convert raw outputs to final rotated box results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.

        Returns:
            List[Dict]: OBB results containing rotated rectangles, scores, and
            class IDs in original image coordinates.
        """
        t0 = time.time()
        raw_outputs = outputs[self.model_name]

        rrects = []
        scores = []
        cids = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_data = raw_outputs[self.output_names[base_idx]].reshape(-1, self.cfg.classes_num)
            box_data = raw_outputs[self.output_names[base_idx + 1]].reshape(-1, 4)
            angle_data = raw_outputs[self.output_names[base_idx + 2]].reshape(-1, 1)

            v_scores, v_ids, valid_indices = post_utils.filter_classification(cls_data, self.conf_raw)
            if valid_indices.size == 0:
                continue

            v_box = np.abs(box_data[valid_indices])
            v_angle = angle_data[valid_indices][:, 0]
            grid = self.grids[stride][valid_indices]

            a_rad = (sigmoid(v_angle) - 0.25) * math.pi * self.cfg.angle_sign + self.angle_offset_rad
            l, t, r, b = v_box.T
            xf, yf = (r - l) / 2.0, (b - t) / 2.0
            c, s = np.cos(a_rad), np.sin(a_rad)
            cx = (grid[:, 0] + xf * c - yf * s) * stride
            cy = (grid[:, 1] + xf * s + yf * c) * stride
            w = (l + r) * stride
            h = (t + b) * stride

            for _cx, _cy, _w, _h, _a, _s, _id in zip(cx, cy, w, h, a_rad, v_scores, v_ids):
                if self.cfg.regularize and _w < _h:
                    _w, _h, _a = _h, _w, _a + math.pi / 2.0
                _a = (_a + math.pi / 2.0) % math.pi - math.pi / 2.0
                rrects.append([_cx, _cy, _w, _h, _a])
                scores.append(float(_s))
                cids.append(int(_id))

        final_res = []
        if rrects:
            keep = self._nms_rotated(rrects, scores, cids, self.cfg.nms_thres)
            for idx in keep:
                cx, cy, w, h, a = self._scale_rrect_to_original(rrects[idx], ori_img_w, ori_img_h)
                final_res.append({
                    "rrect": (cx, cy, w, h, a),
                    "score": scores[idx],
                    "id": cids[idx],
                })

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res

    def predict(self, img: np.ndarray) -> List[Dict]:
        """
        Run the complete OBB inference pipeline on a single image.

        This method orchestrates the full workflow:
        - Pre-process the input image.
        - Execute BPU inference.
        - Decode and filter rotated bounding boxes.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            List[Dict]: Oriented bounding box results from `post_process()`.
        """
        ori_img_h, ori_img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, img: np.ndarray) -> List[Dict]:
        """
        Provide functional-style calling capability.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            List[Dict]: Oriented bounding box results from `predict()`.
        """
        return self.predict(img)
