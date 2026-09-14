# Copyright (c) 2026 D-Robotics Corporation
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
FCOS detection inference module.

This module implements the FCOS object detection pipeline on RDK X5 using
`hbm_runtime`. The implementation keeps the output protocol aligned with the
original FCOS demo in this repository:

- 5 classification outputs
- 5 box regression outputs
- 5 center-ness outputs

The wrapper resolves these tensors in a fixed FCOS-compatible order, performs
NV12 preprocessing, runs BPU inference, decodes the FCOS predictions, and
returns standardized detection results as `boxes`, `scores`, and `cls_ids`.
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


logger = logging.getLogger("FCOS")


@dataclass
class FCOSConfig:
    """
    Configuration for FCOS detection inference.

    Args:
        model_path (str): Path to the FCOS `.bin` model file.
        classes_num (int): Number of detection classes.
        conf_thres (float): Confidence threshold for candidate filtering.
        iou_thres (float): IoU threshold used by NMS.
        resize_type (int): Image resize strategy used in preprocessing.
        strides (List[int]): Feature strides of the FCOS heads.
        use_stride_scaling (bool): Whether box outputs need stride scaling.
    """

    model_path: str
    classes_num: int = 80
    conf_thres: float = 0.5
    iou_thres: float = 0.6
    resize_type: int = 0
    strides: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    use_stride_scaling: bool = True


class FCOSDetect:
    """
    FCOS detection wrapper based on `hbm_runtime`.

    This wrapper encapsulates model loading, scheduling configuration,
    preprocessing, forward execution, and post-processing for FCOS detection
    models on RDK X5.
    """

    def __init__(self, config: FCOSConfig):
        """
        Initialize the FCOS runtime wrapper.

        Args:
            config (FCOSConfig): Runtime configuration for FCOS inference.

        The initialization stage loads the model, reads input/output metadata,
        builds FCOS feature grids for all configured strides, and resolves the
        fixed output ordering used by the original FCOS demo.
        """

        self.cfg = config

        t0 = time.time()
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_shapes = self.model.output_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]

        self.grids: Dict[int, np.ndarray] = {}
        for stride in self.cfg.strides:
            grid_h = self.input_h // stride
            grid_w = self.input_w // stride
            yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w))
            self.grids[stride] = (((np.stack((yv, xv), 2) + 0.5) * stride).reshape(-1, 2)).astype(np.float32)

        self.cls_output_names, self.box_output_names, self.center_output_names = self._resolve_output_order()

    def _resolve_output_order(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Resolve the FCOS output order based on the original fixed protocol.

        Returns:
            Tuple[List[str], List[str], List[str]]: Classification output names,
            box regression output names, and center-ness output names, each
            ordered by ascending stride.

        Raises:
            RuntimeError: Raised when a required FCOS output tensor cannot be
            matched from the loaded model metadata.
        """

        cls_names: List[str] = []
        box_names: List[str] = []
        center_names: List[str] = []
        used_names = set()

        for branch_names, channels in (
            (cls_names, self.cfg.classes_num),
            (box_names, 4),
            (center_names, 1),
        ):
            for stride in self.cfg.strides:
                expected_shape = (1, self.input_h // stride, self.input_w // stride, channels)
                matched_name = None
                for output_name in self.output_names:
                    if output_name in used_names:
                        continue
                    if tuple(self.output_shapes[output_name]) == expected_shape:
                        matched_name = output_name
                        break
                if matched_name is None:
                    raise RuntimeError(f"Failed to resolve FCOS output with shape {expected_shape}")
                used_names.add(matched_name)
                branch_names.append(matched_name)

        return cls_names, box_names, center_names

    def set_scheduling_params(self, priority: Optional[int] = None, bpu_cores: Optional[List[int]] = None) -> None:
        """
        Configure runtime scheduling parameters.

        Args:
            priority (Optional[int]): Runtime priority passed to
                `HB_HBMRuntime.set_scheduling_params`.
            bpu_cores (Optional[List[int]]): BPU core indexes used for
                inference execution.
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
        Convert a BGR image to packed NV12 input for `hbm_runtime`.

        Args:
            image (np.ndarray): Input BGR image.
            resize_type (Optional[int]): Override resize mode. When `None`, the
                value from `FCOSConfig.resize_type` is used.
            image_format (str): Expected source image format. Only `BGR` is
                supported in this sample.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Nested input dictionary in the
            format required by `HB_HBMRuntime.run`.

        Raises:
            ValueError: Raised when the provided image format is unsupported.
        """

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type
        resized = pre_utils.resized_image(image, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)
        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        return {self.model_name: {self.input_names[0]: packed_nv12}}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Execute inference on BPU using `hbm_runtime`.

        Args:
            inputs (Dict[str, Dict[str, np.ndarray]]): Runtime input dictionary
                produced by `pre_process`.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Raw runtime outputs indexed by
            model name and tensor name.
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
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode FCOS outputs into final detections.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw inference outputs.
            ori_img_w (int): Width of the original input image.
            ori_img_h (int): Height of the original input image.
            conf_thres (Optional[float]): Override confidence threshold.
            iou_thres (Optional[float]): Override NMS IoU threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - boxes: `N x 4` float32 array in `xyxy` format
                - scores: `N` float32 confidence scores
                - cls_ids: `N` int32 class ids

        The decoding logic follows the original FCOS demo protocol: outputs are
        dequantized, confidence is computed from class score and center-ness,
        per-level boxes are decoded on FCOS grids, and NMS is applied before
        coordinates are mapped back to the original image size.
        """

        t0 = time.time()
        conf_thres = self.cfg.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.cfg.iou_thres if iou_thres is None else iou_thres

        raw_outputs = outputs[self.model_name]
        fp32_outputs = post_utils.dequantize_outputs(raw_outputs, self.output_quants)

        scores_list: List[np.ndarray] = []
        ids_list: List[np.ndarray] = []
        index_list: List[np.ndarray] = []

        for cls_name, center_name in zip(self.cls_output_names, self.center_output_names):
            cls = fp32_outputs[cls_name].reshape(-1, self.cfg.classes_num)
            center = fp32_outputs[center_name].reshape(-1, 1)
            raw_max_scores = np.max(cls, axis=1)
            max_scores = np.sqrt(post_utils.sigmoid(center[:, 0]) * post_utils.sigmoid(raw_max_scores))
            valid_indices = np.flatnonzero(max_scores >= conf_thres)
            scores_list.append(max_scores[valid_indices])
            ids_list.append(np.argmax(cls[valid_indices, :], axis=1))
            index_list.append(valid_indices)

        boxes_all: List[np.ndarray] = []
        for indices, stride, box_name in zip(index_list, self.cfg.strides, self.box_output_names):
            grid = self.grids[stride][indices, :]
            bbox = fp32_outputs[box_name].reshape(-1, 4)[indices, :]
            if self.cfg.use_stride_scaling:
                bbox = bbox * stride
            x1y1 = grid - bbox[:, 0:2]
            x2y2 = grid + bbox[:, 2:4]
            boxes_all.append(np.hstack([x1y1, x2y2]))

        if not boxes_all or not any(box.shape[0] for box in boxes_all):
            logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        xyxy = np.concatenate(boxes_all, axis=0).astype(np.float32)
        scores = np.concatenate(scores_list, axis=0).astype(np.float32)
        cls_ids = np.concatenate(ids_list, axis=0).astype(np.int32)

        xywh = xyxy.copy()
        xywh[:, 2:] -= xywh[:, :2]
        nms_indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_thres, iou_thres)
        if len(nms_indices) == 0:
            logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        nms_indices = np.array(nms_indices).reshape(-1)
        boxes = xyxy[nms_indices].copy()
        boxes[:, [0, 2]] *= ori_img_w / self.input_w
        boxes[:, [1, 3]] *= ori_img_h / self.input_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ori_img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ori_img_h)
        scores = scores[nms_indices]
        cls_ids = cls_ids[nms_indices]

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return boxes.astype(np.float32), scores.astype(np.float32), cls_ids.astype(np.int32)

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the full FCOS pipeline on one image.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Final `boxes`, `scores`,
            and `cls_ids` for visualization or downstream processing.
        """

        ori_img_h, ori_img_w = image.shape[:2]
        input_tensors = self.pre_process(image)
        outputs = self.forward(input_tensors)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide functional-style inference capability.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Same result as
            `predict(image)`.
        """

        return self.predict(image)
