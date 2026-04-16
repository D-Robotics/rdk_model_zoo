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
Ultralytics YOLO Segmentation Runtime Wrapper.

This module implements the maintained Python segmentation wrapper for the
`ultralytics_yolo` sample on `RDK X5`. It is shared by all segmentation
families delivered in this sample:

    - YOLOv8
    - YOLOv9
    - YOLO11

The exported X5 models use one fixed output protocol:

    - [cls, box, mask_coeff] * 3
    - one final proto output

The wrapper uses shared preprocessing and postprocessing utilities under
`utils/py_utils`, while the task-specific decode flow remains inside this
module.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hbm_runtime
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))

import utils.py_utils.postprocess as post_utils
import utils.py_utils.preprocess as pre_utils


logger = logging.getLogger("Ultralytics_YOLO")


@dataclass
class UltralyticsYOLOSegConfig:
    """Configuration used by the segmentation wrapper.

    Args:
        model_path: Path to the X5 `.bin` model.
        classes_num: Number of categories predicted by the segmentation head.
        score_thres: Confidence threshold used before NMS.
        nms_thres: IoU threshold used by class-wise NMS.
        reg: Number of DFL bins for one box edge.
        mc: Number of mask coefficients per prediction.
        resize_type: Resize policy, `0` for direct resize and `1` for
            letterbox.
        strides: Feature strides used by the three segmentation heads.
    """

    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.70
    reg: int = 16
    mc: int = 32
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class UltralyticsYOLOSeg:
    """Segmentation wrapper built on top of `hbm_runtime`."""

    def __init__(self, config: UltralyticsYOLOSegConfig):
        """Initialize the segmentation runtime wrapper.

        Args:
            config: Segmentation runtime configuration for the current model.
        """
        self.cfg = config
        self.conf_thres_raw = -np.log(1.0 / self.cfg.score_thres - 1.0)
        self.weights_static = np.arange(self.cfg.reg, dtype=np.float32)[None, None, :]

        t0 = time.time()
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info("\033[1;31mLoad Model time = %.2f ms\033[0m", 1000 * (time.time() - t0))

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

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[list] = None,
    ) -> None:
        """Set optional BPU scheduling parameters.

        Args:
            priority: Runtime priority used by `hbm_runtime`.
            bpu_cores: BPU core list used by the model scheduler.
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
        img: np.ndarray,
        resize_type: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert one BGR image into packed NV12 model input.

        Args:
            img: Input BGR image loaded by OpenCV.
            resize_type: Optional resize policy override.

        Returns:
            The nested input tensor dictionary required by `HB_HBMRuntime.run`.
        """
        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type

        resized = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)

        logger.info("\033[1;31mPre-process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return {self.model_name: {self.input_names[0]: packed_nv12}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run BPU inference for one input image."""
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info("\033[1;31mForward time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return outputs

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        ori_img_w: int,
        ori_img_h: int,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Decode segmentation tensors into boxes, classes, and masks.

        Args:
            outputs: Raw runtime outputs returned by `forward`.
            ori_img_w: Original image width.
            ori_img_h: Original image height.
            score_thres: Optional score threshold override.
            nms_thres: Optional NMS threshold override.

        Returns:
            A tuple containing:
                - scaled bounding boxes
                - confidence scores
                - class ids
                - resized binary masks
        """
        t0 = time.time()
        score_thres = self.cfg.score_thres if score_thres is None else score_thres
        nms_thres = self.cfg.nms_thres if nms_thres is None else nms_thres
        conf_thres_raw = -np.log(1.0 / score_thres - 1.0)
        raw_outputs = outputs[self.model_name]

        boxes_all = []
        scores_all = []
        cls_all = []
        mces_all = []

        for level_index, stride in enumerate(self.cfg.strides):
            base_idx = level_index * 3
            cls_output = raw_outputs[self.output_names[base_idx]].reshape(-1, self.cfg.classes_num)
            box_output = raw_outputs[self.output_names[base_idx + 1]]
            mc_output = raw_outputs[self.output_names[base_idx + 2]]

            scores, cls_ids, valid_indices = post_utils.filter_classification(cls_output, conf_thres_raw)
            if valid_indices.size == 0:
                continue

            grid_size = self.input_h // stride
            boxes = post_utils.decode_boxes(
                box_output,
                valid_indices,
                grid_size,
                stride,
                self.weights_static,
            )
            mces = post_utils.filter_mces(mc_output, valid_indices)

            boxes_all.append(boxes)
            scores_all.append(scores)
            cls_all.append(cls_ids.astype(np.int32))
            mces_all.append(mces)

        if not boxes_all:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        boxes = np.concatenate(boxes_all, axis=0).astype(np.float32)
        scores = np.concatenate(scores_all, axis=0).astype(np.float32)
        cls_ids = np.concatenate(cls_all, axis=0).astype(np.int32)
        mces = np.concatenate(mces_all, axis=0).astype(np.float32)

        keep = post_utils.NMS(boxes, scores, cls_ids, nms_thres)
        if not keep:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        mces = mces[keep]

        proto = raw_outputs[self.output_names[9]]
        if proto.shape[0] == 1:
            proto = proto[0]
        if proto.shape[-1] != self.cfg.mc and proto.shape[0] == self.cfg.mc:
            proto = np.transpose(proto, (1, 2, 0))

        masks = post_utils.decode_masks(
            mces,
            boxes,
            proto,
            self.input_w,
            self.input_h,
            proto.shape[1],
            proto.shape[0],
            mask_thresh=0.5,
        )
        scaled_boxes = post_utils.scale_coords_back(
            boxes.copy(),
            ori_img_w,
            ori_img_h,
            self.input_w,
            self.input_h,
            self.cfg.resize_type,
        )
        masks = post_utils.resize_masks_to_boxes(masks, scaled_boxes, ori_img_w, ori_img_h)

        logger.info("\033[1;31mPost Process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return scaled_boxes, scores, cls_ids, masks

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Run the full segmentation pipeline on one image."""
        ori_img_h, ori_img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Provide function-style inference for one image."""
        return self.predict(img)
