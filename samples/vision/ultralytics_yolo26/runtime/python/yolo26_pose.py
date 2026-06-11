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
# flake8: noqa: E402

"""Provide YOLO26 Pose inference wrapper and pipeline utilities.

This module defines a YOLO26 pose estimation runtime wrapper built on
HBM runtime. It handles Box, Class, and Keypoint decoding.

Model output layout (9 tensors):
    Stride 8:  [0] Cls(1), [1] Box(4), [2] Kpt(51)
    Stride 16: [3] Cls(1), [4] Box(4), [5] Kpt(51)
    Stride 32: [6] Cls(1), [7] Box(4), [8] Kpt(51)
"""

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YOLO26PoseConfig:
    """Configuration for the YOLO26 Pose model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for NMS.
        resize_type: Image resize mode (0=stretch, 1=letterbox).
        strides: Feature map strides.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])


class YOLO26Pose:
    """YOLO26 pose estimation wrapper based on HB_HBMRuntime."""

    def __init__(self, config: YOLO26PoseConfig):
        """Initialize the YOLO26 Pose model.

        Args:
            config: Configuration object.
        """
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

    def set_scheduling_params(self, priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters."""
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into NV12 Y/UV tensor format.

        Args:
            img: Input image array.
            image_format: Input image format. Only "BGR" is supported.

        Returns:
            Nested input tensor dictionary.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, self.cfg.resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """Execute model inference."""
        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict, ori_img_w: int, ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw outputs into final pose results.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS override.

        Returns:
            Tuple of (boxes, scores, cls_ids, keypoints).
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        raw_outputs = outputs[self.model_name]
        decoded = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_name = self.output_names[base_idx]
            box_name = self.output_names[base_idx + 1]
            kpt_name = self.output_names[base_idx + 2]

            layer_pred = post_utils.decode_pose_layer(
                raw_outputs[box_name], raw_outputs[cls_name],
                raw_outputs[kpt_name], stride, score_thres)
            decoded.append(layer_pred)

        if not decoded:
            return np.array([]), np.array([]), np.array([]), np.array([])

        pred = np.concatenate(decoded, axis=0)
        if pred.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        kpts = pred[:, 6:].reshape(-1, 17, 3)

        keep = post_utils.NMS(xyxy, score, cls, nms_thres)
        if not keep:
            return np.array([]), np.array([]), np.array([]), np.array([])

        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        kpts = kpts[keep]

        xyxy = post_utils.scale_coords_back(xyxy, ori_img_w, ori_img_h,
                                            self.input_w, self.input_h, self.cfg.resize_type)
        kpts_xy = kpts[..., :2]
        kpts_score = kpts[..., 2:3]
        kpts_xy, kpts_score = post_utils.scale_keypoints_to_original_image(
            kpts_xy, kpts_score, ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)
        kpts = np.concatenate([kpts_xy, kpts_score], axis=-1)

        return xyxy, score, cls.astype(int), kpts

    def predict(self, img: np.ndarray, image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete pose estimation pipeline.

        Args:
            img: Input image array.
            image_format: Input image format.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            Tuple of (boxes, scores, cls_ids, keypoints).
        """
        ori_img_h, ori_img_w = img.shape[:2]
        input_tensor = self.pre_process(img, image_format)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface equivalent to predict()."""
        return self.predict(img, **kwargs)
