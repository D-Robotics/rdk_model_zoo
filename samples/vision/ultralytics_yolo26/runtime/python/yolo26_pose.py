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
YOLO26 Pose Inference Module.

This module implements the YOLO26 pose estimation pipeline on BPU,
including pre-processing, forward execution, and post-processing.

Key Features:
    - Optimized for RDK X5 single-input NV12 models.
    - Supports YOLO26 keypoint decoding on multi-stride feature heads.
    - Maps boxes and keypoints back to original image coordinates.

Typical Usage:
    >>> from yolo26_pose import YOLO26PoseConfig, YOLO26Pose
    >>> config = YOLO26PoseConfig(model_path="path/to/yolo26_pose.bin")
    >>> model = YOLO26Pose(config)
    >>> results = model.predict(image)
"""

import os
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add project root to sys.path to import shared utilities.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_Pose")


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
class YOLO26PoseConfig:
    """
    Configuration for YOLO26 pose inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        score_thres (float): Confidence threshold.
        nms_thres (float): IoU threshold for NMS.
        kpt_conf_thres (float): Keypoint confidence threshold.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        strides (List[int]): Detection head strides.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.7
    kpt_conf_thres: float = 0.5
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26Pose:
    """
    YOLO26 pose wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLO26PoseConfig):
        """
        Initialize the model, load metadata, and precompute decoding grids.

        Args:
            config (YOLO26PoseConfig): Configuration object containing model path and params.
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

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int) -> List[Dict]:
        """
        Convert raw outputs to final pose estimation results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.

        Returns:
            List[Dict]: Pose results containing the bounding box, score, and
            decoded keypoints in original image coordinates.
        """
        t0 = time.time()
        raw_outputs = outputs[self.model_name]
        detections = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_data = raw_outputs[self.output_names[base_idx]].reshape(-1, 1)
            box_data = raw_outputs[self.output_names[base_idx + 1]].reshape(-1, 4)
            kpt_data = raw_outputs[self.output_names[base_idx + 2]].reshape(-1, 17, 3)

            valid_scores, _, valid_indices = post_utils.filter_classification(cls_data, self.conf_raw)
            if valid_indices.size == 0:
                continue

            grid = self.grids[stride][valid_indices]
            valid_box = box_data[valid_indices]
            valid_kpts = kpt_data[valid_indices]

            xyxy = post_utils.decode_ltrb_boxes(grid, valid_box, stride)

            kpt_xy = (valid_kpts[:, :, :2] + grid[:, None, :]) * stride
            kpt_conf = sigmoid(valid_kpts[:, :, 2:3])
            decoded_kpts = np.concatenate([kpt_xy, kpt_conf], axis=-1)

            for box, score, kpts in zip(xyxy, valid_scores, decoded_kpts):
                detections.append({"box": box, "score": score, "kpts": kpts})

        final_res = []
        if detections:
            boxes = np.array([d["box"] for d in detections], dtype=np.float32)
            scores = np.array([d["score"] for d in detections], dtype=np.float32)
            cls = np.zeros(len(boxes), dtype=np.int32)
            keep = post_utils.NMS(boxes, scores, cls, self.cfg.nms_thres)

            for idx in keep:
                det = detections[idx]
                box = post_utils.scale_coords_back(
                    det["box"][None, :].copy(),
                    ori_img_w,
                    ori_img_h,
                    self.input_w,
                    self.input_h,
                    self.cfg.resize_type,
                )[0]

                kpt_xy, kpt_score = post_utils.scale_keypoints_to_original_image(
                    det["kpts"][:, :2][None, :, :],
                    det["kpts"][:, 2:3][None, :, :],
                    ori_img_w,
                    ori_img_h,
                    self.input_w,
                    self.input_h,
                    self.cfg.resize_type,
                )
                kpts = np.concatenate([kpt_xy[0], kpt_score[0]], axis=-1)
                final_res.append({
                    "box": box.astype(int),
                    "score": float(det["score"]),
                    "kpts": kpts,
                })

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res

    def predict(self, img: np.ndarray) -> List[Dict]:
        """
        Run the complete pose inference pipeline on a single image.

        This method orchestrates the full workflow:
        - Pre-process the input image.
        - Execute BPU inference.
        - Decode boxes and keypoints on the original image scale.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            List[Dict]: Pose estimation results from `post_process()`.
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
            List[Dict]: Pose estimation results from `predict()`.
        """
        return self.predict(img)
