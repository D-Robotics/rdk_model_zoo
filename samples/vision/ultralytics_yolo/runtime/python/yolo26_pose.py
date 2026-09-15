# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from yolo26_common import Yolo26Runtime
from yolo_platform import PlatformProfile
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
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import rdk_yolo_utils.preprocess as pre_utils
import rdk_yolo_utils.postprocess as post_utils

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
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None

class YOLO26Pose(Yolo26Runtime):
    task = 'pose'
    'YOLO26 pose estimation wrapper based on HB_HBMRuntime.'

    def post_process(self, outputs: Dict, ori_img_w: int, ori_img_h: int, score_thres: Optional[float]=None, nms_thres: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw outputs into final pose results.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS override.

        Returns:
            Tuple of (boxes, scores, cls_ids, keypoint_xy, keypoint_confidence).
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
            layer_pred = post_utils.decode_pose_layer(raw_outputs[box_name], raw_outputs[cls_name], raw_outputs[kpt_name], stride, score_thres)
            decoded.append(layer_pred)
        if not decoded:
            return (np.empty((0,4)), np.empty(0), np.empty(0,dtype=int), np.empty((0,17,2)), np.empty((0,17,1)))
        pred = np.concatenate(decoded, axis=0)
        if pred.shape[0] == 0:
            return (np.empty((0,4)), np.empty(0), np.empty(0,dtype=int), np.empty((0,17,2)), np.empty((0,17,1)))
        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        kpts = pred[:, 6:].reshape(-1, 17, 3)
        keep = post_utils.NMS(xyxy, score, cls, nms_thres)
        if not keep:
            return (np.empty((0,4)), np.empty(0), np.empty(0,dtype=int), np.empty((0,17,2)), np.empty((0,17,1)))
        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        kpts = kpts[keep]
        xyxy = post_utils.scale_coords_back(xyxy, ori_img_w, ori_img_h, self.input_w, self.input_h, self.cfg.resize_type)
        kpts_xy = kpts[..., :2]
        kpts_score = kpts[..., 2:3]
        kpts_xy, kpts_score = post_utils.scale_keypoints_to_original_image(kpts_xy, kpts_score, ori_img_w, ori_img_h, self.input_w, self.input_h, self.cfg.resize_type)
        kpts = np.concatenate([kpts_xy, kpts_score], axis=-1)
        return xyxy, score, cls.astype(int), kpts[...,:2], kpts[...,2:3]
