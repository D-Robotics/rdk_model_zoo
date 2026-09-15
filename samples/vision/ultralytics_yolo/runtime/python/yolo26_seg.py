# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from yolo26_common import Yolo26Runtime
from yolo_platform import PlatformProfile
"""Provide YOLO26 Segmentation inference wrapper and pipeline utilities.

This module defines a YOLO26 instance segmentation runtime wrapper built on
HBM runtime. It handles Box, Class, Mask Coefficient decoding, and
Prototype Mask assembly.

Model output layout (10 tensors):
    Stride 8:  [0] Cls(80), [1] Box(4), [2] MaskCoef(32)
    Stride 16: [3] Cls(80), [4] Box(4), [5] MaskCoef(32)
    Stride 32: [6] Cls(80), [7] Box(4), [8] MaskCoef(32)
    Proto:     [9] ProtoMaps (1, 160, 160, 32)
"""
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import rdk_yolo_utils.preprocess as pre_utils
import rdk_yolo_utils.postprocess as post_utils

@dataclass
class YOLO26SegConfig:
    """Configuration for the YOLO26 Segmentation model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        classes_num: Number of detection classes.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for NMS.
        resize_type: Image resize mode (0=stretch, 1=letterbox).
        strides: Feature map strides.
    """
    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: list = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None

class YOLO26Seg(Yolo26Runtime):
    task = 'seg'
    'YOLO26 instance segmentation wrapper based on HB_HBMRuntime.'

    def post_process(self, outputs: Dict, ori_img_w: int, ori_img_h: int, score_thres: Optional[float]=None, nms_thres: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Convert raw outputs into final segmentation results.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS override.

        Returns:
            Tuple of (boxes, scores, cls_ids, masks).
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres
        raw_outputs = outputs[self.model_name]
        decoded = []
        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_feat = raw_outputs[self.output_names[base_idx]]
            box_feat = raw_outputs[self.output_names[base_idx + 1]]
            mc_feat = raw_outputs[self.output_names[base_idx + 2]]
            layer_pred = post_utils.decode_seg_layer(box_feat, cls_feat, mc_feat, stride, score_thres, self.cfg.classes_num)
            decoded.append(layer_pred)
        proto_tensor = raw_outputs[self.output_names[9]]
        if proto_tensor.shape[0] == 1:
            proto_tensor = proto_tensor[0]
        proto_tensor = np.transpose(proto_tensor, (2, 0, 1))
        if not decoded:
            return (np.array([]), np.array([]), np.array([]), [])
        pred = np.concatenate(decoded, axis=0)
        if pred.shape[0] == 0:
            return (np.array([]), np.array([]), np.array([]), [])
        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        mask_coefs = pred[:, 6:]
        keep = post_utils.NMS(xyxy, score, cls, nms_thres)
        if not keep:
            return (np.array([]), np.array([]), np.array([]), [])
        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        mask_coefs = mask_coefs[keep]
        full_masks = post_utils.process_mask(proto_tensor, mask_coefs, xyxy, (ori_img_h, ori_img_w), self.input_h, self.input_w, self.cfg.resize_type)
        xyxy = post_utils.scale_coords_back(xyxy, ori_img_w, ori_img_h, self.input_w, self.input_h, self.cfg.resize_type)
        cropped_masks = []
        for i, box in enumerate(xyxy):
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(ori_img_w, int(box[2]))
            y2 = min(ori_img_h, int(box[3]))
            if x2 > x1 and y2 > y1:
                m = full_masks[i][y1:y2, x1:x2]
            else:
                m = np.zeros((0, 0), dtype=bool)
            cropped_masks.append(m)
        return (xyxy, score, cls.astype(int), cropped_masks)
