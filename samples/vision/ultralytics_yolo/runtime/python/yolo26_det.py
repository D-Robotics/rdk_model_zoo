# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from yolo26_common import Yolo26Runtime
from yolo_platform import PlatformProfile
"""Provide a YOLO26 anchor-free detection inference wrapper and pipeline utilities.

This module defines a YOLO26 detection runtime wrapper built on HBM runtime.
It uses LTRB (Left-Top-Right-Bottom) anchor-relative box decoding, as opposed
to DFL-based decoding used by v5u/v8/v11/v13.

Key Features:
    - YOLO26DetectConfig dataclass for configuring model parameters.
    - YOLO26Detect class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Anchor-free LTRB box decoding and class-wise NMS.

Typical Usage:
    >>> from yolo26_det import YOLO26Detect, YOLO26DetectConfig
    >>> cfg = YOLO26DetectConfig(model_path="/path/to/yolo26n_detect.hbm")
    >>> model = YOLO26Detect(cfg)
    >>> boxes, scores, cls_ids = model(img)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - The detection head uses anchor-free LTRB regression. Each detection
      scale emits a paired classification output and a box output.
    - hbm_runtime outputs float32 tensors for YOLO26 models, so no
      dequantization step is needed in post-processing.
"""
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import rdk_yolo_utils.preprocess as pre_utils
import rdk_yolo_utils.postprocess as post_utils

@dataclass
class YOLO26DetectConfig:
    """Configuration for initializing the YOLO26Detect model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO26 detection
    pipeline.

    Attributes:
        model_path: Path to the compiled YOLO26 detection `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80 (COCO).
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes (in pixels) for each detection scale.
    """
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = None
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None

class YOLO26Detect(Yolo26Runtime):
    task = 'detect'
    'YOLO26 anchor-free detection wrapper based on HB_HBMRuntime.\n\n    This class provides a unified inference pipeline for YOLO26 detection,\n    including input preprocessing, model execution, and postprocessing steps\n    such as LTRB box decoding, confidence filtering, and class-wise NMS.\n\n    Attributes:\n        model: Loaded HBM runtime model instance.\n        model_name: Name of the first loaded model.\n        input_names: Input tensor name list.\n        output_names: Output tensor name list.\n        input_shapes: Input tensor shape dictionary.\n        input_h: Model input height (pixels).\n        input_w: Model input width (pixels).\n        grids: Precomputed anchor grid centers for each detection stride.\n        cfg: Model configuration object.\n\n    Notes:\n        YOLO26 uses LTRB (Left-Top-Right-Bottom) anchor-relative box\n        decoding, unlike DFL-based models (v5u/v8/v11/v13) which use\n        distribution-based regression.\n    '

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]], ori_img_w: int, ori_img_h: int, score_thres: Optional[float]=None, nms_thres: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into final detection results.

        This step includes LTRB box decoding, confidence filtering,
        Non-Maximum Suppression (NMS), and coordinate scaling back to the
        original image resolution.

        Note: hbm_runtime returns float32 tensors for YOLO26 models, so no
        dequantization is applied.

        Args:
            outputs: Raw output tensors from inference (as returned by `forward()`).
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override. If `None`, the value
                from the configuration is used.
            nms_thres: IoU threshold for NMS override. If `None`, the value
                from the configuration is used.

        Returns:
            A tuple containing:
                - boxes: Bounding boxes with shape `(N, 4)` in original image
                  coordinates, formatted as `[x1, y1, x2, y2]`.
                - scores: Confidence scores with shape `(N,)`.
                - cls_ids: Class indices with shape `(N,)`.
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres
        conf_thres_raw = -np.log(1.0 / score_thres - 1.0)
        raw_outputs = outputs[self.model_name]
        all_boxes = []
        all_scores = []
        all_ids = []
        for i, (stride, anchor_size) in enumerate(zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key = self.output_names[2 * i]
            box_key = self.output_names[2 * i + 1]
            cls_data = raw_outputs[cls_key].reshape(-1, self.cfg.classes_num)
            box_data = raw_outputs[box_key].reshape(-1, 4)
            scores, ids, valid_indices = post_utils.filter_classification(cls_data, conf_thres_raw)
            if valid_indices.size == 0:
                continue
            grid = self.grids[stride][valid_indices]
            valid_box = box_data[valid_indices]
            dbboxes = post_utils.decode_ltrb_boxes(grid, valid_box, stride)
            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
        if not all_boxes:
            return (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32))
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_ids, axis=0)
        keep = post_utils.NMS(boxes, scores, cls_ids, nms_thres)
        xyxy = post_utils.scale_coords_back(boxes[keep], ori_img_w, ori_img_h, self.input_w, self.input_h, self.cfg.resize_type)
        if self.cfg.platform.family == 'x5':
            order = np.argsort(cls_ids[keep], kind='stable')
            return xyxy[order], scores[keep][order], cls_ids[keep][order]
        return xyxy, scores[keep], cls_ids[keep]
