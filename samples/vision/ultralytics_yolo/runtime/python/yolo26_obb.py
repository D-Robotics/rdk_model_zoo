# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from yolo26_common import Yolo26Runtime
from yolo_platform import PlatformProfile
"""Provide YOLO26 OBB inference wrapper and pipeline utilities.

This module defines a YOLO26 oriented bounding box detection runtime
wrapper built on HBM runtime. It handles rotated box decoding, angle
calculation, and rotated NMS.

Model output layout (9 tensors):
    Stride 8:  [0] Cls(15), [1] Box(4), [2] Angle(1)
    Stride 16: [3] Cls(15), [4] Box(4), [5] Angle(1)
    Stride 32: [6] Cls(15), [7] Box(4), [8] Angle(1)
"""
import os
import sys
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import rdk_yolo_utils.preprocess as pre_utils
import rdk_yolo_utils.postprocess as post_utils

@dataclass
class YOLO26OBBConfig:
    """Configuration for the YOLO26 OBB model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for rotated NMS.
        angle_sign: Multiplier for angle decoding.
        angle_offset: Offset in degrees to add to decoded angle.
        regularize: Whether to regularize boxes (w > h).
        resize_type: Image resize strategy (0=stretch, 1=letterbox).
        strides: Feature map strides.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[tuple] = None
    classes_num: int = 15

class YOLO26OBB(Yolo26Runtime):
    task = 'obb'
    'YOLO26 oriented bounding box wrapper based on HB_HBMRuntime.'

    def post_process(self, outputs: Dict, ori_w: int, ori_h: int, score_thres=None, nms_thres=None) -> List[Dict]:
        """Decode and post-process OBB results.

        Args:
            outputs: Raw output tensors from inference.
            ori_w: Original image width.
            ori_h: Original image height.

        Returns:
            List of dicts with keys 'rrect', 'score', 'id'.
        """
        score_thres = self.cfg.score_thres if score_thres is None else score_thres
        nms_thres = self.cfg.nms_thres if nms_thres is None else nms_thres
        conf_raw = self.logit_threshold(score_thres)
        raw_outputs = outputs[self.model_name]
        all_rrects = []
        all_scores = []
        all_cids = []
        for stride in self.cfg.strides:
            if stride not in self.map_idx:
                continue
            ci, bi, ai = self.map_idx[stride]
            if max(bi, ci, ai) >= len(self.output_names):
                raise ValueError(f'YOLO26 OBB expects 9 fixed outputs, got {len(self.output_names)} outputs.')
            box_feat = raw_outputs[self.output_names[bi]].reshape(-1, 4)
            cls_feat = raw_outputs[self.output_names[ci]].reshape(-1, raw_outputs[self.output_names[ci]].shape[-1])
            angle_feat = raw_outputs[self.output_names[ai]].reshape(-1, 1)
            max_scores = np.max(cls_feat, axis=1)
            mask = max_scores >= conf_raw
            if not np.any(mask):
                continue
            v_scores = post_utils.sigmoid(max_scores[mask])
            v_ids = np.argmax(cls_feat[mask], axis=1)
            v_box = np.abs(box_feat[mask])
            v_angle = angle_feat[mask]
            grid = self.grids[stride][mask]
            # The shared OBB26 exporter emits radians directly on X5 and S.
            a_rad = v_angle[:, 0] * self.cfg.angle_sign + self.angle_offset_rad
            l, t, r, b = v_box.T
            xf, yf = ((r - l) / 2.0, (b - t) / 2.0)
            c_cos, s_sin = (np.cos(a_rad), np.sin(a_rad))
            cx = (grid[:, 0] + xf * c_cos - yf * s_sin) * stride
            cy = (grid[:, 1] + xf * s_sin + yf * c_cos) * stride
            w = (l + r) * stride
            h = (t + b) * stride
            for _cx, _cy, _w, _h, _a, _s, _id in zip(cx, cy, w, h, a_rad, v_scores, v_ids):
                if self.cfg.regularize and _w < _h:
                    _w, _h, _a = (_h, _w, _a + math.pi / 2)
                if self.cfg.platform.family == 'x5':
                    _a = (_a + math.pi/2) % math.pi - math.pi/2
                all_rrects.append((_cx, _cy, _w, _h, _a))
                all_scores.append(float(_s))
                all_cids.append(int(_id))
        final_res = []
        if all_rrects:
            keep = []
            groups = [] if self.cfg.platform.family == 'x5' else [list(range(len(all_rrects)))]
            for group in groups:
                boxes = [((float(all_rrects[i][0]),float(all_rrects[i][1])), (float(all_rrects[i][2]),float(all_rrects[i][3])), float(math.degrees(all_rrects[i][4]))) for i in group]
                indices = cv2.dnn.NMSBoxesRotated(boxes,[all_scores[i] for i in group],score_thres,nms_thres)
                keep.extend(group[int(i)] for i in np.asarray(indices).reshape(-1))
            if self.cfg.platform.family == 'x5':
                keep=self._nms_rotated(all_rrects,all_scores,all_cids,nms_thres)
            for i in keep:
                cx,cy,w,h,angle=all_rrects[i]
                scaled=post_utils.scale_coords_back_obb(np.array([[cx,cy,w,h]]),ori_w,ori_h,self.input_w,self.input_h,self.cfg.resize_type)[0]
                if self.cfg.platform.family == 'x5':
                    scaled=self._scale_rrect_to_original(all_rrects[i],ori_w,ori_h)[:4]
                final_res.append({'rrect':(*scaled,angle),'score':all_scores[i],'id':all_cids[i]})
        return final_res

    def _rotated_iou(self, a, b) -> float:
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

    def _scale_rrect_to_original(self, rrect: List[float], ori_img_w: int, ori_img_h: int) -> List[float]:
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
