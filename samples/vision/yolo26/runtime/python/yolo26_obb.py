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

"""YOLO26 OBB (Oriented Bounding Box) Model Implementation.

This module provides the YOLO26OBB class for rotated object detection.
It encapsulates model loading, preprocessing, BPU inference, and postprocessing
(including angle decoding and rotated NMS).
"""

import time
import logging
import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

logger = logging.getLogger("YOLO26_OBB")


@dataclass
class YOLO26OBBConfig:
    """Configuration for YOLO26 OBB Model.

    Attributes:
        model_path (str): Path to the .bin model file.
        score_thres (float): Confidence threshold for filtering detections.
        nms_thres (float): IoU threshold for Non-Maximum Suppression.
        angle_sign (float): Multiplier for angle decoding (1.0 or -1.0).
        angle_offset (float): Offset angle in degrees.
        regularize (bool): Whether to regularize rotated boxes (w < h swap).
        strides (List[int]): List of strides for the feature maps.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26OBB:
    """YOLO26 Oriented Bounding Box Model Wrapper.

    This class handles the complete inference pipeline:
    1. Model Loading
    2. Preprocessing (Resize, Pad, Color Conversion)
    3. Inference (BPU Forward)
    4. Postprocessing (Rotated Box Decoding, NMS)
    """

    def __init__(self, config: YOLO26OBBConfig):
        """Initialize the YOLO26 OBB model.

        Args:
            config (YOLO26OBBConfig): Configuration object.
        """
        self.cfg = config
        self.conf_raw = -np.log(1 / self.cfg.score_thres - 1)
        self.angle_offset_rad = self.cfg.angle_offset * math.pi / 180.0
        
        # Load Model
        try:
            t0 = time.time()
            self.model = dnn.load(self.cfg.model_path)[0]
            logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e
        
        # Input Shape
        shape = self.model.inputs[0].properties.shape
        if shape[3] == 3:
            self.m_h, self.m_w = shape[1], shape[2]
        else:
            self.m_h, self.m_w = shape[2], shape[3]

        # Pre-compute Grids
        logger.info("Pre-computing Anchor-Free Grids...")
        self.grids = {}
        for s in self.cfg.strides:
            grid_h, grid_w = self.m_h // s, self.m_w // s
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5
        
        # Mapping for output indices (Box, Cls, Angle)
        # Note: This is model-specific hardcoding based on original script
        self.map_idx = {8: (0, 1, 2), 16: (3, 4, 5), 32: (6, 7, 8)}

        self.scale = 1.0
        self.orig_h = 0
        self.orig_w = 0

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Flattened NV12 data ready for inference.
        """
        t0 = time.time()
        self.orig_h, self.orig_w = img.shape[:2]
        self.scale = min(self.m_h / self.orig_h, self.m_w / self.orig_w)
        nw, nh = int(self.orig_w * self.scale), int(self.orig_h * self.scale)
        
        input_tensor = cv2.copyMakeBorder(
            cv2.resize(img, (nw, nh)), 
            0, self.m_h - nh, 0, self.m_w - nw, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        yuv = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.empty((self.m_h * self.m_w * 3 // 2,), dtype=np.uint8)
        y_size = self.m_h * self.m_w
        nv12[:y_size] = yuv[:y_size]
        nv12[y_size::2] = yuv[y_size:y_size + y_size // 4]
        nv12[y_size + 1::2] = yuv[y_size + y_size // 4:]
        
        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return nv12

    def forward(self, nv12: np.ndarray) -> List[np.ndarray]:
        """Run forward inference.

        Args:
            nv12 (np.ndarray): Preprocessed NV12 data.

        Returns:
            List[np.ndarray]: Raw model outputs.
        """
        t0 = time.time()
        out = self.model.forward(nv12)
        logger.info(f"\033[1;31mForward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return out

    def post_process(self, outputs) -> List[Dict]:
        """Process model outputs to generate OBB results.

        Args:
            outputs: Raw outputs from the model.

        Returns:
            List[Dict]: List of OBB dicts containing:
                - 'rrect': Rotated rect tuple (cx, cy, w, h, angle_rad)
                - 'score': Confidence score
                - 'id': Class ID
        """
        t0 = time.time()
        rrects = []
        scores = []
        cids = []

        for stride in self.cfg.strides:
            if stride not in self.map_idx:
                continue
            
            bi, ci, ai = self.map_idx[stride]
            
            # Check bounds
            if max(bi, ci, ai) >= len(outputs):
                logger.warning(f"Stride {stride} indices {bi, ci, ai} out of bounds for outputs len {len(outputs)}")
                continue

            box_data = outputs[bi].buffer.reshape(-1, 4)
            cls_data = outputs[ci].buffer.reshape(-1, outputs[ci].buffer.shape[-1])
            angle_data = outputs[ai].buffer.reshape(-1, 1)

            max_scores = np.max(cls_data, axis=1)
            mask = max_scores >= self.conf_raw
            if not np.any(mask):
                continue

            v_scores = 1 / (1 + np.exp(-max_scores[mask]))
            v_ids = np.argmax(cls_data[mask], axis=1)
            v_box = np.abs(box_data[mask])
            v_angle = angle_data[mask]
            grid = self.grids[stride][mask]
            
            # Angle decoding
            a_rad = (1 / (1 + np.exp(-v_angle[:, 0])) - 0.5) * math.pi * self.cfg.angle_sign + self.angle_offset_rad
            
            l, t, r, b = v_box.T
            xf, yf = (r - l) / 2.0, (b - t) / 2.0
            
            c, s = np.cos(a_rad), np.sin(a_rad)
            cx = (grid[:, 0] + xf * c - yf * s) * stride
            cy = (grid[:, 1] + xf * s + yf * c) * stride
            w = (l + r) * stride
            h = (t + b) * stride
            
            for _cx, _cy, _w, _h, _a, _s, _id in zip(cx, cy, w, h, a_rad, v_scores, v_ids):
                if self.cfg.regularize and _w < _h:
                    _w, _h, _a = _h, _w, _a + math.pi / 2
                
                rrects.append((_cx, _cy, _w, _h, _a))
                scores.append(float(_s))
                cids.append(int(_id))

        final_res = []
        if rrects:
            try:
                # NMS Rotated
                box_list = [((r[0], r[1]), (r[2], r[3]), r[4] * 180 / math.pi) for r in rrects]
                indices = cv2.dnn.NMSBoxesRotated(
                    box_list, scores, 
                    self.cfg.score_thres, self.cfg.nms_thres
                )
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        cx, cy, w, h, a = rrects[i]
                        final_res.append({
                            'rrect': (cx / self.scale, cy / self.scale, w / self.scale, h / self.scale, a),
                            'score': scores[i],
                            'id': cids[i]
                        })
            except AttributeError:
                logger.warning("cv2.dnn.NMSBoxesRotated not available.")
                
        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res