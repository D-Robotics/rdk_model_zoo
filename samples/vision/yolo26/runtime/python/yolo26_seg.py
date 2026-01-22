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

"""YOLO26 Segmentation Model Implementation.

This module provides the YOLO26Seg class for instance segmentation inference.
It encapsulates model loading, preprocessing, BPU inference, and postprocessing
(including mask generation and NMS).
"""

import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

logger = logging.getLogger("YOLO26_Seg")


@dataclass
class YOLO26SegConfig:
    """Configuration for YOLO26 Segmentation Model.

    Attributes:
        model_path (str): Path to the .bin model file.
        classes_num (int): Number of object classes. Default is 80 (COCO).
        score_thres (float): Confidence threshold for filtering detections.
        nms_thres (float): IoU threshold for Non-Maximum Suppression.
        strides (List[int]): List of strides for the feature maps.
    """
    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.7
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26Seg:
    """YOLO26 Instance Segmentation Model Wrapper.

    This class handles the complete inference pipeline:
    1. Model Loading
    2. Preprocessing (Resize, Pad, Color Conversion)
    3. Inference (BPU Forward)
    4. Postprocessing (Box Decoding, Mask Proto-mask combination, NMS)
    """

    def __init__(self, config: YOLO26SegConfig):
        """Initialize the YOLO26 segmentation model.

        Args:
            config (YOLO26SegConfig): Configuration object.
        """
        self.cfg = config
        self.conf_raw = -np.log(1 / self.cfg.score_thres - 1)

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
        if shape[3] == 3:  # NHWC
            self.m_h, self.m_w = shape[1], shape[2]
        else:  # NCHW
            self.m_h, self.m_w = shape[2], shape[3]
        
        self.proto_h, self.proto_w = self.m_h // 4, self.m_w // 4

        # Pre-compute Grids
        logger.info("Pre-computing Anchor-Free Grids...")
        self.grids = {}
        for s in self.cfg.strides:
            grid_h, grid_w = self.m_h // s, self.m_w // s
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5
        
        # Init scale params
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
        """Process model outputs to generate segmentation results.

        Args:
            outputs: Raw outputs from the model.

        Returns:
            List[Dict]: List of detection dicts containing:
                - 'box': Bounding box [x1, y1, x2, y2]
                - 'score': Confidence score
                - 'id': Class ID
                - 'mask': Cropped mask area (float32)
        """
        t0 = time.time()
        feats = {}
        protos = None
        
        # Parse outputs
        for out in outputs:
            h, w, c = out.properties.shape[1], out.properties.shape[2], out.properties.shape[3]
            if h == self.proto_h and w == self.proto_w and c == 32:
                protos = out.buffer.reshape(self.proto_h, self.proto_w, 32)
                continue
            
            stride = self.m_h // h
            if stride not in self.cfg.strides:
                continue
            
            if stride not in feats:
                feats[stride] = {}
            
            if c == 4:
                feats[stride]['box'] = out.buffer.reshape(-1, 4)
            elif c == self.cfg.classes_num:
                feats[stride]['cls'] = out.buffer.reshape(-1, c)
            else:
                feats[stride]['mc'] = out.buffer.reshape(-1, c)

        if protos is None:
            logger.warning("Proto head not found!")
            return []

        # Decode
        dets = []
        for stride, f in feats.items():
            if not all(k in f for k in ['box', 'cls', 'mc']):
                continue
            
            max_scores = np.max(f['cls'], axis=1)
            mask = max_scores >= self.conf_raw
            if not np.any(mask):
                continue
            
            grid = self.grids[stride][mask]
            v_box = f['box'][mask]
            v_score = 1 / (1 + np.exp(-max_scores[mask]))
            v_id = np.argmax(f['cls'][mask], axis=1)
            v_mc = f['mc'][mask]
            
            xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
            
            for b, s, i, m in zip(xyxy, v_score, v_id, v_mc):
                dets.append({'box': b, 'score': s, 'id': i, 'mc': m})

        # NMS and Mask Processing
        final_res = []
        if dets:
            boxes = np.array([d['box'] for d in dets])
            scores = np.array([d['score'] for d in dets])
            xywh = boxes.copy()
            xywh[:, 2:] -= xywh[:, :2]
            
            indices = cv2.dnn.NMSBoxes(
                xywh.tolist(), scores.tolist(), 
                self.cfg.score_thres, self.cfg.nms_thres
            )
            
            scale_p = self.proto_h / self.m_h
            
            if len(indices) > 0:
                for i in indices.flatten():
                    d = dets[i]
                    # Process mask
                    mask_prob = 1 / (1 + np.exp(-np.dot(protos, d['mc'])))
                    px1 = int(max(0, d['box'][0] * scale_p))
                    py1 = int(max(0, d['box'][1] * scale_p))
                    px2 = int(min(self.proto_w, d['box'][2] * scale_p))
                    py2 = int(min(self.proto_h, d['box'][3] * scale_p))
                    
                    if px2 > px1 and py2 > py1:
                        mask_crop = mask_prob[py1:py2, px1:px2]
                    else:
                        mask_crop = np.zeros((0, 0), np.float32)

                    final_res.append({
                        'box': np.clip(d['box'] / self.scale, 0, [self.orig_w, self.orig_h, self.orig_w, self.orig_h]).astype(int),
                        'score': d['score'],
                        'id': d['id'],
                        'mask': mask_crop
                    })

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res