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

"""YOLO26 Pose Estimation Model Implementation.

This module provides the YOLO26Pose class for pose estimation inference.
It encapsulates model loading, preprocessing, BPU inference, and postprocessing
(including keypoint decoding and NMS).
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

logger = logging.getLogger("YOLO26_Pose")


@dataclass
class YOLO26PoseConfig:
    """Configuration for YOLO26 Pose Model.

    Attributes:
        model_path (str): Path to the .bin model file.
        score_thres (float): Confidence threshold for filtering detections.
        nms_thres (float): IoU threshold for Non-Maximum Suppression.
        kpt_conf_thres (float): Threshold for keypoint visibility.
        strides (List[int]): List of strides for the feature maps.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.7
    kpt_conf_thres: float = 0.5
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26Pose:
    """YOLO26 Pose Estimation Model Wrapper.

    This class handles the complete inference pipeline:
    1. Model Loading
    2. Preprocessing (Resize, Pad, Color Conversion)
    3. Inference (BPU Forward)
    4. Postprocessing (Box/Keypoint Decoding, NMS)
    """

    def __init__(self, config: YOLO26PoseConfig):
        """Initialize the YOLO26 pose estimation model.

        Args:
            config (YOLO26PoseConfig): Configuration object.
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
        """Process model outputs to generate pose results.

        Args:
            outputs: Raw outputs from the model.

        Returns:
            List[Dict]: List of pose dicts containing:
                - 'box': Bounding box [x1, y1, x2, y2]
                - 'score': Confidence score
                - 'kpts': Keypoints array (K, 3) where last dim is (x, y, conf)
        """
        t0 = time.time()
        features = {}
        for out in outputs:
            h, w, c = out.properties.shape[1], out.properties.shape[2], out.properties.shape[3]
            stride = self.m_h // h
            
            if stride not in self.cfg.strides:
                continue
            if stride not in features:
                features[stride] = {}
            
            if c == 4:
                features[stride]['box'] = out.buffer.reshape(-1, 4)
            elif c == 51:
                features[stride]['kpt'] = out.buffer.reshape(-1, 17, 3)
            else:
                features[stride]['cls'] = out.buffer.reshape(-1, c)

        detections = []
        for stride, feats in features.items():
            if not all(k in feats for k in ['box', 'cls', 'kpt']):
                continue
            
            box_data, cls_data, kpt_data = feats['box'], feats['cls'], feats['kpt']
            
            if cls_data.shape[1] == 1:
                scores = cls_data[:, 0]
            else:
                scores = np.max(cls_data, axis=1)

            mask = scores >= self.conf_raw
            if not np.any(mask):
                continue

            v_scores = 1 / (1 + np.exp(-scores[mask]))
            v_box = box_data[mask]
            v_kpts = kpt_data[mask]
            grid = self.grids[stride][mask]
            
            xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
            
            # Keypoints decoding
            kpt_xy = (v_kpts[:, :, :2] + grid[:, None, :]) * stride
            kpt_conf = 1 / (1 + np.exp(-v_kpts[:, :, 2:3]))
            decoded_kpts = np.concatenate([kpt_xy, kpt_conf], axis=-1)

            for box, score, kpts in zip(xyxy, v_scores, decoded_kpts):
                detections.append({'box': box, 'score': score, 'kpts': kpts})

        final_res = []
        if detections:
            boxes = np.array([d['box'] for d in detections])
            scores = np.array([d['score'] for d in detections])
            xywh = boxes.copy()
            xywh[:, 2:] -= xywh[:, :2]
            
            indices = cv2.dnn.NMSBoxes(
                xywh.tolist(), scores.tolist(), 
                self.cfg.score_thres, self.cfg.nms_thres
            )
            
            if len(indices) > 0:
                for i in indices.flatten():
                    det = detections[i]
                    x1 = max(0, min(self.orig_w, det['box'][0] / self.scale))
                    y1 = max(0, min(self.orig_h, det['box'][1] / self.scale))
                    x2 = max(0, min(self.orig_w, det['box'][2] / self.scale))
                    y2 = max(0, min(self.orig_h, det['box'][3] / self.scale))
                    
                    kpts = det['kpts'].copy()
                    kpts[:, 0] /= self.scale
                    kpts[:, 1] /= self.scale
                    
                    final_res.append({
                        'box': np.array([x1, y1, x2, y2], dtype=int),
                        'score': det['score'],
                        'kpts': kpts
                    })
        
        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return final_res

    def predict(self, img: np.ndarray) -> List[Dict]:
        """End-to-end prediction pipeline.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Dict]: List of pose dicts containing:
                - 'box': Bounding box [x1, y1, x2, y2]
                - 'score': Confidence score
                - 'kpts': Keypoints array (K, 3) where last dim is (x, y, conf)
        """
        nv12 = self.pre_process(img)
        outputs = self.forward(nv12)
        results = self.post_process(outputs)
        return results

    def __call__(self, img: np.ndarray) -> List[Dict]:
        """Callable interface for prediction.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Dict]: List of pose dicts containing:
                - 'box': Bounding box [x1, y1, x2, y2]
                - 'score': Confidence score
                - 'kpts': Keypoints array (K, 3) where last dim is (x, y, conf)
        """
        return self.predict(img)