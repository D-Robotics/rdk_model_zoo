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

"""YOLO26 Detection Model Implementation.

This module provides the YOLO26Detect class for performing object detection
inference using the pyeasy_dnn backend. It encapsulates model loading,
preprocessing (Letterbox -> NV12), forward inference, and postprocessing
(Anchor-Free decoding + NMS).
"""

import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# Import hobot_dnn, handling potential import errors gracefully in main scope if needed,
# but here we assume it's available as per the sample requirements.
try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

# Configure logger
logger = logging.getLogger("YOLO26")


@dataclass
class YOLO26Config:
    """Configuration for YOLO26 Detection Model.

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


class YOLO26Detect:
    """YOLO26 Object Detection Model Wrapper.

    This class handles the complete inference pipeline:
    1. Model Loading
    2. Preprocessing (Resize, Pad, Color Conversion)
    3. Inference (BPU Forward)
    4. Postprocessing (Decoding, NMS, Coordinate Rescaling)
    """

    def __init__(self, config: YOLO26Config):
        """Initialize the YOLO26 detector.

        Args:
            config (YOLO26Config): Configuration object containing model path
                and inference parameters.
        """
        self.cfg = config
        self.conf_raw = -np.log(1 / self.cfg.score_thres - 1)

        # Load Model
        try:
            t0 = time.time()
            self.model = dnn.load(self.cfg.model_path)[0]
            logger.info(f"\033[1;31mLoad D-Robotics Quantize model time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

        # Get Input Shape (NCHW or NHWC)
        shape = self.model.inputs[0].properties.shape
        if shape[3] == 3:  # NHWC
            self.m_h, self.m_w = shape[1], shape[2]
        else:  # NCHW
            self.m_h, self.m_w = shape[2], shape[3]

        # Pre-compute Anchor-Free Grids
        logger.info("Pre-computing Anchor-Free Grids...")
        self.grids = {}
        for s in self.cfg.strides:
            grid_h, grid_w = self.m_h // s, self.m_w // s
            # np.indices returns (y_grid, x_grid), we need (x, y) order, so [::-1]
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5

        # Initialize preprocessing state variables
        self.scale = 1.0
        self.x_shift = 0
        self.y_shift = 0
        self.orig_h = 0
        self.orig_w = 0

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Performs letterbox resizing (padding) and converts BGR to NV12.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Flattened NV12 data ready for inference.
        """
        t0 = time.time()
        self.orig_h, self.orig_w = img.shape[:2]
        
        # Calculate scaling and padding
        self.scale = min(self.m_h / self.orig_h, self.m_w / self.orig_w)
        nw, nh = int(self.orig_w * self.scale), int(self.orig_h * self.scale)
        self.x_shift, self.y_shift = (self.m_w - nw) // 2, (self.m_h - nh) // 2

        # Resize and Pad
        resized_img = cv2.resize(img, (nw, nh))
        input_tensor = cv2.copyMakeBorder(
            resized_img,
            self.y_shift, self.m_h - nh - self.y_shift,
            self.x_shift, self.m_w - nw - self.x_shift,
            cv2.BORDER_CONSTANT, value=127
        )

        # Convert to NV12
        yuv = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.empty((self.m_h * self.m_w * 3 // 2,), dtype=np.uint8)
        y_size = self.m_h * self.m_w
        nv12[:y_size] = yuv[:y_size]
        nv12[y_size::2] = yuv[y_size:y_size + y_size // 4]
        nv12[y_size + 1::2] = yuv[y_size + y_size // 4:]

        logger.info(f"\033[1;31mpre process(Letterbox -> NV12) time = {1000 * (time.time() - t0):.2f} ms\033[0m")
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
        logger.info(f"\033[1;31mforward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return out

    def post_process(self, outputs) -> List[Tuple[int, float, int, int, int, int]]:
        """Process model outputs to generate detection results.

        Performs decoding, filtering, and NMS.

        Args:
            outputs: Raw outputs from the model.

        Returns:
            List[Tuple]: Detected objects formatted as (class_id, score, x1, y1, x2, y2).
        """
        t0 = time.time()
        
        # Check output shape assumption
        if outputs[0].buffer.shape[-1] != 4:
            logger.warning("⚠️ Model output shape mismatch! Expected last dim to be 4 (bbox).")

        dets = []
        # Group outputs: Box [0, 2, 4], Cls [1, 3, 5] corresponding to strides 8, 16, 32
        # Note: This assumes specific output order from the model export.
        bboxes = [outputs[i].buffer.reshape(-1, 4) for i in [0, 2, 4]]
        clses = [outputs[i].buffer.reshape(-1, self.cfg.classes_num) for i in [1, 3, 5]]

        for box_data, cls_data, stride in zip(bboxes, clses, self.cfg.strides):
            max_scores = np.max(cls_data, axis=1)
            mask = max_scores >= self.conf_raw
            if not np.any(mask):
                continue

            # Decode boxes
            grid = self.grids[stride][mask]
            v_box = box_data[mask]
            v_score = 1 / (1 + np.exp(-max_scores[mask]))
            v_id = np.argmax(cls_data[mask], axis=1)

            # xyxy calculation: (grid +/- box) * stride
            xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
            
            # Append to detections
            dets.extend(np.hstack([xyxy, v_score[:, None], v_id[:, None]]))

        final_res = []
        if dets:
            dets = np.array(dets)
            # Apply NMS per class
            for i in np.unique(dets[:, 5]):
                cls_dets = dets[dets[:, 5] == i]
                # Convert xyxy to xywh for NMS
                xywh = cls_dets[:, :4].copy()
                xywh[:, 2:] -= xywh[:, :2]
                
                indices = cv2.dnn.NMSBoxes(
                    xywh.tolist(), 
                    cls_dets[:, 4].tolist(), 
                    self.cfg.score_thres, 
                    self.cfg.nms_thres
                )

                if len(indices) > 0:
                    for idx in indices.flatten():
                        d = cls_dets[idx]
                        # Rescale coords back to original image
                        x1, y1, x2, y2 = (d[:4] - [self.x_shift, self.y_shift, self.x_shift, self.y_shift]) / self.scale
                        
                        final_res.append((
                            int(d[5]),  # class_id
                            d[4],       # score
                            int(np.clip(x1, 0, self.orig_w)),
                            int(np.clip(y1, 0, self.orig_h)),
                            int(np.clip(x2, 0, self.orig_w)),
                            int(np.clip(y2, 0, self.orig_h))
                        ))

    def predict(self, img: np.ndarray) -> List[Tuple[int, float, int, int, int, int]]:
        """End-to-end prediction pipeline.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Tuple]: Detected objects formatted as (class_id, score, x1, y1, x2, y2).
        """
        nv12 = self.pre_process(img)
        outputs = self.forward(nv12)
        results = self.post_process(outputs)
        return results

    def __call__(self, img: np.ndarray) -> List[Tuple[int, float, int, int, int, int]]:
        """Callable interface for prediction.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Tuple]: Detected objects formatted as (class_id, score, x1, y1, x2, y2).
        """
        return self.predict(img)
