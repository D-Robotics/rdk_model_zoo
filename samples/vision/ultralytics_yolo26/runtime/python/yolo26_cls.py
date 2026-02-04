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

"""YOLO26 Classification Model Implementation.

This module provides the YOLO26Cls class for image classification.
It encapsulates model loading, preprocessing, BPU inference, and postprocessing.
"""

import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

logger = logging.getLogger("YOLO26_Cls")


@dataclass
class YOLO26ClsConfig:
    """Configuration for YOLO26 Classification Model.

    Attributes:
        model_path (str): Path to the .bin model file.
        topk (int): Number of top predictions to return.
    """
    model_path: str
    topk: int = 5


class YOLO26Cls:
    """YOLO26 Classification Model Wrapper.

    This class handles the complete inference pipeline:
    1. Model Loading
    2. Preprocessing (Resize, NV12 Conversion)
    3. Inference (BPU Forward)
    4. Postprocessing (Softmax, Top-K Sorting)
    """

    def __init__(self, config: YOLO26ClsConfig):
        """Initialize the YOLO26 classification model.

        Args:
            config (YOLO26ClsConfig): Configuration object.
        """
        self.cfg = config
        
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
            self.ih, self.iw = shape[1], shape[2]
        else:
            self.ih, self.iw = shape[2], shape[3]

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Flattened NV12 data ready for inference.
        """
        t0 = time.time()
        # Resize directly for classification
        yuv = cv2.cvtColor(cv2.resize(img, (self.iw, self.ih)), cv2.COLOR_BGR2YUV_I420).flatten()
        
        nv12 = np.empty((self.ih * self.iw * 3 // 2,), dtype=np.uint8)
        y_size = self.ih * self.iw
        
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

    def post_process(self, outputs) -> List[Tuple[int, float]]:
        """Process model outputs to generate classification results.

        Args:
            outputs: Raw outputs from the model.

        Returns:
            List[Tuple[int, float]]: List of top-K results as (class_id, score).
        """
        t0 = time.time()
        logits = outputs[0].buffer.reshape(-1)
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()
        
        top_indices = np.argsort(probs)[::-1][:self.cfg.topk]
        results = list(zip(top_indices, probs[top_indices]))
        
        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return results

    def predict(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """End-to-end prediction pipeline.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Tuple[int, float]]: List of top-K results as (class_id, score).
        """
        nv12 = self.pre_process(img)
        outputs = self.forward(nv12)
        results = self.post_process(outputs)
        return results

    def __call__(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """Callable interface for prediction.

        Args:
            img (np.ndarray): Input BGR image.

        Returns:
            List[Tuple[int, float]]: List of top-K results as (class_id, score).
        """
        return self.predict(img)