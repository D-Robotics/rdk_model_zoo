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
YOLO26 Classification Inference Module.

This module implements the YOLO26 image classification pipeline on BPU,
including pre-processing, forward execution, and Top-K post-processing.

Key Features:
    - Optimized for RDK X5 single-input NV12 models.
    - Supports Top-K classification result decoding.
    - Provides a concise wrapper for sample-level image inference.

Typical Usage:
    >>> from yolo26_cls import YOLO26ClsConfig, YOLO26Cls
    >>> config = YOLO26ClsConfig(model_path="path/to/yolo26_cls.bin")
    >>> model = YOLO26Cls(config)
    >>> results = model.predict(image)
"""

import os
import sys
import time
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path to import shared utilities.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils

logger = logging.getLogger("YOLO26_Cls")


@dataclass
class YOLO26ClsConfig:
    """
    Configuration for YOLO26 classification inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        topk (int): Number of top results to return.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
    """
    model_path: str
    topk: int = 5
    resize_type: int = 0


class YOLO26Cls:
    """
    YOLO26 classification wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLO26ClsConfig):
        """
        Initialize the model and extract metadata.

        Args:
            config (YOLO26ClsConfig): Configuration object containing model path and params.
        """
        self.cfg = config

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

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> List[Tuple[int, float]]:
        """
        Convert raw output tensor to Top-K classification results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.

        Returns:
            List[Tuple[int, float]]: Top-K `(class_id, probability)` pairs sorted
            from high to low confidence.
        """
        t0 = time.time()
        raw_outputs = outputs[self.model_name]
        logits = raw_outputs[self.output_names[0]].reshape(-1).astype(np.float32)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()

        top_indices = np.argsort(probs)[::-1][:self.cfg.topk]
        results = [(int(idx), float(probs[idx])) for idx in top_indices]

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return results

    def predict(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """
        Run the complete classification pipeline on a single image.

        This method orchestrates the full workflow:
        - Pre-process the input image.
        - Execute BPU inference.
        - Decode the Top-K classification results.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            List[Tuple[int, float]]: Top-K `(class_id, probability)` pairs.
        """
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs)

    def __call__(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """
        Provide functional-style calling capability.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            List[Tuple[int, float]]: Classification results from `predict()`.
        """
        return self.predict(img)
