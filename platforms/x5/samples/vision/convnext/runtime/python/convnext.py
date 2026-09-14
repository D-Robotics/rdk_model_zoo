# Copyright (c) 2026 D-Robotics Corporation
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
ConvNeXt Inference Module.

This module implements the ConvNeXt image classification pipeline on BPU, 
including pre-processing, forward execution, and post-processing.

Key Features:
    - Optimized for RDK BPU architecture.
    - Supports high-performance NV12 input.
    - Efficient ImageNet-1k classification post-processing.

Typical Usage:
    >>> from convnext import ConvNeXtConfig, ConvNeXt
    >>> config = ConvNeXtConfig(model_path="path/to/model.bin")
    >>> model = ConvNeXt(config)
    >>> topk_idx, topk_prob, topk_labels = model.predict(image)
"""

import os
import cv2
import sys
import time
import hbm_runtime
import numpy as np
from scipy.special import softmax
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.preprocess as pre_utils


@dataclass
class ConvNeXtConfig:
    """
    Configuration for ConvNeXt inference.
    
    Args:
        model_path (str): Path to the compiled HBM model file.
        label_file (Optional[str]): Path to the ImageNet class names file.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        topk (int): Number of top results to return.
    """
    model_path: str
    label_file: Optional[str] = None
    resize_type: int = 1
    topk: int = 5


class ConvNeXt:
    """
    ConvNeXt classification wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: ConvNeXtConfig):
        """
        Initializes the model, loads the HBM, and extracts metadata.

        Args:
            config (ConvNeXtConfig): Configuration object containing model path and params.
        """
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        
        # Assume input_0 is the primary image input
        self.input_h = self.input_shapes[self.input_names[0]][2]
        self.input_w = self.input_shapes[self.input_names[0]][3]
        
        # Load labels if provided
        self.labels = file_io.load_imagenet_labels(config.label_file) if config.label_file else {}

    def set_scheduling_params(self, 
                              priority: Optional[int] = None, 
                              bpu_cores: Optional[List[int]] = None) -> None:
        """
        Sets BPU scheduling parameters like priority and core affinity.

        Args:
            priority (Optional[int]): Scheduling priority (0-255).
            bpu_cores (Optional[List[int]]): BPU core indexes to run inference.
        """
        kwargs = {}
        if priority is not None: 
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None: 
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs: 
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, 
                    image: np.ndarray, 
                    resize_type: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Converts input image to the format required by BPU runtime (NV12).

        Args:
            image (np.ndarray): Input image in BGR format (H, W, 3).
            resize_type (Optional[int]): Override default resize strategy.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared input tensors for hbm_runtime.run().
                  Format: {model_name: {input_name: tensor_data}}

        Raises:
            ValueError: If input image is None.
        """
        if image is None:
            raise ValueError("Input image is None")

        resize_type = resize_type if resize_type is not None else self.cfg.resize_type
        
        # Resize and color space conversion
        resize_img = pre_utils.resized_image(
            image, self.input_w, self.input_h, resize_type, interpolation=cv2.INTER_LINEAR
        )
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0).reshape(
            (1, self.input_h * 3 // 2, self.input_w, 1)
        )
        
        return {self.model_name: {self.input_names[0]: nv12.astype(np.uint8)}}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Any:
        """
        Executes inference on BPU using hbm_runtime.

        Args:
            inputs (Dict): Prepared input tensors from pre_process().

        Returns:
            Any: Direct output results from hbm_runtime.run().
        """
        return self.model.run(inputs)[self.model_name]

    def post_process(self, 
                     outputs: Any, 
                     topk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Converts raw BPU output tensors to classification results.

        Args:
            outputs (Any): Raw output tensors from forward().
            topk (Optional[int]): Number of top results to return.

        Returns:
            Tuple: (topk_idx, topk_prob, topk_labels)
                - topk_idx (np.ndarray): Top-K class indices.
                - topk_prob (np.ndarray): Top-K probabilities.
                - topk_labels (List[str]): Top-K label strings.
        """
        topk = topk or self.cfg.topk
        raw_output = outputs[self.output_names[0]]
        
        # Apply Softmax and sort
        prob = softmax(np.squeeze(raw_output))
        topk_idx = np.argsort(prob)[-topk:][::-1]
        topk_prob = prob[topk_idx]
        
        topk_labels = []
        if self.labels:topk_labels = [self.labels.get(int(idx), str(int(idx))) for idx in topk_idx]
            
        return topk_idx, topk_prob, topk_labels

    def predict(self, 
                image: np.ndarray, 
                resize_type: Optional[int] = None, 
                topk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        High-level interface for one-click inference.

        Orchestrates the pipeline: pre_process -> forward -> post_process.

        Args:
            image (np.ndarray): Input image.
            resize_type (Optional[int]): Resize strategy.
            topk (Optional[int]): Top-K results.

        Returns:
            Tuple: Results from post_process().
        """
        # Step A: Pre-processing
        s1 = time.perf_counter()
        input_tensors = self.pre_process(image, resize_type)
        t1 = (time.perf_counter() - s1) * 1000

        # Step B: Forward Execution
        s2 = time.perf_counter()
        raw_outputs = self.forward(input_tensors)
        t2 = (time.perf_counter() - s2) * 1000

        # Step C: Post-processing
        s3 = time.perf_counter()
        results = self.post_process(raw_outputs, topk)
        t3 = (time.perf_counter() - s3) * 1000

        print(f"\n[Log] Pre-process: {t1:.2f} ms | Inference: {t2:.2f} ms | Post-process: {t3:.2f} ms")
        return results

    def __call__(self, 
                 image: np.ndarray, 
                 resize_type: Optional[int] = None, 
                 topk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Provides functional-style calling capability."""
        return self.predict(image, resize_type, topk)
