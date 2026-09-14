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
FasterNet inference module.

This module implements the standardized FasterNet image classification
pipeline for `RDK X5`, including preprocessing, BPU execution, and Top-K
classification post-processing.

Key Features:
    - Load quantized `.bin` models with `hbm_runtime`.
    - Convert BGR images to packed NV12 tensors required by the runtime.
    - Run ImageNet-1k classification and return Top-K results.
    - Keep the runtime interface aligned with other classification samples.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np
from scipy.special import softmax

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.preprocess as pre_utils


@dataclass
class FasterNetConfig:
    """
    Configuration for FasterNet classification inference.

    Args:
        model_path: Path to the compiled `.bin` model file.
        label_file: Path to the ImageNet label file used for result decoding.
        resize_type: Resize strategy used during preprocessing.
        topk: Number of Top-K classes to return.
    """

    model_path: str
    label_file: Optional[str] = None
    resize_type: int = 1
    topk: int = 5


class FasterNet:
    """
    FasterNet classification wrapper based on `hbm_runtime`.

    This class exposes the standard pipeline required by this repository:
    `pre_process`, `forward`, `post_process`, `predict`, and `__call__`.
    The runtime assumes a single packed-NV12 input and a single logits output.
    """

    def __init__(self, config: FasterNetConfig):
        """
        Initialize the runtime wrapper and parse model metadata.

        Args:
            config: Runtime configuration including model path and inference
                parameters.
        """

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.input_h = self.input_shapes[self.input_names[0]][2]
        self.input_w = self.input_shapes[self.input_names[0]][3]
        self.labels = file_io.load_imagenet_labels(config.label_file) if config.label_file else {}

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """
        Set optional runtime scheduling parameters.

        Args:
            priority: Scheduling priority in the range `0~255`.
            bpu_cores: Optional list of BPU core indexes used for inference.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert one BGR image into the packed NV12 tensor expected by the model.

        Args:
            image: Input image in OpenCV BGR format.
            resize_type: Optional override for the preprocessing resize strategy.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.
        """

        resize_type = resize_type if resize_type is not None else self.cfg.resize_type
        resize_img = pre_utils.resized_image(
            image,
            self.input_w,
            self.input_h,
            resize_type,
            interpolation=cv2.INTER_LINEAR,
        )
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0).reshape(
            (1, self.input_h * 3 // 2, self.input_w, 1)
        )
        return {self.model_name: {self.input_names[0]: nv12.astype(np.uint8)}}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Execute one forward pass on BPU.

        Args:
            inputs: Prepared input tensors returned by `pre_process()`.

        Returns:
            Raw model outputs keyed by output tensor name.
        """

        return self.model.run(inputs)[self.model_name]

    def post_process(
        self,
        outputs: Dict[str, np.ndarray],
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert the raw logits tensor into Top-K classification results.

        Args:
            outputs: Raw runtime outputs from `forward()`.
            topk: Optional override for the Top-K result count.

        Returns:
            A tuple of `(topk_idx, topk_prob, topk_labels)`.
        """

        topk = topk or self.cfg.topk
        prob = softmax(np.squeeze(outputs[self.output_names[0]]))
        topk_idx = np.argsort(prob)[-topk:][::-1]
        topk_prob = prob[topk_idx]
        topk_labels = [self.labels.get(int(idx), str(int(idx))) for idx in topk_idx]
        return topk_idx, topk_prob, topk_labels

    def predict(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run the complete FasterNet inference pipeline on one image.

        Args:
            image: Input image in BGR format.
            resize_type: Optional override for preprocessing resize strategy.
            topk: Optional override for Top-K result count.

        Returns:
            The Top-K class IDs, probabilities, and labels produced by
            `post_process()`.
        """

        s1 = time.perf_counter()
        inputs = self.pre_process(image, resize_type)
        t1 = (time.perf_counter() - s1) * 1000

        s2 = time.perf_counter()
        outputs = self.forward(inputs)
        t2 = (time.perf_counter() - s2) * 1000

        s3 = time.perf_counter()
        results = self.post_process(outputs, topk)
        t3 = (time.perf_counter() - s3) * 1000

        print(f"\n[Log] Pre-process: {t1:.2f} ms | Inference: {t2:.2f} ms | Post-process: {t3:.2f} ms")
        return results

    def __call__(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Provide functional-style access to `predict()`.

        Args:
            image: Input image in BGR format.
            resize_type: Optional override for preprocessing resize strategy.
            topk: Optional override for Top-K result count.

        Returns:
            The same result tuple returned by `predict()`.
        """

        return self.predict(image, resize_type, topk)
