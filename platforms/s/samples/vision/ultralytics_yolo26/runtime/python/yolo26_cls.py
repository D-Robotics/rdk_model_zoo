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

# flake8: noqa: E501
# flake8: noqa: E402

"""Provide YOLO26 Classification inference wrapper and pipeline utilities.

This module defines a YOLO26 classification runtime wrapper built on HBM
runtime. It handles NV12 preprocessing, BPU inference, and Top-K
postprocessing.

Model output: single tensor with classification logits/probs.
"""

import os
import sys
import hbm_runtime
import numpy as np
from scipy.special import softmax
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YOLO26ClsConfig:
    """Configuration for the YOLO26 Classification model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        topk: Number of top classes to return.
        resize_type: Image resize strategy (0=stretch, 1=letterbox).
    """
    model_path: str
    topk: int = 5
    resize_type: int = 0


class YOLO26Cls:
    """YOLO26 classification wrapper based on HB_HBMRuntime."""

    def __init__(self, config: YOLO26ClsConfig):
        """Initialize the YOLO26 Classification model.

        Args:
            config: Configuration object.
        """
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

    def set_scheduling_params(self, priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters."""
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into NV12 Y/UV tensor format.

        Args:
            img: Input image array.
            image_format: Input image format. Only "BGR" is supported.

        Returns:
            Nested input tensor dictionary.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        resized_img = pre_utils.resized_image(img, self.input_w, self.input_h, self.cfg.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized_img)

        if len(self.input_names) != 2:
            raise ValueError(
                f"YOLO26 classification expects fixed NV12 Y/UV inputs, got {len(self.input_names)} inputs.")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """Execute model inference."""
        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict,
                     topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Process raw logits to get Top-K classification results.

        Args:
            outputs: Raw output tensors.
            topk: Number of top results to return.

        Returns:
            List of `(class_id, probability)` sorted by probability descending.
        """
        topk = topk if topk is not None else self.cfg.topk

        raw_output = outputs[self.model_name]
        logits = raw_output[self.output_names[0]].reshape(-1)
        probs = softmax(logits)

        top_indices = np.argsort(probs)[::-1][:topk]
        return [(int(idx), float(probs[idx])) for idx in top_indices]

    def predict(self, img: np.ndarray, image_format: str = "BGR",
                topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Run the complete classification pipeline.

        Args:
            img: Input image.
            image_format: Input format (default "BGR").
            topk: Top-K override.

        Returns:
            List of (class_id, probability) tuples.
        """
        inp = self.pre_process(img, image_format)
        out = self.forward(inp)
        return self.post_process(out, topk)

    def __call__(self, img: np.ndarray, **kwargs) -> List[Tuple[int, float]]:
        """Callable interface equivalent to predict()."""
        return self.predict(img, **kwargs)
