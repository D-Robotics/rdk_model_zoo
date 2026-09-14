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

"""MobileNetV2 image classification runtime wrapper.

This module provides a lightweight MobileNetV2 inference wrapper built on
HB_HBMRuntime. It defines model configuration and implements a complete
classification pipeline, including preprocessing, inference, and top-K
postprocessing utilities.

Key Features:
    - MobileNetV2 HBM model loading and runtime execution
    - NV12-based image preprocessing
    - Top-K classification result extraction
    - Optional runtime scheduling configuration

Typical Usage:
    >>> from mobilenetv2 import MobileNetV2Config, MobileNetV2
    >>> config = MobileNetV2Config(model_path="mobilenetv2_224x224_nv12.hbm")
    >>> model = MobileNetV2(config)
    >>> results = model(img, topk=5)

Notes:
    - Input images must be provided in BGR format (as returned by cv2.imread).
    - Model input format is NV12; conversion is handled internally.
"""

import os
import cv2
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess.py
#   utils/py_utils/postprocess.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.visualize as visualize


@dataclass
class MobileNetV2Config:
    """Configuration for initializing a MobileNetV2 classification model.

    This dataclass defines the configuration parameters required to load
    and run a MobileNetV2 HBM model, including model path and preprocessing options.

    Attributes:
        model_path: Path to the compiled MobileNetV2 `.hbm` model.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding.
    """
    model_path: str
    resize_type: int = 1


class MobileNetV2:
    """MobileNetV2 image classification wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for MobileNetV2, including
    input preprocessing, model execution, and top-K classification postprocessing.
    """

    def __init__(self, config: MobileNetV2Config):
        """Initialize the MobileNetV2 model with the given configuration.

        Args:
            config: Configuration object that specifies the model path and
                preprocessing options.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # Classification and preprocessing configuration
        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.

        Returns:
            None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model input tensors.

        The input image is resized to the model input resolution and converted
        from BGR format to NV12 planes as required by the runtime.

        Args:
            img: Input image array.
            resize_type: Resize strategy override. If None, the value from
                the configuration is used.
            image_format: Input image format. Currently only "BGR" is supported.

        Returns:
            A nested dictionary in the form:
            {model_name: {input_name: input_tensor}}.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if resize_type is None:
            resize_type = self.cfg.resize_type
        else:
            self.cfg.resize_type = resize_type

        # Resize and convert to NV12
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     topk: Optional[int] = None
                     ) -> List[Tuple[int, float]]:
        """Post-process raw outputs into top-K classification results.

        Args:
            outputs: Raw output tensors from inference.
            topk: Number of top classes to return.

        Returns:
            A list of (class_id, probability) tuples sorted by confidence
            in descending order.

        Notes:
            MobileNetV2 output node "prob" already contains post-softmax
            probabilities, so no additional softmax is applied here.
        """
        if topk is None:
            topk = 5

        # Squeeze to 1-D: output shape is [1, 1000, 1, 1] → [1000]
        prob = np.squeeze(outputs[self.model_name][self.output_names[0]])

        # Sort by descending probability and return top-k (no softmax needed)
        topk_idx = np.argsort(prob)[-topk:][::-1]
        return [(int(idx), float(prob[idx])) for idx in topk_idx]

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                topk: Optional[int] = None
                ) -> List[Tuple[int, float]]:
        """Run the complete classification pipeline on a single image.

        This method performs preprocessing, inference, and postprocessing
        internally and returns top-K classification results.

        Args:
            img: Input image array.
            image_format: Input image format. Currently supports "BGR".
            resize_type: Resize strategy override.
            topk: Number of top classes to return.

        Returns:
            A list of (class_id, probability) tuples.
        """
        # 1) Preprocess
        input_tensor = self.pre_process(img, resize_type, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        cls_results = self.post_process(outputs, topk)

        return cls_results

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 topk: Optional[int] = None
                 ) -> List[Tuple[int, float]]:
        """Callable interface for image classification.

        This method is equivalent to calling `predict()`.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            topk: Number of top classes to return.

        Returns:
            Same return value as `predict()`.
        """
        return self.predict(img, image_format, resize_type, topk)
