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

"""ResNet50 image classification runtime wrapper.

This module provides a lightweight ResNet50 inference wrapper built on
HB_HBMRuntime. It defines model configuration and implements a complete
classification pipeline, including preprocessing, inference, and top-K
postprocessing utilities.

Key Features:
    - ResNet50 HBM model loading and runtime execution
    - NV12-based image preprocessing
    - Top-K classification result extraction
    - Optional runtime scheduling configuration
"""

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.visualize as visualize


@dataclass
class Resnet50Config:
    """Configuration for initializing a ResNet50 classification model.

    Attributes:
        model_path: Path to the compiled ResNet50 `.hbm` model.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding.
    """
    model_path: str = "../../model/s100/resnet50_224x224_nv12.hbm"
    resize_type: int = 1


class Resnet50:
    """ResNet50 image classification wrapper based on HB_HBMRuntime."""

    def __init__(self, config: Resnet50Config):
        """Initialize the ResNet50 model with the given configuration.

        Args:
            config: Configuration object that specifies the model path and
                preprocessing options.
        """
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.
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

        Args:
            img: Input image array.
            resize_type: Resize strategy override.
            image_format: Input image format. Currently only "BGR" is supported.

        Returns:
            A nested dictionary: {model_name: {input_name: input_tensor}}.
        """
        if resize_type is None:
            resize_type = self.cfg.resize_type
        else:
            self.cfg.resize_type = resize_type

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

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary.

        Returns:
            A nested dictionary containing raw output tensors returned by the runtime.
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
            A list of (class_id, probability) tuples sorted by confidence.
        """
        return visualize.get_topk_predictions(outputs[self.model_name][self.output_names[0]][0], topk)

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                topk: Optional[int] = None
                ) -> List[Tuple[int, float]]:
        """Run the complete classification pipeline on a single image.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            topk: Number of top classes to return.

        Returns:
            A list of (class_id, probability) tuples.
        """
        input_tensor = self.pre_process(img, resize_type, image_format)
        outputs = self.forward(input_tensor)
        cls_results = self.post_process(outputs, topk)
        return cls_results

    def __call__(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                topk: Optional[int] = None
                ) -> List[Tuple[int, float]]:
        """Callable interface for image classification.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            topk: Number of top classes to return.

        Returns:
            Same return value as `predict()`.
        """
        return self.predict(img, image_format, resize_type, topk)
