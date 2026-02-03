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

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess_utils.py
#   utils/py_utils/postprocess_utils.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils


@dataclass
class EfficientNetConfig:
    """Configuration for initializing the EfficientNet model.

    This dataclass contains the model path and all runtime parameters required
    for preprocessing and postprocessing.

    Attributes:
        model_path: Path to the compiled EfficientNet `.hbm` model.
        num_classes: Number of output classes. Default: 1000.
        topk: Number of top predictions to return. Default: 5.
        resize_type: Image resize mode used during preprocessing.
            0: Stretch resize.
            1: Keep aspect ratio with padding (letterbox). Default: 1.
    """
    model_path: str
    num_classes: int = 1000
    topk: int = 5
    resize_type: int = 1


class EfficientNet:
    """EfficientNet image classification wrapper using HB_HBMRuntime.

    This class provides a unified inference pipeline for EfficientNet, including
    input preprocessing, model execution, and postprocessing (Softmax + TopK).
    """

    def __init__(self, config: EfficientNetConfig):
        """Initialize the EfficientNet model with the given configuration.

        Args:
            config: Configuration object containing model path and parameters.
        """
        self.cfg = config

        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        # Shape format: (N, H, W, C) for typical BPU models input
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

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
        """Preprocess an input image into model-required tensor format.

        The input image is resized according to the specified resize strategy
        and converted from BGR format to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            resize_type: Resize strategy override. If `None`, the value from
                the configuration is used.
            image_format: Input image format. Currently, only `"BGR"` is
                supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if resize_type is None:
            resize_type = self.cfg.resize_type

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
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw model outputs into final classification results.

        Performs Softmax and extracts Top-K probabilities and indices.

        Args:
            outputs: Raw output tensors from inference.
            topk: Number of top predictions to return. If `None`, the value
                from the configuration is used.

        Returns:
            A tuple containing:
                - topk_probs: Top-K probabilities with shape `(K,)`.
                - topk_indices: Top-K class indices with shape `(K,)`.
        """
        if topk is None:
            topk = self.cfg.topk

        # Get the first output tensor (classification logits)
        raw_output = outputs[self.model_name][self.output_names[0]]

        # Flatten to (Classes,)
        logits = raw_output.flatten()

        # Compute Softmax with stability adjustment
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Get Top-K indices
        # argsort sorts ascending, so take last k and reverse
        topk_indices = np.argsort(probabilities)[-topk:][::-1]
        topk_probs = probabilities[topk_indices]

        return topk_probs, topk_indices

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                topk: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete classification pipeline on a single image.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            topk: Top-K override.

        Returns:
            A tuple containing:
                - topk_probs: Top-K probabilities.
                - topk_indices: Top-K class indices.
        """
        # 1) Preprocess
        input_tensor = self.pre_process(img, resize_type, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        return self.post_process(outputs, topk)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 topk: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Callable interface for the classification pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            topk: Top-K override.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format, resize_type, topk)
