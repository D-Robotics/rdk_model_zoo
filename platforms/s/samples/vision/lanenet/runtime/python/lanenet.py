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

"""LaneNet lane detection inference wrapper and pipeline utilities.

This module defines a LaneNet runtime wrapper built on HBM runtime.
It includes a configuration dataclass and a complete inference pipeline
(preprocess, forward, postprocess) for lane segmentation tasks,
including instance segmentation and binary mask generation.

Key Features:
    - Float32 RGB preprocessing with ImageNet normalization
    - Instance segmentation and binary segmentation mask output
    - Supports RDK S100 platform

Typical Usage:
    >>> from lanenet import LaneNet, LaneNetConfig
    >>> config = LaneNetConfig(model_path='/opt/hobot/model/s100/basic/lanenet256x512.hbm')
    >>> model = LaneNet(config)
    >>> instance_pred, binary_pred = model.predict(img)

Notes:
    - This model only supports RDK S100 platform.
    - RDK S600 users: the S100 model is not compatible with S600 BPU. Refer to
      README.md for platform compatibility details.
"""

import os
import cv2
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io


@dataclass
class LaneNetConfig:
    """Configuration for initializing the LaneNet model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing and inference in the LaneNet pipeline.

    Attributes:
        model_path: Path to the compiled LaneNet `.hbm` model.

    Notes:
        - This model only supports RDK S100 platform.
        - The default model path targets the S100 platform. RDK S600 is not
          supported; refer to the README for details.
    """
    model_path: str = '/opt/hobot/model/s100/basic/lanenet256x512.hbm'


class LaneNet:
    """LaneNet lane detection wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for LaneNet, including
    input preprocessing, model execution, and postprocessing steps to generate
    instance segmentation and binary segmentation masks.

    Args:
        config: Configuration object containing the model path. All field
            semantics and constraints are defined in the `LaneNetConfig`
            dataclass.

    Attributes:
        model: Loaded HBM runtime model handle.
        model_name: Name of the loaded model.
        input_names: List of input tensor names.
        output_names: List of output tensor names.
        input_shapes: Dictionary of input tensor shapes.
        output_quants: Output quantization parameters.
        input_h: Model input height.
        input_w: Model input width.
        cfg: LaneNetConfig object with runtime parameters.

    Notes:
        - This model only supports RDK S100. Running on S600 is not supported.
    """

    def __init__(self, config: LaneNetConfig) -> None:
        """Initialize the LaneNet model with the given configuration.

        Args:
            config: Configuration object containing the model path. All field
                semantics and constraints are defined in the `LaneNetConfig`
                dataclass.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution (H, W) inferred from model metadata
        # Shape format is (N, C, H, W) for NCHW layout
        self.input_h = self.input_shapes[self.input_names[0]][2]
        self.input_w = self.input_shapes[self.input_names[0]][3]

        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
                Higher values mean higher priority.
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

    def pre_process(self,
                    img: np.ndarray,
                    image_format: str = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is resized to the model input resolution, converted
        from BGR to RGB, normalized with ImageNet mean/std, and formatted
        as a float32 NCHW tensor.

        Args:
            img: Input image array in BGR format (H, W, 3).
            image_format: Input image format. Only `"BGR"` is supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`, ready to be passed to
            `forward()`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}. Only 'BGR' is supported.")

        # Convert BGR to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input resolution
        image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1], then standardize with ImageNet mean/std
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std

        # Add batch dimension: (1, C, H, W)
        image = np.expand_dims(image, axis=0)

        return {
            self.model_name: {
                self.input_names[0]: image
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`, in the form `{model_name: {input_name: tensor}}`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime,
            in the form `{model_name: {output_name: tensor}}`.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw model outputs into lane segmentation masks.

        Decodes the instance segmentation and binary segmentation outputs
        from the model into visualizable uint8 images.

        Args:
            outputs: Raw output tensors from `forward()`, in the form
                `{model_name: {output_name: tensor}}`.

        Returns:
            A tuple containing:
                - instance_pred: Colored instance segmentation mask as a
                  uint8 image of shape `(H, W, 3)`.
                - binary_pred: Binary lane segmentation mask as a uint8
                  image of shape `(H, W)`.
        """
        model_outputs = outputs[self.model_name]

        # Decode instance segmentation output: (3, H, W) -> (H, W, 3) uint8
        instance_pred = model_outputs["instance_seg_logits"].reshape(
            (3, self.input_h, self.input_w)
        )
        instance_pred = (instance_pred * 255).transpose(1, 2, 0).astype(np.uint8)

        # Decode binary segmentation output: (H, W) uint8
        binary_pred = model_outputs["binary_seg_pred"].reshape(
            (self.input_h, self.input_w)
        )
        binary_pred = (binary_pred * 255).astype(np.uint8)

        return instance_pred, binary_pred

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR"
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete lane detection pipeline on a single image.

        This method internally performs preprocessing, inference, and
        postprocessing.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently supports `"BGR"`.

        Returns:
            A tuple containing:
                - instance_pred: Colored instance segmentation mask, shape
                  `(H, W, 3)`, dtype uint8.
                - binary_pred: Binary lane segmentation mask, shape `(H, W)`,
                  dtype uint8.
        """
        # 1) Preprocess
        input_tensor = self.pre_process(img, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        instance_pred, binary_pred = self.post_process(outputs)

        return instance_pred, binary_pred

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR"
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Callable interface for the lane detection pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format)
