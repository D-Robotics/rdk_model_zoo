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

"""Depth Anything V2 monocular depth estimation runtime wrapper.

This module provides a Depth Anything V2 inference wrapper built on
HB_HBMRuntime. The model accepts NCHW RGB float32 input and outputs
a single-channel depth map.

Key Features:
    - Depth Anything V2 HBM model loading and runtime execution
    - NCHW RGB float32 preprocessing (resize + normalize)
    - Depth map postprocessing with bilinear interpolation and normalization
    - Optional runtime scheduling configuration

Typical Usage:
    >>> from depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2
    >>> config = DepthAnythingV2Config(model_path="depth_any.hbm")
    >>> model = DepthAnythingV2(config)
    >>> depth_map = model.predict(img)
"""

import os
import cv2
import sys
import hbm_runtime
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.nn_math as nn_math


@dataclass
class DepthAnythingV2Config:
    """Configuration for initializing the Depth Anything V2 model.

    Attributes:
        model_path: Path to the compiled ``.hbm`` model.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding.
    """
    model_path: str
    resize_type: int = 0


class DepthAnythingV2:
    """Depth Anything V2 monocular depth estimation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for Depth Anything V2,
    including input preprocessing (resize + NCHW RGB normalize), model execution,
    and depth map postprocessing with bilinear interpolation back to original
    resolution.
    """

    def __init__(self, config: DepthAnythingV2Config):
        """Initialize the Depth Anything V2 model with the given configuration.

        Args:
            config: Configuration object that specifies the model path and
                preprocessing options.
        """
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        self.input_h = self.input_shapes[self.input_names[0]][2]
        self.input_w = self.input_shapes[self.input_names[0]][3]

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
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into NCHW RGB float32 tensor format.

        The input image is resized to the model input resolution, converted
        from BGR to RGB, and normalized using ImageNet mean and std.

        Args:
            img: Input image array.
            image_format: Input image format. Currently only "BGR" is supported.

        Returns:
            A nested dictionary in the form:
            ``{model_name: {input_name: input_tensor}}``.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, self.cfg.resize_type)
            rgb_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        # Normalize with ImageNet mean/std, convert to NCHW float32
        pixel_values = nn_math.zscore_normalize_lastdim(rgb_img)
        pixel_values = np.transpose(pixel_values, (2, 0, 1))[np.newaxis].astype(np.float32)

        return {
            self.model_name: {
                self.input_names[0]: pixel_values
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                ``pre_process()``.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_h: int,
                     ori_w: int
                     ) -> np.ndarray:
        """Post-process raw depth output into a normalized depth map.

        Applies bilinear interpolation to resize the depth map back to the
        original image resolution, then normalizes values to [0, 255] range.

        Args:
            outputs: Raw output tensors from inference.
            ori_h: Original image height.
            ori_w: Original image width.

        Returns:
            A uint8 depth map of shape ``(ori_h, ori_w)`` with values in
            [0, 255].
        """
        pred_depth = outputs[self.model_name][self.output_names[0]]
        pred_depth = torch.tensor(pred_depth)

        # Bilinear interpolation to original resolution
        depth = F.interpolate(pred_depth[None], (ori_h, ori_w),
                              mode="bilinear", align_corners=False)[0, 0]

        # Normalize to [0, 255]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().detach().numpy().astype(np.uint8)

        return depth

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR"
                ) -> np.ndarray:
        """Run the complete depth estimation pipeline on a single image.

        Args:
            img: Input image array.
            image_format: Input image format. Currently supports "BGR".

        Returns:
            A uint8 depth map of shape ``(H, W)`` with values in [0, 255].
        """
        ori_h, ori_w = img.shape[:2]

        input_tensor = self.pre_process(img, image_format)
        outputs = self.forward(input_tensor)
        depth_map = self.post_process(outputs, ori_h, ori_w)

        return depth_map

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR"
                 ) -> np.ndarray:
        """Callable interface for depth estimation.

        Args:
            img: Input image array.
            image_format: Input image format.

        Returns:
            Same return value as ``predict()``.
        """
        return self.predict(img, image_format)
