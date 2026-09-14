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
MODNet Inference Module.

This module implements the MODNet portrait matting pipeline on BPU,
including pre-processing, forward execution, and alpha matte generation.

MODNet (Mobile-friendly One-stage Deep Image Matting Network) takes a
single RGB image and directly predicts an alpha matte without requiring
a trimap. The model uses float32 NCHW RGB input with normalization
to [-1, 1].

Key Features:
    - One-stage end-to-end portrait matting.
    - Float32 NCHW RGB input (no NV12 conversion).
    - Aspect-ratio-preserving resize with symmetric zero-padding.

Typical Usage:
    >>> from modnet import MODNetConfig, MODNet
    >>> config = MODNetConfig(model_path="path/to/modnet.bin")
    >>> model = MODNet(config)
    >>> matte = model.predict(image)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect


logger = logging.getLogger("MODNet")


@dataclass
class MODNetConfig:
    """Configuration for MODNet inference.

    Args:
        model_path: Path to the compiled BIN model file.
        ref_size: Target input resolution (longer side).
    """
    model_path: str
    ref_size: int = 512


class MODNet:
    """MODNet portrait matting wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    It encapsulates the MODNet model, providing a unified inference pipeline
    with aspect-ratio-preserving preprocessing and alpha matte generation.
    """

    def __init__(self, config: MODNetConfig) -> None:
        """Initialize the model and extract runtime metadata.

        Args:
            config: Configuration object containing model path and params.
        """
        t0 = time.time()
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]

        self._pad_x: int = 0
        self._pad_y: int = 0
        self._new_h: int = 0
        self._new_w: int = 0
        self._orig_h: int = 0
        self._orig_w: int = 0

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set BPU scheduling parameters like priority and core affinity.

        Args:
            priority: Scheduling priority (0-255).
            bpu_cores: BPU core indexes to run inference.
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    @staticmethod
    def _resize_with_padding(
        image: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """Resize image with aspect-ratio-preserving scale and symmetric zero-padding.

        Args:
            image: Input image.
            target_size: Target size for the longer side.

        Returns:
            Tuple of (padded_image, pad_x, pad_y, new_w, new_h).
        """
        orig_h, orig_w = image.shape[:2]
        scale = target_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = target_size - new_w
        pad_h = target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_h - pad_top,
            pad_left,
            pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return padded, pad_left, pad_top, new_w, new_h

    def pre_process(
        self, image: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert a BGR image to float32 NCHW RGB input for hbm_runtime.

        The preprocessing pipeline:
        1. BGR -> RGB
        2. Normalize to [-1, 1]: ``(pixel - 127.5) / 127.5``
        3. Resize with padding to ``(ref_size, ref_size)``
        4. HWC -> NCHW, add batch dimension

        Args:
            image: Input image in BGR format.

        Returns:
            Prepared input tensors for hbm_runtime.run().

        Raises:
            ValueError: If input image is None.
        """
        if image is None:
            raise ValueError("Input image is None")

        t0 = time.time()
        self._orig_h, self._orig_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5

        padded, pad_x, pad_y, new_w, new_h = self._resize_with_padding(
            normalized, self.cfg.ref_size
        )
        self._pad_x = pad_x
        self._pad_y = pad_y
        self._new_w = new_w
        self._new_h = new_h

        tensor = np.transpose(padded, (2, 0, 1))[None, :, :, :].astype(np.float32)

        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return {self.model_name: {self.input_names[0]: tensor}}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute inference on BPU using hbm_runtime.

        Args:
            input_tensor: Prepared input tensors.

        Returns:
            Raw output tensors from the runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info(f"\033[1;31mForward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Convert raw model output to an alpha matte in original image space.

        Args:
            outputs: Raw model outputs.

        Returns:
            Alpha matte as a uint8 grayscale image matching the original
            input dimensions.
        """
        t0 = time.time()
        matte = outputs[self.model_name][self.output_names[0]].squeeze()

        matte = (matte * 255).astype(np.uint8)
        matte_unpad = matte[
            self._pad_y : self._pad_y + self._new_h,
            self._pad_x : self._pad_x + self._new_w,
        ]
        matte_final = cv2.resize(
            matte_unpad,
            (self._orig_w, self._orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return matte_final

    def predict(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Run the complete matting pipeline on a single image.

        Args:
            image: Input image in BGR format.

        Returns:
            Alpha matte as a uint8 grayscale image.
        """
        input_tensors = self.pre_process(image)
        outputs = self.forward(input_tensors)
        return self.post_process(outputs)

    def __call__(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Provide functional-style calling capability, equivalent to ``predict()``.

        Args:
            image: Input image in BGR format.

        Returns:
            Alpha matte as a uint8 grayscale image.
        """
        return self.predict(image)
