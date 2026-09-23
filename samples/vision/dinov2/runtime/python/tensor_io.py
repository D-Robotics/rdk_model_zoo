# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2-specific BGR to normalized RGB tensor conversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from samples.vision.dinov2.runtime.python.model_binding import ModelBinding

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ImageContext:
    """Realized geometry for one preprocessing call."""

    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    crop_origin: tuple[int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Owned physical tensors and immutable per-call context."""

    tensors: Mapping[str, np.ndarray]
    context: ImageContext


def prepare_image(image: np.ndarray, binding: ModelBinding) -> PreparedInput:
    """Convert BGR uint8 HWC to the fixed DINOv2 float32 NCHW input."""

    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Expected nonempty BGR uint8 image shaped HxWx3.")
    if binding.input_name != "input" or binding.input_shape != (1, 3, 224, 224):
        raise ValueError("DINOv2 binding does not describe the fixed input contract.")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if height <= width:
        resized_height = RESIZE_SIZE
        resized_width = int(RESIZE_SIZE * width / height)
    else:
        resized_height = int(RESIZE_SIZE * height / width)
        resized_width = RESIZE_SIZE
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    crop_y = (resized_height - IMAGE_SIZE) // 2
    crop_x = (resized_width - IMAGE_SIZE) // 2
    cropped = resized[crop_y:crop_y + IMAGE_SIZE, crop_x:crop_x + IMAGE_SIZE]
    tensor = np.transpose(cropped, (2, 0, 1))[None].astype(np.float32)
    tensor = tensor / 255.0
    tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    tensor = np.ascontiguousarray(tensor)
    if tensor.shape != binding.input_shape or tensor.dtype != np.float32:
        raise ValueError(f"Preprocessed input is {tensor.shape}/{tensor.dtype}, expected fixed F32 input.")
    return PreparedInput(
        tensors={binding.input_name: tensor},
        context=ImageContext((height, width), (resized_height, resized_width), (crop_y, crop_x)),
    )


__all__ = ["ImageContext", "PreparedInput", "prepare_image"]
