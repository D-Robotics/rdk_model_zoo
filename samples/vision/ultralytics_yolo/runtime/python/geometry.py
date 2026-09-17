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

"""Image geometry shared by the detector's preprocessing and postprocessing.

The old detector reconstructed letterbox padding from the ideal floating point
scale.  That is subtly wrong whenever the resized dimensions are rounded (and
it is especially visible for rectangular images).  This module records the
dimensions and padding actually used to create the model input and uses that
record for the inverse mapping.

Coordinates in this module are expressed as ``(x1, y1, x2, y2)``.  Image sizes
are expressed as ``(height, width)``.  Padding is ordered
``(left, top, right, bottom)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Size = Tuple[int, int]
Padding = Tuple[int, int, int, int]


def _size(value: Sequence[int], name: str) -> Size:
    """Return a validated ``(height, width)`` pair."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain (height, width).")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain positive dimensions.")
    return height, width


@dataclass(frozen=True)
class ImageTransform:
    """Describe the concrete transform used for one model input.

    Attributes:
        original_size: Source image dimensions as ``(height, width)``.
        model_size: Final model input dimensions as ``(height, width)``.
        resized_size: Actual resized image dimensions before padding.
        padding: Actual integer padding as ``(left, top, right, bottom)``.
        scale_x: Actual horizontal scale from source pixels to resized pixels.
        scale_y: Actual vertical scale from source pixels to resized pixels.
        crop_offset: Source coordinate offset when a crop was applied.  It is
            normally ``(0, 0)`` and is retained for task pipelines that crop
            before resizing.
        resize_type: ``0`` for stretch and ``1`` for letterbox.
    """

    original_size: Size
    model_size: Size
    resized_size: Size
    padding: Padding
    scale_x: float
    scale_y: float
    crop_offset: Tuple[float, float] = (0.0, 0.0)
    resize_type: int = 1

    def __post_init__(self) -> None:
        original = _size(self.original_size, "original_size")
        model = _size(self.model_size, "model_size")
        resized = _size(self.resized_size, "resized_size")
        if len(self.padding) != 4 or any(int(v) < 0 for v in self.padding):
            raise ValueError("padding must contain four non-negative integers.")
        padding = tuple(int(v) for v in self.padding)
        if resized[0] + padding[1] + padding[3] != model[0]:
            raise ValueError("vertical resize and padding do not fill model_size.")
        if resized[1] + padding[0] + padding[2] != model[1]:
            raise ValueError("horizontal resize and padding do not fill model_size.")
        if self.resize_type not in (0, 1):
            raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox).")
        if not np.isfinite(self.scale_x) or not np.isfinite(self.scale_y):
            raise ValueError("scale_x and scale_y must be finite.")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("scale_x and scale_y must be positive.")
        if len(self.crop_offset) != 2:
            raise ValueError("crop_offset must contain (x, y).")
        object.__setattr__(self, "original_size", original)
        object.__setattr__(self, "model_size", model)
        object.__setattr__(self, "resized_size", resized)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "crop_offset", (float(self.crop_offset[0]),
                                                   float(self.crop_offset[1])))
        object.__setattr__(self, "scale_x", float(self.scale_x))
        object.__setattr__(self, "scale_y", float(self.scale_y))

    @property
    def target_size(self) -> Size:
        """Alias for the final model input size."""
        return self.model_size

    @property
    def input_size(self) -> Size:
        """Alias for the final model input size."""
        return self.model_size

    @property
    def original_height(self) -> int:
        return self.original_size[0]

    @property
    def original_width(self) -> int:
        return self.original_size[1]

    @property
    def model_height(self) -> int:
        return self.model_size[0]

    @property
    def model_width(self) -> int:
        return self.model_size[1]

    @property
    def pad_left(self) -> int:
        return self.padding[0]

    @property
    def pad_top(self) -> int:
        return self.padding[1]

    @property
    def pad_right(self) -> int:
        return self.padding[2]

    @property
    def pad_bottom(self) -> int:
        return self.padding[3]

    @property
    def scale(self) -> Tuple[float, float]:
        """Return actual ``(scale_x, scale_y)`` values."""
        return self.scale_x, self.scale_y


def make_transform(original_size: Sequence[int],
                   model_size: Sequence[int],
                   resize_type: int = 1) -> ImageTransform:
    """Calculate a transform using the same integer rounding as preprocessing.

    Letterbox dimensions intentionally use truncation, matching the existing
    sample's ``int(original * scale)`` behavior.  The returned scales are
    calculated from those actual dimensions, rather than from the ideal scale.
    """
    original = _size(original_size, "original_size")
    model = _size(model_size, "model_size")
    if resize_type == 0:
        resized = model
        padding: Padding = (0, 0, 0, 0)
    elif resize_type == 1:
        source_h, source_w = original
        target_h, target_w = model
        ideal_scale = min(target_h / source_h, target_w / source_w)
        resized_h = max(1, min(target_h, int(source_h * ideal_scale)))
        resized_w = max(1, min(target_w, int(source_w * ideal_scale)))
        resized = (resized_h, resized_w)
        pad_w = target_w - resized_w
        pad_h = target_h - resized_h
        padding = (pad_w // 2, pad_h // 2,
                   pad_w - pad_w // 2, pad_h - pad_h // 2)
    else:
        raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox).")

    return ImageTransform(
        original_size=original,
        model_size=model,
        resized_size=resized,
        padding=padding,
        scale_x=resized[1] / original[1],
        scale_y=resized[0] / original[0],
        resize_type=resize_type,
    )


def resize_with_transform(image: np.ndarray,
                          model_size: Sequence[int],
                          resize_type: int = 1,
                          interpolation: Optional[int] = None,
                          pad_value=(127, 127, 127)) -> Tuple[np.ndarray, ImageTransform]:
    """Resize an image and return both pixels and the concrete transform.

    ``model_size`` is ``(height, width)``.  The interpolation defaults retain
    the old sample behavior: nearest-neighbor for stretch and OpenCV's linear
    interpolation for the letterbox resize.
    """
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("image must be a NumPy array with at least two dimensions.")
    transform = make_transform(image.shape[:2], model_size, resize_type)
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST if resize_type == 0 else cv2.INTER_LINEAR
    resized_h, resized_w = transform.resized_size
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    left, top, right, bottom = transform.padding
    if any((left, top, right, bottom)):
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    if tuple(resized.shape[:2]) != transform.model_size:
        raise ValueError(
            f"preprocessing produced {resized.shape[:2]}, expected {transform.model_size}.")
    return resized, transform


def inverse_boxes(boxes: np.ndarray,
                  transform: ImageTransform,
                  clip: bool = True) -> np.ndarray:
    """Map ``xyxy`` boxes from model-input coordinates to source coordinates."""
    array = np.asarray(boxes)
    if array.size == 0:
        if array.ndim == 1:
            return np.empty((0, 4), dtype=np.float32)
        if array.shape[-1:] != (4,):
            raise ValueError("boxes must have a final dimension of four.")
        return np.empty(array.shape, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4).")
    result = np.asarray(array, dtype=np.float32).copy()
    left, top, _, _ = transform.padding
    crop_x, crop_y = transform.crop_offset
    result[:, [0, 2]] = (result[:, [0, 2]] - left) / transform.scale_x + crop_x
    result[:, [1, 3]] = (result[:, [1, 3]] - top) / transform.scale_y + crop_y
    if clip:
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, transform.original_width)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, transform.original_height)
    return result


def restore_boxes(boxes: np.ndarray,
                  transform: ImageTransform,
                  clip: bool = True) -> np.ndarray:
    """Compatibility alias for :func:`inverse_boxes`."""
    return inverse_boxes(boxes, transform, clip=clip)


def scale_boxes_to_original(boxes: np.ndarray,
                            transform: ImageTransform,
                            clip: bool = True) -> np.ndarray:
    """Descriptive alias used by callers that prefer a scaling name."""
    return inverse_boxes(boxes, transform, clip=clip)


__all__ = [
    "ImageTransform",
    "make_transform",
    "resize_with_transform",
    "inverse_boxes",
    "restore_boxes",
    "scale_boxes_to_original",
]
