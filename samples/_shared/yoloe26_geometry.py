# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE-26 PF geometry for inference, calibration and offline evaluation.

This is the fixed round/114 protocol, not a replacement for other samples'
truncate/127 letterboxing. No model, SDK, file I/O or global image state.
"""

from dataclasses import dataclass
from numbers import Integral
import cv2
import numpy as np


@dataclass(frozen=True)
class PFGeometry:
    """Validated, immutable concrete geometry for one 640-square PF input."""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]

    def __post_init__(self):
        original = tuple(self.original_size)
        if len(original) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
            for v in original
        ):
            raise ValueError(
                "Original size must contain two positive integer dimensions."
            )
        h, w = original
        gain = min(640 / h, 640 / w)
        rh, rw = max(1, round(h * gain)), max(1, round(w * gain))
        left, top = (640 - rw) // 2, (640 - rh) // 2
        if tuple(self.resized_size) != (rh, rw) or tuple(self.padding) != (
            left,
            top,
            640 - rw - left,
            640 - rh - top,
        ):
            raise ValueError(
                "Geometry does not describe YOLOE-26 round/114 letterboxing."
            )
        object.__setattr__(self, "original_size", (int(h), int(w)))
        object.__setattr__(self, "resized_size", (rh, rw))
        object.__setattr__(
            self, "padding", (left, top, 640 - rw - left, 640 - rh - top)
        )

    @property
    def scale_x(self):
        return self.resized_size[1] / self.original_size[1]

    @property
    def scale_y(self):
        return self.resized_size[0] / self.original_size[0]


def letterbox(image):
    """Return contiguous uint8 BGR[640,640,3] pixels and actual round/114 geometry."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("Expected a nonempty uint8 HWC BGR image.")
    h, w = image.shape[:2]
    gain = min(640 / h, 640 / w)
    rh, rw = max(1, round(h * gain)), max(1, round(w * gain))
    left, top = (640 - rw) // 2, (640 - rh) // 2
    context = PFGeometry((h, w), (rh, rw), (left, top, 640 - rw - left, 640 - rh - top))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pixels = cv2.copyMakeBorder(
        resized,
        top,
        context.padding[3],
        left,
        context.padding[2],
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return np.ascontiguousarray(pixels), context


def prepare_rgb(image):
    """Calibration/export RGB F32[1,3,640,640], using the same round/114 pixels."""
    pixels, _ = letterbox(image)
    return (
        np.ascontiguousarray(
            pixels[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        )
        / 255
    )
