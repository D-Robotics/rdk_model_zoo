# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""FCOS X5 image geometry and packed-NV12 preparation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from samples._shared.image import bgr_to_nv12_planes


@dataclass(frozen=True)
class ImageContext:
    """Geometry for one call; no mutable task state is used."""

    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    resize_type: int
    resized_shape: tuple[int, int]
    pad: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Packed tensors plus the context consumed by FCOS post-processing."""

    tensors: dict[str, np.ndarray]
    context: ImageContext


def prepare(image: np.ndarray, binding, *, resize_type: int | None = None) -> PreparedInput:
    """Convert BGR uint8 input to one flat packed NV12 tensor."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("FCOS input must be a BGR array with shape (H,W,3).")
    if image.dtype != np.uint8 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("FCOS input must be a non-empty uint8 BGR array.")
    chosen = binding.contract.resize_type if resize_type is None else resize_type
    if chosen not in (0, 1):
        raise ValueError("resize_type must be 0 (direct) or 1 (letterbox).")
    height, width = image.shape[:2]
    target_h, target_w = binding.input_height, binding.input_width
    if chosen == 0:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        pad = (0, 0, 0, 0)
        resized_shape = (target_h, target_w)
    else:
        scale = min(target_h / height, target_w / width)
        new_w, new_h = int(width * scale), int(height * scale)
        resized_small = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = target_w - new_w, target_h - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        resized = cv2.copyMakeBorder(resized_small, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        pad = (top, bottom, left, right)
        resized_shape = (new_h, new_w)
    y, uv = bgr_to_nv12_planes(resized)
    packed = np.ascontiguousarray(np.concatenate((y.reshape(-1), uv.reshape(-1))), dtype=np.uint8)
    tensors = {binding.input_names[0]: packed}
    binding.validate_inputs(tensors)
    return PreparedInput(tensors=tensors, context=ImageContext((height, width), (target_h, target_w), chosen, resized_shape, pad))


__all__ = ["ImageContext", "PreparedInput", "prepare"]
