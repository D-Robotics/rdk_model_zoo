# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-frame source resize geometry; no mutable task state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageContext:
    original_h: int
    original_w: int
    resize_type: int
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


def make_context(height, width, resize_type):
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    if type(resize_type) is not int or resize_type not in (0, 1):
        raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox)")
    if resize_type == 0:
        return ImageContext(height, width, resize_type)
    scale = min(518 / height, 686 / width)
    h, w = int(height * scale), int(width * scale)
    if min(h, w) <= 0:
        raise ValueError("Letterbox dimension collapsed to zero")
    ph, pw = 518 - h, 686 - w
    return ImageContext(
        height, width, resize_type, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    )


def validate_context(context, resize_type):
    if not isinstance(context, ImageContext) or context.resize_type != resize_type:
        raise ValueError("Wrong image context/profile")
    if context != make_context(context.original_h, context.original_w, resize_type):
        raise ValueError("Geometry context does not match original dimensions")
