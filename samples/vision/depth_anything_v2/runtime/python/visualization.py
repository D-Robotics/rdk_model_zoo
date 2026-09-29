# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-image display normalization, kept separate from relative-depth values."""

import cv2
import numpy as np


def normalize_depth(depth):
    """Finite 2D relative depth → uint8 display values; constant maps yield zero.

    Float64 arithmetic avoids overflow in the range of finite float32 values.
    The constant-map policy replaces source division by zero, not a depth claim.
    """
    value = np.asarray(depth)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("Expected nonempty finite 2D depth")
    value = value.astype(np.float64)
    low, high = float(value.min()), float(value.max())
    if high == low:
        return np.zeros(value.shape, np.uint8)
    return np.clip((value - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def colorize_depth(depth):
    return cv2.applyColorMap(normalize_depth(depth), cv2.COLORMAP_INFERNO)
