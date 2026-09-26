# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Relative-depth display only; colors are not calibrated distance values."""

import cv2
import numpy as np


def colorize_depth(depth):
    """Source 2nd/98th percentile normalization and inverted TURBO palette.

    Nonempty finite floating-point H×W arrays are required. Runtime already
    rejects nonfinite inference; rendering never hides invalid model output.
    """
    if (
        not isinstance(depth, np.ndarray)
        or depth.ndim != 2
        or not depth.size
        or depth.dtype.kind != "f"
        or not np.isfinite(depth).all()
    ):
        raise ValueError("Expected nonempty finite floating-point HxW depth")
    low, high = np.percentile(depth, (2, 98))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
