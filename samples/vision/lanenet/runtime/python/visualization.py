# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source-native-style displays; colors do not represent clustered lane IDs."""

import numpy as np


def embedding_image(embedding):
    """Finite float32 CHW3 → HWC uint8 clip/round; channel order is preserved.

    This unifies Python with native clipping/saturating rounding and deliberately
    replaces source Python truncation/wrap. It is a display, not instance labels.
    """
    value = np.asarray(embedding)
    if (
        value.ndim != 3
        or value.shape[0] != 3
        or value.size == 0
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected finite float32 CHW embedding with three channels")
    return np.rint(np.clip(value, 0, 1) * 255).transpose(1, 2, 0).astype(np.uint8)


def binary_image(binary):
    """2D discrete labels → 0/255 display; reject invalid labels."""
    value = np.asarray(binary)
    if (
        value.ndim != 2
        or value.size == 0
        or value.dtype.kind not in "iu"
        or not np.isin(value, (0, 1)).all()
    ):
        raise ValueError("Expected 2D integer labels 0/1")
    return value.astype(np.uint8) * 255
