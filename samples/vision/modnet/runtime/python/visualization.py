# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional image output operations for MODNet."""

from __future__ import annotations

import cv2
import numpy as np


def composite(image: np.ndarray, matte: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Composite a uint8 matte over a resized BGR background."""

    if image.ndim != 3 or background.ndim != 3 or matte.shape != image.shape[:2]:
        raise ValueError("Image/background must be HWC and matte must match image geometry.")
    bg = cv2.resize(background, (image.shape[1], image.shape[0]))
    alpha = matte.astype(np.float32)[:, :, None] / 255.0
    return (image.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


__all__ = ["composite"]
