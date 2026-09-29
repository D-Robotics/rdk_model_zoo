# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared source input arithmetic for inference and explicit calibration."""

import cv2
import numpy as np


def image_to_tensor(image):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("Expected nonempty BGR uint8 HWC image")
    rgb = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (512, 256),
        interpolation=cv2.INTER_AREA,
    )
    chw = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return np.ascontiguousarray(((chw - mean) / std)[None])
