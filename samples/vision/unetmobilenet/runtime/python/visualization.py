# Copyright (c) 2025-2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source rdk_colors palette; alpha_f weights the ORIGINAL image."""
import cv2
import numpy as np

# Preserved from rdk_s utils/py_utils/visualize.py, consumed as BGR by source.
PALETTE_BGR = np.array([
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
], dtype=np.uint8)


def render_overlay(image, labels, *, alpha_f=0.75):
    """Blend original BGR with source colors; neither inference nor file IO."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or not all(image.shape[:2])):
        raise ValueError('Expected nonempty BGR uint8 HWC image')
    if (not isinstance(labels, np.ndarray) or labels.shape != image.shape[:2]
            or labels.dtype != np.int32 or np.any(labels < 0) or np.any(labels >= 19)):
        raise ValueError('Expected original-resolution int32 class IDs 0..18')
    if not 0 <= alpha_f <= 1:
        raise ValueError('alpha_f must be in [0,1]')
    return cv2.addWeighted(image, alpha_f, PALETTE_BGR[labels], 1-alpha_f, 0.0)
