# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth restoration shared by runtime postprocessing and offline evaluation."""

import cv2
import numpy as np


def resize_opencv(depth, height, width):
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def restore_log_depth(log_depth, context, *, resize=resize_opencv):
    """Finite calibrated F32 H×W log-depth → original-size relative depth.

    Caller validates profile/geometry. Optional resize supplies the source
    evaluator's Torch backend without changing the runtime's OpenCV default.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        depth = np.exp(log_depth)
    if not np.isfinite(depth).all():
        raise ValueError("Depth exponential overflow; verify model output boundary")
    if context.profile == "nv12":
        square = resize(depth, context.size, context.size)
        depth = square[
            context.top : context.size - context.bottom,
            context.left : context.size - context.right,
        ]
    return resize(depth, context.original_height, context.original_width)
