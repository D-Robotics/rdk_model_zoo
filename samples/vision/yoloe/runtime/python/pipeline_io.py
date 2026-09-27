# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Protocol-specific prepared/result types and option validation, with no SDK I/O."""

from dataclasses import dataclass
from samples.vision.yoloe.runtime.python.config import validate_config
from typing import Any
import numpy as np
from samples._shared.image import bgr_to_nv12_planes
from samples._shared.yoloe26_geometry import letterbox, PFGeometry
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    resize_with_transform,
    ImageTransform,
)


@dataclass(frozen=True)
class Prepared:
    tensors: Any
    context: Any


@dataclass(frozen=True)
class Result:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: Any
    mask_layout: str


def prepare(image, selection, cfg, runner):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("Expected a nonempty uint8 HWC BGR image.")
    if selection.variant.startswith("26"):
        pixels, context = letterbox(image)
    else:
        pixels, context = resize_with_transform(image, (640, 640), cfg.resize_type)
    y, uv = bgr_to_nv12_planes(pixels)
    return Prepared(runner.prepare_input(y, uv), context)


def validate_context(context, selection, cfg):
    if selection.variant.startswith("26"):
        if not isinstance(context, PFGeometry):
            raise ValueError("YOLOE-26 requires its prepared PFGeometry.")
    elif (
        not isinstance(context, ImageTransform)
        or context.model_size != (640, 640)
        or context.resize_type != cfg.resize_type
    ):
        raise ValueError("YOLOE-11 requires its matching prepared ImageTransform.")
