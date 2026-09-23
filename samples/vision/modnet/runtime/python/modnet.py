# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""MODNet preprocessing, raw execution, geometry restoration, and compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from samples.vision.modnet.runtime.python.model_binding import INPUT_SHAPE, ModelBinding


@dataclass(frozen=True)
class GeometryContext:
    """Immutable geometry required to restore one matte to one source image."""

    original_height: int
    original_width: int
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int
    target_size: int


@dataclass(frozen=True)
class PreparedInput:
    """One MODNet tensor mapping and its independent geometry context."""

    tensors: Mapping[str, np.ndarray]
    context: GeometryContext


def resize_with_padding(image: np.ndarray, target_size: int) -> tuple[np.ndarray, GeometryContext]:
    """Apply the source long-side resize, centered zero padding, and context capture."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR HWC image with three channels.")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or target_size <= 0:
        raise ValueError("Image and target size must be positive.")
    scale = target_size / max(height, width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    pad_w = target_size - resized_width
    pad_h = target_size - resized_height
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x,
        cv2.BORDER_CONSTANT, value=0,
    )
    return padded, GeometryContext(height, width, pad_x, pad_y, resized_width, resized_height, target_size)


class MODNetTask:
    """Four-stage MODNet task with per-call geometry and raw output ownership."""

    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Convert BGR HWC input to source RGB F32 NCHW ``[-1,1]``."""

        if image is None:
            raise ValueError("Input image is None.")
        rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        padded, context = resize_with_padding(normalized, INPUT_SHAPE[2])
        tensor = np.transpose(padded, (2, 0, 1))[None].astype(np.float32, copy=True)
        return PreparedInput({self.binding.input_name: tensor}, context)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Run the selected model and return an owned raw F32 matte tensor."""

        return self.runner(tensors)

    def post_process(self, raw: np.ndarray, context: GeometryContext) -> np.ndarray:
        """Convert raw ``[0,1]`` matte to uint8 original-image geometry."""

        value = np.asarray(raw)
        if value.shape != (1, 1, context.target_size, context.target_size) or value.dtype != np.float32:
            raise ValueError(f"Expected raw float32 matte (1,1,{context.target_size},{context.target_size}), got {value.shape}/{value.dtype}.")
        matte = (value[0, 0] * 255).astype(np.uint8)
        unpadded = matte[
            context.pad_y:context.pad_y + context.resized_height,
            context.pad_x:context.pad_x + context.resized_width,
        ]
        return cv2.resize(unpadded, (context.original_width, context.original_height), interpolation=cv2.INTER_LINEAR)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, raw forward, and geometry-aware postprocessing."""

        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors), prepared.context)


MODNet = MODNetTask

__all__ = ["GeometryContext", "MODNet", "MODNetTask", "PreparedInput", "resize_with_padding"]
