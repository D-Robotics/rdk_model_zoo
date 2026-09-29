# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Actual source pixelwise RGB z-score → raw inference → relative float depth."""

from dataclasses import dataclass
from typing import Mapping
import cv2
import numpy as np
from samples.vision.depth_anything_v2.runtime.python.geometry import (
    ImageContext,
    make_context,
    validate_context,
)


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


@dataclass(frozen=True)
class DepthResult:
    depth_native: np.ndarray
    context: ImageContext


class DepthAnythingV2Task:
    def __init__(self, runner, binding, *, resize_type=0):
        make_context(1, 1, resize_type)
        self.runner, self.binding, self.resize_type = runner, binding, resize_type

    def pre_process(self, image):
        """BGR uint8 HWC → owned normalized RGB float32 NCHW plus geometry.

        Stretch uses source INTER_NEAREST; optional letterbox uses INTER_LINEAR
        and gray127. Normalize each pixel across RGB, not ImageNet constants.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected nonempty BGR uint8 HWC image")
        ctx = make_context(*image.shape[:2], self.resize_type)
        if self.resize_type == 0:
            resized = cv2.resize(image, (686, 518), interpolation=cv2.INTER_NEAREST)
        else:
            resized = cv2.resize(
                image,
                (686 - ctx.left - ctx.right, 518 - ctx.top - ctx.bottom),
                interpolation=cv2.INTER_LINEAR,
            )
            resized = cv2.copyMakeBorder(
                resized,
                ctx.top,
                ctx.bottom,
                ctx.left,
                ctx.right,
                cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb - rgb.mean(axis=-1, keepdims=True)) / np.sqrt(
            rgb.var(axis=-1, keepdims=True) + 1e-5
        )
        tensor = np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[None], dtype=np.float32
        )
        return PreparedInput({self.binding.input_name: tensor}, ctx)

    def forward(self, tensors):
        """Return owned raw float32 [1,518,686] without scaling or rendering."""
        return self.runner(tensors)

    def post_process(self, raw, context):
        """Finite float depth → crop optional padding, restore original H×W.

        OpenCV bilinear replaces source Torch align_corners=False. No numerical
        bit-identity is claimed. Output retains relative values, not meters or
        display-normalized intensities; visualization is a separate module.
        """
        validate_context(context, self.resize_type)
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (1, 518, 686)
            or raw.dtype != np.float32
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Expected finite float32 [1,518,686] output")
        plane = raw[
            0, context.top : 518 - context.bottom, context.left : 686 - context.right
        ]
        result = cv2.resize(
            plane,
            (context.original_w, context.original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite restored depth")
        return DepthResult(result.copy(), context)

    def predict(self, image):
        """Execute the same three stages with no IO, visualization or timing."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors), prepared.context)
