# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three-stage depth inference with explicit per-call restoration geometry."""

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from samples._shared.image import bgr_to_nv12_planes
from samples.vision.yolo26_depth.runtime.python.tensor_io import restore_log_depth
from samples.vision.yolo26_depth.runtime.python.geometry import (
    ImageContext,
    make_context,
    validate_context,
)
from samples.vision.yolo26_depth.runtime.python.model_binding import LITE_CALIBRATION


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


@dataclass(frozen=True)
class DepthResult:
    log_depth: np.ndarray
    depth_native: np.ndarray
    raw_logit: np.ndarray | None
    context: ImageContext


class Yolo26DepthTask:
    """BGR → profile-specific input → raw F32 → original-size relative depth."""

    def __init__(self, runner, binding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Nonempty BGR uint8 HWC → owned flat NV12 or RGB float32 NCHW.

        NV12 uses INTER_LINEAR letterbox with padding 114; lite uses scale-fill
        and /255. Geometry is returned with the tensor, never stored on the task.
        Invalid images or a collapsed letterbox dimension raise ValueError.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected a nonempty BGR uint8 HWC image")
        s = self.binding.selection
        ctx = make_context(*image.shape[:2], s.profile, s.variant)
        if s.profile == "lite":
            resized = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            value = (
                np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
                / 255.0
            )
        else:
            h, w = 768 - ctx.top - ctx.bottom, 768 - ctx.left - ctx.right
            resized = (
                image
                if image.shape[:2] == (h, w)
                else cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            )
            padded = cv2.copyMakeBorder(
                resized,
                ctx.top,
                ctx.bottom,
                ctx.left,
                ctx.right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            y, uv = bgr_to_nv12_planes(padded)
            value = np.concatenate((y.reshape(-1), uv.reshape(-1)))
        return PreparedInput({self.binding.input_name: value}, ctx)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return the runner's raw float32 single-channel tensor without decoding."""
        return self.runner(tensors)

    def post_process(self, raw: np.ndarray, context: ImageContext) -> DepthResult:
        """Raw F32 192-square output → calibrated log map and relative depth.

        NV12 is already calibrated: exp, resize to 768, crop padding, restore.
        Lite alone clips [-4,5] and applies source calibration before exp and
        direct restoration. Wrong tensors/context, NaN/Inf and exp overflow
        raise ValueError; arrays are owned and no metric-depth claim is made.
        """
        validate_context(context, self.binding.selection)
        shape = self.binding.metadata.output_shapes[self.binding.output_name]
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != shape
            or raw.dtype != np.float32
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Expected finite float32 output matching the bound shape")
        plane = raw.reshape(192, 192).copy()
        raw_logit = None
        if context.profile == "lite":
            raw_logit = plane.copy()
            a, b = LITE_CALIBRATION[context.variant]
            log_depth = np.clip(plane, -4.0, 5.0) * a + b
        else:
            log_depth = plane
        restored = restore_log_depth(log_depth, context)
        return DepthResult(log_depth, restored, raw_logit, context)

    def predict(self, image: np.ndarray) -> DepthResult:
        """Run the same three stages once, with no warmup, timing or file IO."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors), prepared.context)
