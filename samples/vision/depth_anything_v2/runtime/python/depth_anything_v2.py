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


@dataclass(frozen=True)
class DepthPredictionDetails:
    """One predict call's owned result plus its prepared input and raw output.

    Callers that archive the raw tensor (``raw_depth.npy``) request this record
    with ``return_details=True`` instead of recomputing stages.  It describes
    only its own call; the task never retains a last image or last output.
    """

    result: DepthResult
    prepared: PreparedInput
    raw: np.ndarray


class DepthAnythingV2Task:
    """Actual source pixelwise RGB z-score → raw inference → relative float depth.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.
    """

    def __init__(self, runner, binding, *, resize_type=0):
        make_context(1, 1, resize_type)
        self.runner, self.binding, self.resize_type = runner, binding, resize_type

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image):
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

    def infer(self, tensors):
        """Return owned raw float32 [1,518,686] without scaling or rendering."""
        return self.runner(tensors)

    def postprocess(self, raw, context):
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

    def predict(self, image, *, return_details=False):
        """Execute the same three stages with no IO, visualization or timing.

        ``return_details=True`` wraps the usual :class:`DepthResult` with this
        call's prepared input and raw output, so one production inference also
        serves callers that archive ``raw_depth.npy``; the default return stays
        the plain :class:`DepthResult`.
        """
        prepared = self.preprocess(image)
        raw = self.infer(prepared.tensors)
        result = self.postprocess(raw, prepared.context)
        if return_details:
            return DepthPredictionDetails(result, prepared, raw)
        return result

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw, context):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw, context)
