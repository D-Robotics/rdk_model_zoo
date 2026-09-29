# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Cityscapes segmentation stages; file IO, rendering and SDK loading live elsewhere."""
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from samples._shared.image import bgr_to_nv12_planes
from samples._shared.quantization import dequantize_tensor
from samples.vision.unetmobilenet.runtime.python.model_binding import ModelBinding


@dataclass(frozen=True)
class ImageContext:
    original_height: int
    original_width: int


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class UnetMobileNetTask:
    """BGR → split NV12 → raw logits → original-resolution int32 IDs."""

    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Stretch nonempty BGR uint8 HWC with INTER_AREA to 2048×1024.

        Returns Y [1,1024,2048,1], UV [1,512,1024,2], both contiguous uint8,
        plus immutable per-call original geometry. No normalization/letterbox.
        ValueError rejects non-images, empty dimensions and non-uint8 input.
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3 or image.dtype != np.uint8
                or not all(image.shape[:2])):
            raise ValueError('Expected nonempty BGR uint8 HWC input')
        resized = cv2.resize(image, (self.binding.input_width, self.binding.input_height),
                             interpolation=cv2.INTER_AREA)
        y, uv = bgr_to_nv12_planes(resized)
        return PreparedInput({self.binding.y_name: y, self.binding.uv_name: uv},
                             ImageContext(*image.shape[:2]))

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return runner-validated raw [1,H,W,19] int32/float32 unchanged."""
        return self.runner(tensors)

    def post_process(self, raw: np.ndarray, context: ImageContext) -> np.ndarray:
        """Decode scores and resize class IDs directly to the original image.

        SCALE logits are affine-dequantized only here, using float64 comparison
        to retain int32 differences; explicit NONE int32 and F32 remain raw.
        Argmax ties choose the lowest class ID. INTER_NEAREST restores geometry;
        result is an owned int32 [original_height, original_width] mask (0..18).
        Invalid logits/context raise ValueError. No coloring or blending occurs.
        """
        meta = self.binding.metadata
        name = self.binding.output_name
        if (not isinstance(raw, np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError('Logits do not match the bound shape/dtype or contain nonfinite values')
        if (not isinstance(context, ImageContext) or context.original_height <= 0
                or context.original_width <= 0):
            raise ValueError('A valid per-call ImageContext is required')
        scores = raw
        if raw.dtype == np.int32:
            scores = dequantize_tensor(raw, meta.output_quants[name], dtype='float64')
        labels = np.argmax(scores[0], axis=-1).astype(np.int32)
        return cv2.resize(labels, (context.original_width, context.original_height),
                          interpolation=cv2.INTER_NEAREST).copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run the same three stages and return class IDs, not a visualization."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors), prepared.context)
