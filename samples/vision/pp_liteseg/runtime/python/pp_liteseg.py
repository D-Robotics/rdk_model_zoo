# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PP-LiteSeg four-stage class-map inference; no plotting or file IO."""
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from samples._shared.image import bgr_to_nv12_planes
from samples.vision.pp_liteseg.runtime.python.model_binding import ModelBinding


@dataclass(frozen=True)
class ImageContext:
    original_height: int
    original_width: int
    input_height: int = 512
    input_width: int = 1024


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class PPLiteSegTask:
    """Input BGR → packed NV12 → raw integer map → model-resolution labels."""

    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Nonempty BGR uint8 HWC → owned contiguous NV12 uint8 (768,1024).

        INTER_LINEAR stretches to 1024×512; no letterbox or CPU normalization.
        Invalid shape/dtype/empty dimensions raise ValueError. Original geometry
        belongs to the returned frozen context and is never saved on the task.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or not all(image.shape[:2])
        ):
            raise ValueError('Expected a nonempty BGR HWC uint8 image.')
        resized = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        y, uv = bgr_to_nv12_planes(resized)
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(768, 1024)
        return PreparedInput(
            {self.binding.input_name: packed}, ImageContext(*image.shape[:2])
        )

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return raw runner-validated (1,512,1024,1) int32 class IDs unchanged."""
        return self.runner(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Validate class IDs 0..18 and return owned int32 (512,1024) labels.

        The deployment boundary is already a class map: no argmax, softmax or
        dequantization is permitted. No context restoration occurs because the
        source contract returns model-resolution labels. ValueError rejects
        logits, wrong dtype/shape and out-of-range IDs rather than guessing.
        """
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (1, 512, 1024, 1)
            or raw.dtype != np.int32
            or np.any(raw < 0)
            or np.any(raw > 18)
        ):
            raise ValueError(
                'Expected int32 class IDs 0..18 shaped (1,512,1024,1), not logits.'
            )
        return raw[0, :, :, 0].copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run the same three stages, returning model-resolution class IDs."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors))
