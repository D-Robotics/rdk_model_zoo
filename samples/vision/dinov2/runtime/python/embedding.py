# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 feature task: preprocessing, raw forward, and output conversion."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from samples._shared.quantization import apply_output_transform
from samples.vision.dinov2.runtime.python.model_binding import (
    ModelBinding,
    OUTPUTS,
)
from samples.vision.dinov2.runtime.python.tensor_io import PreparedInput, prepare_image

CLS_FEAT = "cls_feat"
PATCH_FEAT = "patch_feat"


class DINOv2Task:
    """One selected DINOv2 output over a validated dual-output runner.

    ``RawOutputs`` is a flat mapping containing both source outputs.  Numeric
    conversion happens only in ``post_process`` and no activation is applied.
    Returned arrays are owned float32 feature tensors.
    """

    def __init__(self, runner, binding: ModelBinding, output: str = CLS_FEAT):
        if output not in OUTPUTS:
            raise ValueError(f"Unsupported DINOv2 output: {output}")
        if not callable(runner):
            raise TypeError("runner must be callable")
        self.runner = runner
        self.binding = binding
        self.output = output

    def pre_process(self, image: np.ndarray, image_format: str = "BGR") -> PreparedInput:
        """Validate BGR uint8 image and build one per-call prepared input."""

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")
        return prepare_image(image, self.binding)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Call the runner and preserve its raw output values and containers."""

        return self.runner(tensors)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Dequantize/validate the selected output and return an owned F32 array."""

        if not isinstance(outputs, Mapping) or set(outputs) != set(OUTPUTS):
            raise ValueError("DINOv2 raw output must contain exactly cls_feat and patch_feat.")
        raw_by_name: dict[str, np.ndarray] = {}
        for name in OUTPUTS:
            value = np.asarray(outputs[name])
            if value.shape != self.binding.output_shapes[name]:
                raise ValueError(f"DINOv2 {name} shape differs from bound metadata.")
            if value.dtype != np.dtype(self.binding.output_dtypes[name]):
                raise ValueError(f"DINOv2 {name} dtype differs from bound metadata.")
            raw_by_name[name] = value
        name = self.output
        transformed = apply_output_transform(
            self.binding.output_transforms[name],
            {name: raw_by_name[name]},
            {name: self.binding.output_quants.get(name)},
        )[name]
        result = np.asarray(transformed).astype(np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError(f"DINOv2 {name} output contains NaN or Inf values.")
        return result

    def predict(self, image: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Compose exactly pre_process → forward → post_process."""

        prepared = self.pre_process(image, image_format)
        return self.post_process(self.forward(prepared.tensors))

    def __call__(self, image: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        return self.predict(image, image_format)


Dinov2Task = DINOv2Task

__all__ = ["CLS_FEAT", "PATCH_FEAT", "DINOv2Task", "Dinov2Task"]
