# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SigLIP feature task: no SDK, model loading, summary formatting or file I/O."""
from typing import Callable, Mapping
import numpy as np
from samples.vision.siglip.runtime.python.model_binding import ModelBinding
from samples.vision.siglip.runtime.python.tensor_io import PreparedInput, prepare_image


class SigLIPTask:
    """One selected feature submodel with an injected flat callable runner.

    Input is BGR U8 H×W×3; prepared tensors are RGB F32 NCHW [-1,1].
    Immutable image context lives in PreparedInput, never on the task; feature
    extraction does not consume geometry in post_process. RawOutputs is the
    flat `_output_0` mapping. Result is an owned ndarray in the bound native
    numeric dtype and shape (global or per-patch features), with no softmax,
    normalization, squeeze or dequantization. No SDK thread-safety is promised.
    """
    def __init__(self, runner: Callable, binding: ModelBinding):
        if not callable(runner):
            raise TypeError('runner must be callable')
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Validate BGR image and construct F32 tensors with per-call context."""
        return prepare_image(image, self.binding.selection.image_size)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Invoke runner only, returning its raw mapping without numeric changes."""
        return self.runner(tensors)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Validate shape/dtype/finiteness and own the selected raw embedding.

        NaN/Inf or changed runtime shape/dtype raises ValueError. A copy prevents
        later SDK calls reusing buffers from mutating an already returned result.
        """
        if not isinstance(outputs, Mapping) or set(outputs) != {'_output_0'}:
            raise ValueError('SigLIP output must contain only _output_0.')
        result = np.asarray(outputs['_output_0'])
        if result.shape != self.binding.output_shape or result.dtype != np.dtype(self.binding.output_dtype):
            raise ValueError('SigLIP output shape/dtype differs from bound metadata.')
        if not np.isfinite(result).all():
            raise ValueError('SigLIP output contains NaN or Inf.')
        return result.copy()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Compose exactly pre_process → forward → post_process."""
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared.tensors))
