# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""The readable SigLIP vision feature task.

:class:`SigLIPTask` keeps one complete feature pipeline visible in a single
file: construction binds one selected feature submodel through an injected
flat callable runner, :meth:`preprocess` turns one BGR image into the bound
RGB float32 NCHW tensor with per-call context, :meth:`infer` executes
exactly one runner call, :meth:`postprocess` validates and owns the raw
embedding, and :meth:`predict` chains the three steps.  The reusable pieces
stay shared: geometry and the ``PreparedInput`` record in ``tensor_io.py``,
selection/contracts in ``model_binding.py`` and the lazy packed-runtime
lifecycle in ``model_runner.py``.  No SDK, model loading, summary
formatting or file I/O happens here.

Input is BGR U8 H×W×3; prepared tensors are RGB F32 NCHW [-1,1].
Immutable image context lives in PreparedInput, never on the task; feature
extraction does not consume geometry in postprocess. RawOutputs is the
flat `_output_0` mapping. Result is an owned ndarray in the bound native
numeric dtype and shape (global or per-patch features), with no softmax,
normalization, squeeze or dequantization. No SDK thread-safety is promised.
"""
from typing import Callable, Mapping
import numpy as np
from samples.vision.siglip.runtime.python.model_binding import ModelBinding
from samples.vision.siglip.runtime.python.tensor_io import PreparedInput, prepare_image


class SigLIPTask:
    """One selected feature submodel with an injected flat callable runner."""

    def __init__(self, runner: Callable, binding: ModelBinding):
        if not callable(runner):
            raise TypeError('runner must be callable')
        self.runner = runner
        self.binding = binding

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Validate BGR image and construct F32 tensors with per-call context."""
        return prepare_image(image, self.binding.selection.image_size)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Invoke runner only, returning its raw mapping without numeric changes."""
        return self.runner(tensors)

    def postprocess(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
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
        """Compose exactly preprocess → infer → postprocess."""
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared.tensors))

    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)


__all__ = ['SigLIPTask']
