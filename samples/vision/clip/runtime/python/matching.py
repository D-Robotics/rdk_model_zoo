# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""The readable CLIP image/text matching task.

:class:`CLIPTask` keeps the whole matching flow visible in one file:
:meth:`preprocess` turns one BGR image plus the prompt list into the bound
image tensor and real BPE tokens, :meth:`infer` executes exactly one paired
runner call (BPU image encoder + CPU ONNX text encoder), :meth:`postprocess`
computes the cosine scores and ranking, and :meth:`predict` chains the three
steps.  The reusable pieces stay shared: geometry/token record construction
in ``tensor_io.py``, BPE encoding in ``tokenization.py`` and both runtime
lifecycles in ``model_runner.py``.

Input is BGR uint8 image and a nonempty sequence of texts. Prepared tensors
use semantic keys image (F32[1,3,224,224]) and texts (I32[N,77]); runner
adapts those keys to observed SDK/ONNX names. Per-call geometry/texts remain
in PreparedInput.context. RawOutputs contains F32 image_feature[1,512]
and text_features[N,512]. Result contains cosine scores[N] and rank IDs[N].
No mutable request context is retained on this object; no SDK/visualization
or file I/O happens here.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from samples.vision.clip.runtime.python.tensor_io import prepare_inputs


@dataclass(frozen=True)
class MatchResult:
    scores: np.ndarray
    order: np.ndarray


class CLIPTask:
    """Injected runner and initialized tokenizer; no file/SDK/visualization I/O."""

    def __init__(self, runner, binding, tokenizer):
        if not callable(runner) or not callable(tokenizer):
            raise TypeError('runner and tokenizer must be callable.')
        self.runner, self.binding, self.tokenizer = runner, binding, tokenizer

    def preprocess(self, image, texts):
        """Prepare source image bytes and BPE tokens, retaining request context."""
        return prepare_inputs(image, texts, self.tokenizer)

    def infer(self, tensors):
        """Invoke the paired runner only; preserve the raw encoder outputs."""
        return self.runner(tensors)

    def postprocess(self, outputs):
        """Compute source F32 cosine with norm+1e-12 and descending argsort."""
        if not isinstance(outputs, Mapping) or set(outputs) != {'image_feature', 'text_features'}:
            raise ValueError('CLIP requires both raw feature outputs.')
        image = np.asarray(outputs['image_feature'])
        texts = np.asarray(outputs['text_features'])
        if image.shape != (1, 512) or texts.ndim != 2 or texts.shape[1] != 512 or texts.shape[0] < 1:
            raise ValueError('CLIP feature output shapes are invalid.')
        if image.dtype != np.float32 or texts.dtype != np.float32 or not np.isfinite(image).all() or not np.isfinite(texts).all():
            raise ValueError('CLIP feature outputs must be finite float32.')
        image = image.reshape(-1).astype(np.float32)
        texts = texts.astype(np.float32)
        image_norm = np.linalg.norm(image) + 1e-12
        text_norm = np.linalg.norm(texts, axis=1) + 1e-12
        scores = (texts @ image) / (text_norm * image_norm)
        if not np.isfinite(scores).all():
            raise ValueError('CLIP cosine overflowed; check runtime feature values.')
        return MatchResult(scores, np.argsort(scores)[::-1])

    def predict(self, image, texts):
        """Compose exactly preparation → raw runner → cosine/ranking."""
        prepared = self.preprocess(image, texts)
        return self.postprocess(self.infer(prepared.tensors))

    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).

    def pre_process(self, image, texts):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image, texts)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)


__all__ = ['CLIPTask', 'MatchResult']
