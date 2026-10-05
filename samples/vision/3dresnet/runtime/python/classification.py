# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Four-stage R3D-18 video classification task."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from samples._shared.classification import ClassificationResult, topk_from_scores
from .model_binding import CLASS_COUNT, ModelBinding
from .tensor_io import PreparedInput, prepare_clip


class VideoClassificationTask:
    """Classify one already-normalized ``(1,3,16,112,112)`` video clip.

    The readable pipeline makes the inference-contract data flow explicit:

    - **Input**: one numeric RGB clip with exact shape ``(1,3,16,112,112)``;
      the clip is already normalized and is not decoded or resized here.
    - **Tensors**: one contiguous float32 tensor under the input name exposed
      by runtime metadata; the name is never guessed from the task.
    - **Context**: a fresh :class:`VideoContext` in every :class:`PreparedInput`
      containing the source shape, dtype, and bound tensor shape.  The task
      has no mutable per-call context field.
    - **RawOutputs**: one finite float32 400-score tensor under the output
      name exposed by runtime metadata.  ``infer`` does no softmax or I/O.
    - **Result**: an owned :class:`ClassificationResult` produced by the
      shared source-compatible softmax and stable Top-K helper.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``.  The
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.
    """

    def __init__(
        self,
        runner,
        binding: ModelBinding,
        *,
        top_k: int = 5,
        labels: Optional[Mapping[int, str] | Sequence[str]] = None,
    ) -> None:
        if not callable(runner):
            raise TypeError("runner must be callable.")
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or not 1 <= int(top_k) <= CLASS_COUNT:
            raise ValueError(f"top_k must be between 1 and {CLASS_COUNT}.")
        self.runner = runner
        self.binding = binding
        self.top_k = int(top_k)
        self.labels = labels

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, clip: np.ndarray) -> PreparedInput:
        return prepare_clip(clip, self.binding)

    def infer(self, prepared: PreparedInput | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    def postprocess(self, outputs: Mapping[str, np.ndarray], *, top_k: int | None = None) -> ClassificationResult:
        if not isinstance(outputs, Mapping) or set(outputs) != {self.binding.output_name}:
            raise ValueError("R3D-18 raw outputs must contain exactly the bound score tensor.")
        raw = np.asarray(outputs[self.binding.output_name])
        if raw.shape != self.binding.output_shape:
            raise ValueError(
                f"R3D-18 output shape must be {self.binding.output_shape}, got {raw.shape}."
            )
        if raw.dtype != np.dtype(self.binding.output_dtype):
            raise ValueError(f"R3D-18 output dtype must be {self.binding.output_dtype}, got {raw.dtype}.")
        if not np.isfinite(raw).all():
            raise ValueError("R3D-18 output contains NaN or infinity.")
        selected_k = self.top_k if top_k is None else top_k
        if not isinstance(selected_k, (int, np.integer)) or isinstance(selected_k, bool) or not 1 <= int(selected_k) <= CLASS_COUNT:
            raise ValueError(f"top_k must be between 1 and {CLASS_COUNT}.")
        return topk_from_scores(raw, int(selected_k), self.labels, softmax=True)

    def predict(self, clip: np.ndarray, *, top_k: int | None = None) -> ClassificationResult:
        prepared = self.preprocess(clip)
        return self.postprocess(self.infer(prepared.tensors), top_k=top_k)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, clip: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""

        return self.preprocess(clip)

    def forward(self, prepared: PreparedInput | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Compatibility alias for :meth:`infer`."""

        return self.infer(prepared)

    def post_process(self, outputs: Mapping[str, np.ndarray], *, top_k: int | None = None) -> ClassificationResult:
        """Compatibility alias for :meth:`postprocess`."""

        return self.postprocess(outputs, top_k=top_k)

    def __call__(self, clip: np.ndarray, *, top_k: int | None = None) -> ClassificationResult:
        return self.predict(clip, top_k=top_k)


ResNet3DTask = VideoClassificationTask

__all__ = ["ResNet3DTask", "VideoClassificationTask"]
