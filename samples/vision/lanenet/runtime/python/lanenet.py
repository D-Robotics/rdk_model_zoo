# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three stages return raw embeddings and model-grid binary labels, not lane IDs.

``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
established ``pre_process``/``forward``/``post_process`` names stay thin
aliases of those implementations.
"""

from dataclasses import dataclass
import numpy as np
from samples.vision.lanenet.runtime.python.tensor_io import validate_raw
from samples.vision.lanenet.runtime.python.image_preprocess import image_to_tensor


@dataclass(frozen=True)
class LaneResult:
    embedding: np.ndarray
    binary: np.ndarray


@dataclass(frozen=True)
class LanePredictionDetails:
    """One predict call's owned result plus its prepared input and raw outputs.

    Callers that archive ``raw_outputs.npz`` request this record with
    ``return_details=True`` instead of recomputing stages.  It describes only
    its own call; the task never retains a last output.
    """

    result: LaneResult
    prepared: dict
    raw: dict


class LaneNetTask:
    def __init__(self, runner, binding):
        self.runner, self.binding = runner, binding

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image):
        """Nonempty BGR uint8 HWC → source RGB/ImageNet float32 NCHW.

        INTER_AREA stretch preserves the source input policy. There is no
        letterbox, per-frame geometry, normalization guess or model-grid resize.
        """
        return {self.binding.input_name: image_to_tensor(image)}

    def infer(self, tensors):
        """Return all named raw outputs, including observed auxiliaries, unchanged."""
        return self.runner(tensors)

    def postprocess(self, outputs):
        """Bound raw tensors → owned float32 CHW embedding and uint8 0/1 labels.

        Binary labels must already be discrete 0/1. No sigmoid, argmax, cluster,
        color scaling, original-size restoration or quantized-logit guess occurs.
        """
        validate_raw(outputs, self.binding)
        binary = outputs[self.binding.binary_name]
        if not np.isin(binary, (0, 1)).all():
            raise ValueError("Binary prediction must contain only labels 0 and 1")
        return LaneResult(
            outputs[self.binding.embedding_name][0].copy(),
            binary.reshape(256, 512).astype(np.uint8, copy=True),
        )

    def predict(self, image, *, return_details=False):
        """Execute the same three stages once; no rendering, timing or IO.

        ``return_details=True`` wraps the usual :class:`LaneResult` with this
        call's prepared input and raw outputs, so one production inference
        also serves callers that archive ``raw_outputs.npz``; the default
        return stays the plain :class:`LaneResult`.
        """
        prepared = self.preprocess(image)
        raw = self.infer(prepared)
        result = self.postprocess(raw)
        if return_details:
            return LanePredictionDetails(result, prepared, raw)
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

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)
