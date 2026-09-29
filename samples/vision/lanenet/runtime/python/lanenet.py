# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three stages return raw embeddings and model-grid binary labels, not lane IDs."""

from dataclasses import dataclass
import numpy as np
from samples.vision.lanenet.runtime.python.tensor_io import validate_raw
from samples.vision.lanenet.runtime.python.image_preprocess import image_to_tensor


@dataclass(frozen=True)
class LaneResult:
    embedding: np.ndarray
    binary: np.ndarray


class LaneNetTask:
    def __init__(self, runner, binding):
        self.runner, self.binding = runner, binding

    def pre_process(self, image):
        """Nonempty BGR uint8 HWC → source RGB/ImageNet float32 NCHW.

        INTER_AREA stretch preserves the source input policy. There is no
        letterbox, per-frame geometry, normalization guess or model-grid resize.
        """
        return {self.binding.input_name: image_to_tensor(image)}

    def forward(self, tensors):
        """Return all named raw outputs, including observed auxiliaries, unchanged."""
        return self.runner(tensors)

    def post_process(self, outputs):
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

    def predict(self, image):
        """Execute the same three stages once; no rendering, timing or IO."""
        return self.post_process(self.forward(self.pre_process(image)))
