# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNet model-resolution semantic masks; file IO and plotting are separate.

``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
established ``pre_process``/``forward``/``post_process`` names stay thin
aliases of those implementations.
"""
from dataclasses import dataclass
from typing import Mapping
import cv2
import numpy as np
from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.quantization import apply_output_transform
from samples.vision.unet.runtime.python.model_binding import ModelBinding


@dataclass(frozen=True)
class ImageContext:
    original_height: int
    original_width: int
    input_height: int = 512
    input_width: int = 512


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


class UNetTask:
    """Preprocess, raw forward, 21-class mask decoding and predict; no IO."""
    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """BGR uint8 nonempty HWC → owned packed NV12 (1,768,512,1).

        Direct INTER_LINEAR resize preserves the source stretch geometry.
        No letterbox, normalization or automatic original-resolution restoration.
        Invalid array shape/dtype/empty dimensions raise ValueError.
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or not all(image.shape[:2])):
            raise ValueError('Expected a nonempty BGR HWC uint8 image.')
        resized = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
        y, uv = bgr_to_nv12_planes(resized)
        packed = np.concatenate((y.reshape(-1),uv.reshape(-1))).reshape(1,768,512,1)
        return PreparedInput({self.binding.input_name: packed}, ImageContext(*image.shape[:2]))

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return raw runner-validated logits without dequantization or argmax."""
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Raw NCHW/NHWC 21-class logits → owned uint8 (512,512) class IDs.

        Integer logits require SCALE decoding; float32 is already decoded.
        No softmax. Ties select the lowest class ID. Input-resolution context
        is not consumed because the source contract returns model-resolution masks.
        Shape/dtype/nonfinite mismatches raise ValueError.
        """
        name = self.binding.output_name
        meta = self.binding.metadata
        if (not isinstance(raw,np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError('UNet raw output does not match bound shape/dtype/finite values.')
        transform = 'raw_f32' if raw.dtype == np.float32 else 'dequant'
        scores = apply_output_transform(transform,{name:raw},meta.output_quants)[name]
        if not np.isfinite(scores).all():
            raise ValueError('UNet dequantization produced nonfinite scores.')
        axis = 0 if scores.shape[1] == 21 else -1
        return scores[0].argmax(axis=axis).astype(np.uint8)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run the same three stages, returning a model-resolution class mask."""
        prepared = self.preprocess(image)
        return self.postprocess(self.infer(prepared.tensors))

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw)
