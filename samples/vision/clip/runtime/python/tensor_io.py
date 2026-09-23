# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP image geometry and per-call multimodal input records."""
from dataclasses import dataclass
from typing import Mapping
import cv2
import numpy as np


@dataclass(frozen=True)
class InputContext:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    crop_origin: tuple[int, int]
    texts: tuple[str, ...]


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: InputContext


def prepare_inputs(image, texts, tokenizer):
    """Match source RGB/bicubic/rounded short-side/center crop/F32 divide."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError('image must be a nonempty BGR uint8 H×W×3 ndarray.')
    tokens = tokenizer(texts)
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = 224, int(round(w * 224 / h))
    else:
        new_h, new_w = int(round(h * 224 / w)), 224
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    y, x = max((new_h - 224)//2, 0), max((new_w - 224)//2, 0)
    cropped = resized[y:y+224, x:x+224].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(cropped.transpose(2, 0, 1)[None])
    return PreparedInput({'image': tensor, 'texts': tokens},
                         InputContext((h, w), (new_h, new_w), (y, x), tuple(texts)))
