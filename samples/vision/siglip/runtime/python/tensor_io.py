# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SigLIP-specific RGB AREA letterbox conversion; no model or file access."""
from dataclasses import dataclass
from typing import Mapping
import cv2
import numpy as np


@dataclass(frozen=True)
class ImageContext:
    """Per-call (height,width) geometry and (top,bottom,left,right) padding."""
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    padding: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Owned F32 NCHW input and immutable geometry for one image."""
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


def prepare_image(image: np.ndarray, size: int) -> PreparedInput:
    """BGR U8 H×W×3 to RGB F32 [1,3,size,size], range [-1,1].

    Source numerical order is preserved: AREA resize, integer floor with
    minimum one pixel, padding 127, transpose/cast, then /127.5 - 1.
    Raises ValueError for non-U8, empty or non-three-channel input.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Expected nonempty BGR uint8 image shaped H×W×3.')
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(int(h*scale), 1), max(int(w*scale), 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    dh, dw = size-nh, size-nw
    padding = (dh//2, dh-dh//2, dw//2, dw-dw//2)
    padded = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, value=(127,127,127))
    tensor = np.transpose(padded, (2,0,1))[None].astype(np.float32)
    tensor = tensor / 127.5 - 1.0
    return PreparedInput({'_input_0': np.ascontiguousarray(tensor)}, ImageContext((h,w),(nh,nw),padding))
