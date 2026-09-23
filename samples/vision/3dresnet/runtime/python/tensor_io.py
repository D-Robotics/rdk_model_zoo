# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Five-dimensional video clip validation and float32 preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model_binding import INPUT_SHAPE, ModelBinding


@dataclass(frozen=True)
class VideoContext:
    original_shape: tuple[int, ...]
    original_dtype: str
    input_shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: VideoContext


def prepare_clip(clip: np.ndarray, binding: ModelBinding) -> PreparedInput:
    if not isinstance(clip, np.ndarray):
        raise ValueError("Expected a NumPy video clip.")
    if clip.shape != INPUT_SHAPE:
        raise ValueError(f"Expected input shape {INPUT_SHAPE}, got {clip.shape}.")
    if not np.issubdtype(clip.dtype, np.number):
        raise ValueError(f"Video clip must be numeric, got {clip.dtype}.")
    tensor = np.ascontiguousarray(clip.astype(np.float32, copy=True))
    if not np.isfinite(tensor).all():
        raise ValueError("Video clip contains NaN or infinity.")
    if tensor.shape != binding.input_shape or tensor.dtype != np.dtype(binding.input_dtype):
        raise ValueError("Prepared video clip does not match the bound input.")
    return PreparedInput(
        tensors={binding.input_name: tensor},
        context=VideoContext(tuple(clip.shape), str(clip.dtype), tuple(tensor.shape)),
    )


__all__ = ["PreparedInput", "VideoContext", "prepare_clip"]
