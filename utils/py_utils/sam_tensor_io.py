# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free tensor preparation for the EfficientSAM and MobileSAM contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


IMAGE_SIZE = 512
DEFAULT_BOX = (185.0, 120.0, 380.0, 445.0)
EFFICIENT_MEAN = np.zeros((3, 1, 1), dtype=np.float32)
EFFICIENT_STD = np.ones((3, 1, 1), dtype=np.float32) * 255.0
MOBILE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
MOBILE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class StageContext:
    """Immutable facts for one stage call."""

    sample: str
    stage: str
    source_shape: tuple[int, ...] | None
    tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PreparedStageInput:
    """Owned stage tensors plus immutable per-call context."""

    tensors: Mapping[str, np.ndarray]
    context: StageContext


def _as_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("image must be a three-channel HWC array")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be a three-channel HWC array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must have a numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError("image must be finite")
    return array


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    bgr = _as_image(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def _owned_f32(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("tensor must have a numeric dtype")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("tensor must be finite")
    return result


def _shape_map(binding: Any, *, outputs: bool = False) -> Mapping[str, tuple[int, ...]]:
    metadata = binding.metadata
    field = "output_shapes" if outputs else "input_shapes"
    values = getattr(metadata, field, {})
    return {str(name): tuple(int(dim) for dim in shape) for name, shape in values.items()}


def prepare_encoder_input(image: Any, binding: Any) -> PreparedStageInput:
    """Apply source RGB conversion, linear 512 resize and sample normalization."""

    image_array = _as_image(image)
    names = tuple(binding.input_names)
    if len(names) != 1:
        raise ValueError("SAM encoder requires exactly one input tensor")
    name = names[0]
    shapes = _shape_map(binding)
    expected = shapes.get(name)
    if expected != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"SAM encoder input metadata must be (1,3,512,512), got {expected!r}")
    rgb = _resize_rgb(image_array).transpose(2, 0, 1).astype(np.float32, copy=False)
    if binding.sample == "mobile_sam":
        tensor = ((rgb - MOBILE_MEAN) / MOBILE_STD)[None]
    elif binding.sample == "efficient_sam":
        tensor = ((rgb - EFFICIENT_MEAN) / EFFICIENT_STD)[None]
    else:
        raise ValueError(f"unsupported SAM sample {binding.sample!r}")
    owned = np.array(tensor, dtype=np.float32, order="C", copy=True)
    return PreparedStageInput(
        {name: owned},
        StageContext(binding.sample, "encoder", tuple(image_array.shape), ((name, tuple(owned.shape)),)),
    )


def validate_box(box: Sequence[float] = DEFAULT_BOX) -> tuple[float, float, float, float]:
    """Validate an inclusive 512-coordinate box prompt and return plain floats."""

    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError):
        raise ValueError("box must contain four finite coordinates") from None
    if len(values) != 4 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("box must contain four finite coordinates")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= IMAGE_SIZE and 0.0 <= y1 < y2 <= IMAGE_SIZE):
        raise ValueError("box must be ordered and lie in [0, 512]")
    return values


def _decoder_input_names(binding: Any) -> tuple[str, str | None]:
    shapes = _shape_map(binding)
    embedding: str | None = None
    box_name: str | None = None
    for name in binding.input_names:
        shape = shapes.get(name)
        if shape == (1, 4) or shape == (1, 4, 1, 1):
            if box_name is not None:
                raise ValueError("SAM decoder exposes multiple box inputs")
            box_name = name
        elif shape is not None and len(shape) == 4:
            if embedding is not None:
                raise ValueError("SAM decoder exposes multiple embedding inputs")
            embedding = name
    if embedding is None:
        raise ValueError("SAM decoder embedding input is missing from metadata")
    return embedding, box_name


def prepare_decoder_input(embedding: Any, binding: Any, box: Sequence[float] | None = None) -> PreparedStageInput:
    """Prepare an owned float32 embedding and metadata-shaped box prompt."""

    embedding_name, box_name = _decoder_input_names(binding)
    if box_name is None and box is not None:
        raise ValueError("EfficientSAM uses a fixed exported prompt and does not accept a runtime box.")
    expected = _shape_map(binding).get(embedding_name)
    owned_embedding = _owned_f32(embedding)
    if expected is None or tuple(owned_embedding.shape) != expected:
        raise ValueError(f"decoder embedding shape must be {expected!r}, got {owned_embedding.shape}")
    tensors: dict[str, np.ndarray] = {embedding_name: owned_embedding}
    checked_box: tuple[float, float, float, float] | None = None
    if box_name is not None:
        checked_box = validate_box(DEFAULT_BOX if box is None else box)
        box_shape = _shape_map(binding).get(box_name)
        if box_shape not in ((1, 4), (1, 4, 1, 1)):
            raise ValueError(f"unsupported decoder box shape {box_shape!r}")
        tensors[box_name] = np.array(checked_box, dtype=np.float32, copy=True).reshape(box_shape)
    context = StageContext(
        binding.sample,
        "decoder",
        tuple(owned_embedding.shape),
        tuple((name, tuple(value.shape)) for name, value in tensors.items()),
        checked_box,
    )
    return PreparedStageInput(tensors, context)


__all__ = [
    "DEFAULT_BOX",
    "EFFICIENT_MEAN",
    "EFFICIENT_STD",
    "IMAGE_SIZE",
    "MOBILE_MEAN",
    "MOBILE_STD",
    "PreparedStageInput",
    "StageContext",
    "prepare_decoder_input",
    "prepare_encoder_input",
    "validate_box",
]
