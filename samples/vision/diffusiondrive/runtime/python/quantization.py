# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validated affine IO transforms; no SDK, activation, rendering or data loading."""

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from samples._shared.quantization import validate_scale_quantization, dequantize_tensor

DTYPES = frozenset(
    ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32")
)


@dataclass(frozen=True)
class AffineTransform:
    dtype: str
    shape: tuple[int, ...]
    scale: tuple[float, ...] = ()
    zero: tuple[float, ...] = ()
    axis: int = 0


def transform(dtype, shape, quant, *, input_tensor=False):
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported physical dtype: {dtype}")
    integer = np.issubdtype(np.dtype(dtype), np.integer)
    scales = np.asarray(getattr(quant, "scale", []), dtype=np.float32)
    if not scales.size:
        if integer:
            raise ValueError("Integer physical tensors require explicit SCALE metadata")
        zeros = np.asarray(getattr(quant, "zero_point", []))
        if (
            not np.isfinite(zeros).all()
            or np.any(zeros != 0)
            or str(
                getattr(
                    getattr(quant, "quant_type", None),
                    "name",
                    getattr(quant, "quant_type", None),
                )
            )
            not in ("None", "NONE", "0")
        ):
            raise ValueError(
                "Floating tensor has inconsistent empty quantization metadata"
            )
        return AffineTransform(dtype, tuple(shape))
    validate_scale_quantization(quant, shape)
    if input_tensor and scales.size != 1:
        raise ValueError("Source input contract supports per-tensor quantization only")
    zeros = np.asarray(getattr(quant, "zero_point", []), dtype=np.float64).reshape(-1)
    if integer:
        limits = np.iinfo(dtype)
        if (
            np.any(zeros != np.rint(zeros))
            or np.any(zeros < limits.min)
            or np.any(zeros > limits.max)
        ):
            raise ValueError(
                "Integer zero points must be integral and within dtype range"
            )
    axis = int(getattr(quant, "axis", 0)) if scales.size > 1 else 0
    return AffineTransform(
        dtype,
        tuple(shape),
        tuple(float(x) for x in scales.reshape(-1)),
        tuple(float(x) for x in zeros),
        axis,
    )


def quantize(value, spec):
    tensor = np.asarray(value)
    if (
        tensor.shape != spec.shape
        or tensor.dtype != np.dtype("float32")
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("Logical input requires exact finite float32 shape")
    if not spec.scale:
        result = tensor.astype(spec.dtype, copy=True)
    else:
        zero = spec.zero[0] if spec.zero else 0.0
        # Preserve source float32 arithmetic; clip in float64 to avoid an int32/
        # uint32 upper bound rounding beyond the dtype range before integer cast.
        with np.errstate(over="ignore"):
            values = np.rint(tensor / float(spec.scale[0]) + zero)
        if np.issubdtype(np.dtype(spec.dtype), np.integer):
            limits = np.iinfo(spec.dtype)
            values = np.clip(values.astype(np.float64), limits.min, limits.max)
        result = values.astype(spec.dtype)
    if not np.isfinite(result).all():
        raise ValueError("Physical input conversion produced nonfinite values")
    return np.ascontiguousarray(result)


def decode(value, spec):
    tensor = np.asarray(value)
    if (
        tensor.shape != spec.shape
        or tensor.dtype != np.dtype(spec.dtype)
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("Raw output differs from physical metadata")
    if spec.scale:
        q = SimpleNamespace(
            quant_type="SCALE",
            scale=np.asarray(spec.scale, np.float32),
            zero_point=np.asarray(spec.zero, np.float32),
            axis=spec.axis,
        )
        result = dequantize_tensor(tensor, q)
    else:
        result = tensor.astype(np.float32, copy=True)
    result = np.array(result, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("Decoded output contains nonfinite values")
    return result
