# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declared output transforms for quantized BPU artifacts (Phase 1.5 H1).

A binding declares, per artifact, how raw runtime outputs become the float32
values a task ``post_process`` may decode:

- ``raw_f32``: the artifact already returns F32 tensors.  Non-F32 outputs are
  a contract mismatch — the pilot era rejected int8 outputs implicitly; this
  is the same rule, declared.  A vestigial quant descriptor riding along an
  F32 output is accepted and ignored (real X5 artifacts ship such
  descriptors; board evidence 2026-09-21).
- ``dequant``: the artifact may return int8/uint8 outputs together with
  ``output_quants`` descriptors; the chain applies the scale/zero-point
  dequantization ported from the delivery branches'
  ``utils/py_utils/postprocess.py::dequantize_outputs`` (per-tensor and
  per-channel, SCALE-only).

Activation semantics are *not* part of this chain.  Whether a raw logit needs
a sigmoid, or a dequantized output is already activated, is a task-level fact
declared and executed by each sample's ``post_process`` (for example the X5
YOLO raw-logit -> sigmoid path versus the S dequantized-and-activated path).

The module imports NumPy lazily so metadata-only host checks stay lightweight.
"""
from __future__ import annotations

from typing import Any, Mapping

OUTPUT_TRANSFORMS = ("raw_f32", "dequant")


class OutputTransformError(ValueError):
    """A declared output transform cannot be applied to these outputs."""


def validate_output_transform(transform: Any) -> str:
    """Return the canonical transform name or raise for unknown names."""

    name = str(transform)
    if name not in OUTPUT_TRANSFORMS:
        raise OutputTransformError(
            f"Unknown output transform {transform!r}; expected one of "
            f"{list(OUTPUT_TRANSFORMS)}."
        )
    return name


def apply_output_transform(
    transform: Any,
    outputs: Mapping[str, Any],
    output_quants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one declared transform to a flat output mapping.

    ``output_quants`` is the per-output descriptor mapping as exposed by the
    runtime (``{output_name: quant_info}``).  For ``raw_f32`` the values must
    already be F32; a vestigial descriptor riding along an F32 output is
    accepted and ignored (real X5 artifacts ship such descriptors — board
    evidence 2026-09-21 — and legacy consumers ignored them).  For ``dequant``
    the mapping must describe all outputs.  Returned values are float32 NumPy
    arrays.
    """

    import numpy as np

    name = validate_output_transform(transform)
    quants = output_quants or {}
    result: dict[str, Any] = {}
    for key, value in outputs.items():
        array = np.asarray(value)
        quant_info = quants.get(str(key))
        if name == "raw_f32":
            if array.dtype != np.dtype("float32"):
                raise OutputTransformError(
                    f"raw_f32 output {key!r} has dtype {array.dtype!r}; "
                    "quantized outputs require the 'dequant' transform."
                )
            result[str(key)] = array
        else:  # dequant
            if quant_info is None:
                raise OutputTransformError(
                    f"dequant output {key!r} has no quantization descriptor in "
                    "output_quants; F32 artifacts use the 'raw_f32' transform."
                )
            result[str(key)] = dequantize_tensor(array, quant_info)
    return result


def dequantize_tensor(q_tensor: Any, quant_info: Any) -> Any:
    """Dequantize one tensor with its runtime quantization descriptor.

    Ported from the delivery branches' ``utils/py_utils/postprocess.py``
    (source of record: rdk_s @ 380e1a2), with the scalar zero-point broadcast
    corrected: a single offset applies to every channel instead of being
    discarded. Empty zero-points still mean symmetric quantization (zero).
    Per-tensor and per-channel SCALE dequantization are supported; descriptors
    whose ``quant_type`` is not
    SCALE are passed through unchanged, exactly like the source helper.
    """

    import numpy as np

    quant_type = getattr(quant_info, "quant_type", quant_info)
    quant_type_name = getattr(quant_type, "name", str(quant_type))
    if quant_type_name not in ("SCALE", "1"):
        return q_tensor

    scale = np.asarray(quant_info.scale)
    zero_point = np.asarray(quant_info.zero_point).astype(np.float32)
    if zero_point.size == 0:
        zero_point = np.zeros((1,), dtype=np.float32)

    tensor = np.asarray(q_tensor)
    if scale.ndim == 0 or tensor.ndim == 1 or scale.size == 1:
        # Per-tensor dequantization
        return (tensor.astype(np.float32) - zero_point.reshape(-1)[0]) * scale
    # Per-channel dequantization
    shape = [1] * tensor.ndim
    shape[int(quant_info.axis)] = -1
    reshaped_scale = scale.reshape(shape)
    if zero_point.size == 1:
        reshaped_zero_point = zero_point.reshape(-1)[0]
    else:
        reshaped_zero_point = zero_point.reshape(shape).astype(np.float32)
    return (tensor.astype(np.float32) - reshaped_zero_point) * reshaped_scale


def dequantize_outputs(
    outputs: Mapping[str, Any], quan_infos: Mapping[str, Any]
) -> dict[str, Any]:
    """Dequantize a flat output mapping with per-output descriptors."""

    return {
        str(name): dequantize_tensor(value, quan_infos[str(name)])
        for name, value in outputs.items()
    }


__all__ = [
    "OUTPUT_TRANSFORMS",
    "OutputTransformError",
    "apply_output_transform",
    "dequantize_outputs",
    "dequantize_tensor",
    "validate_output_transform",
]
