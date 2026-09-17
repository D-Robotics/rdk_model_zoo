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

"""Validated tensor adapters for the YOLO model boundary.

This module contains transport concerns only.  It does not decide which
tensor is a detection head and it never invents quantization parameters.  The
binding layer supplies the named roles and the physical metadata; this module
checks and performs the requested layout conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class TensorContractError(ValueError):
    """Raised when runtime tensors do not satisfy a declared contract."""


def normalize_shape(shape: Sequence[int], name: str = "shape") -> Tuple[int, ...]:
    """Normalize a runtime shape and reject non-positive dimensions."""
    try:
        result = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise TensorContractError(f"{name} must be a sequence of dimensions.") from exc
    if not result or any(value <= 0 for value in result):
        raise TensorContractError(f"{name} must contain positive dimensions.")
    return result


def normalize_dtype(value: Any) -> Optional[np.dtype]:
    """Convert common SDK dtype representations to ``numpy.dtype``."""
    if value is None:
        return None
    if isinstance(value, np.dtype):
        return value
    # SDK enum instances expose their semantic spelling through ``.name``.
    # Check that spelling before NumPy parses strings such as ``U8`` as a
    # Unicode dtype.
    text = str(getattr(value, "name", value)).strip().lower()
    aliases = {
        # SDK enum instances expose these bare names through ``.name``.
        "f32": "float32",
        "u8": "uint8",
        "nv12": "uint8",
        "hbdnndatatype.f32": "float32",
        "hbdnndatatype.f16": "float16",
        "hbdnndatatype.u8": "uint8",
        # The X5 runtime labels a compact NV12 transport as a distinct
        # SDK dtype even though the Python boundary receives uint8 bytes.
        "hbdnndatatype.nv12": "uint8",
    }
    if text in aliases:
        return np.dtype(aliases[text])
    try:
        return np.dtype(value)
    except (TypeError, ValueError):
        candidate = getattr(value, "dtype", None)
        if candidate is not None and candidate is not value:
            try:
                return np.dtype(candidate)
            except (TypeError, ValueError):
                pass
        try:
            return np.dtype(aliases.get(text, text))
        except (TypeError, ValueError) as exc:
            raise TensorContractError(f"Unsupported tensor dtype {value!r}.") from exc


def element_count(shape: Sequence[int]) -> int:
    """Return the product of tensor dimensions."""
    count = 1
    for value in shape:
        count *= int(value)
    return count


def _profile_is_packed(profile: Any) -> bool:
    """Read the platform's input protocol without importing platform code."""
    value = getattr(profile, "is_packed_input", None)
    if callable(value):
        value = value()
    if value is None:
        protocol = str(getattr(profile, "input_protocol", "")).lower()
        value = protocol == "packed"
    return bool(value)


def _shape_for_input(shape: Sequence[int],
                     *,
                     packed: bool,
                     override: Optional[Tuple[int, int]],
                     label: str) -> Tuple[int, int, str]:
    """Derive ``(height, width, layout)`` from one input shape."""
    shape = normalize_shape(shape, f"input {label!r} shape")
    if len(shape) == 4 and shape[0] != 1:
        raise TensorContractError("Only batch-one NV12 inputs are supported.")

    if packed:
        if len(shape) == 4 and shape[1] == 3:
            return shape[2], shape[3], "NCHW"
        raise TensorContractError(
            f"Input {label!r} reports shape {shape}; packed NV12 requires "
            "the observed NCHW (1, 3, H, W) descriptor.")

    # Source-supported split NV12 inputs are NHWC Y and UV planes.
    if len(shape) == 4 and shape[3] == 1:
        return shape[1], shape[2], "NHWC"
    raise TensorContractError(
        f"Input {label!r} reports shape {shape}; split NV12 requires the observed "
        "NHWC (1, H, W, 1) Y descriptor.")


def _validate_nv12_geometry(height: int, width: int, label: str) -> None:
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise TensorContractError(
            f"Input {label!r} has {height}x{width}; NV12 dimensions must be positive and even.")


@dataclass(frozen=True)
class InputBinding:
    """Bind logical NV12 roles to physical runtime input tensors."""

    model_name: str
    roles: Mapping[str, str]
    shapes: Mapping[str, Tuple[int, ...]]
    dtypes: Mapping[str, Optional[np.dtype]]
    packed: bool
    height: int
    width: int
    layouts: Mapping[str, str] = field(default_factory=dict)

    @property
    def input_names(self) -> Tuple[str, ...]:
        """Return physical names in role declaration order."""
        return tuple(self.roles.values())

    @property
    def input_height(self) -> int:
        return self.height

    @property
    def input_width(self) -> int:
        return self.width

    @property
    def is_packed(self) -> bool:
        return self.packed

    @property
    def is_packed_input(self) -> bool:
        return self.packed

    def expected_anchor_sizes(self, strides: Sequence[int]):
        """Return rectangular ``(height, width)`` grid shapes for strides."""
        result = []
        for stride in strides:
            stride = int(stride)
            if stride <= 0 or self.height % stride or self.width % stride:
                raise TensorContractError(
                    f"Stride {stride} does not divide input {self.height}x{self.width}.")
            result.append((self.height // stride, self.width // stride))
        return result

    def describe(self) -> str:
        protocol = "packed" if self.packed else "split"
        return f"{protocol} NV12 {self.height}x{self.width} ({', '.join(self.input_names)})"

    def build(self, y_plane: np.ndarray, uv_plane: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Pack Y and UV arrays into the declared runtime input structure."""
        y = np.asarray(y_plane)
        uv = np.asarray(uv_plane)
        expected_y = self.height * self.width
        expected_uv = expected_y // 2
        if y.size != expected_y:
            raise TensorContractError(
                f"Luma plane holds {y.size} elements; expected {expected_y}.")
        if uv.size != expected_uv:
            raise TensorContractError(
                f"Chroma plane holds {uv.size} elements; expected {expected_uv}.")
        y_flat = np.ascontiguousarray(y.reshape(-1), dtype=np.uint8)
        uv_flat = np.ascontiguousarray(uv.reshape(-1), dtype=np.uint8)

        if self.packed:
            name = self.roles.get("image")
            if name is None:
                raise TensorContractError("Packed NV12 binding must define an image role.")
            packed = np.concatenate((y_flat, uv_flat)).astype(np.uint8, copy=False)
            return {self.model_name: {name: np.ascontiguousarray(packed)}}

        y_name = self.roles.get("y")
        uv_name = self.roles.get("uv")
        if y_name is None or uv_name is None:
            raise TensorContractError("Split NV12 binding must define y and uv roles.")
        y_shape = self.shapes[y_name]
        uv_shape = self.shapes[uv_name]
        y_value = _reshape_plane(y_flat, y_shape, self.height, self.width, y_name)
        uv_value = _reshape_plane(uv_flat, uv_shape, self.height // 2,
                                  self.width // 2, uv_name)
        return {self.model_name: {y_name: y_value, uv_name: uv_value}}


def _reshape_plane(values: np.ndarray,
                   declared_shape: Sequence[int],
                   height: int,
                   width: int,
                   name: str) -> np.ndarray:
    shape = normalize_shape(declared_shape, f"input {name!r} shape")
    if element_count(shape) != values.size:
        raise TensorContractError(
            f"Input {name!r} shape {shape} holds {element_count(shape)} elements; "
            f"expected {values.size} for {height}x{width} NV12 data.")
    return np.ascontiguousarray(values.reshape(shape), dtype=np.uint8)


def bind_nv12_inputs(profile: Any,
                     model_name: str,
                     input_names: Sequence[str],
                     input_shapes: Mapping[str, Sequence[int]],
                     input_dtypes: Optional[Mapping[str, Any]] = None,
                     roles: Optional[Mapping[str, str]] = None,
                     input_shape_override: Optional[Sequence[int]] = None) -> InputBinding:
    """Create a validated NV12 input binding from runtime metadata.

    ``roles`` maps logical names (``image`` or ``y``/``uv``) to physical names.
    When omitted, only the reviewed S-series pairs ``images_y``/``images_uv``
    or ``y``/``uv`` are accepted; an arbitrary pair must be declared explicitly.
    """
    names = tuple(str(name) for name in input_names)
    if not names:
        raise TensorContractError("Runtime reports no model inputs.")
    packed = _profile_is_packed(profile)
    expected_count = 1 if packed else 2
    if len(names) != expected_count:
        raise TensorContractError(
            f"Expected {expected_count} {'packed' if packed else 'split'} NV12 input(s); "
            f"runtime reports {len(names)} ({', '.join(names)}).")
    shapes = {}
    for name in names:
        if name not in input_shapes:
            raise TensorContractError(f"Runtime reports input {name!r} without a shape.")
        shapes[name] = normalize_shape(input_shapes[name], f"input {name!r} shape")
    dtype_map = {str(name): normalize_dtype(value)
                 for name, value in (input_dtypes or {}).items()}
    missing = [name for name in names if name not in dtype_map]
    if missing:
        raise TensorContractError(
            f"Runtime does not expose dtype metadata for input(s): {', '.join(missing)}.")
    if any(name in dtype_map and dtype_map[name] != np.dtype(np.uint8) for name in names):
        bad = next(name for name in names if name in dtype_map and
                   dtype_map[name] != np.dtype(np.uint8))
        raise TensorContractError(f"NV12 input {bad!r} must have uint8 dtype.")

    override = None
    if input_shape_override is not None:
        if len(input_shape_override) != 2:
            raise TensorContractError("input_shape_override must be (height, width).")
        override = (int(input_shape_override[0]), int(input_shape_override[1]))
        _validate_nv12_geometry(*override, "input_shape_override")

    if roles is None:
        if packed:
            role_map = {"image": names[0]}
        elif set(("images_y", "images_uv")) <= set(names):
            # These are the names recorded for the S-series pilot artifacts.
            role_map = {"y": "images_y", "uv": "images_uv"}
        elif set(("y", "uv")) <= set(names):
            # Keep the established generic names usable for reviewed fixtures.
            role_map = {"y": "y", "uv": "uv"}
        else:
            raise TensorContractError(
                "Split NV12 input names are not a reviewed pair. Supply an explicit "
                "logical {'y': ..., 'uv': ...} role mapping.")
    else:
        role_map = {str(role): str(name) for role, name in roles.items()}
        required = {"image"} if packed else {"y", "uv"}
        if set(role_map) != required:
            missing_roles = required - set(role_map)
            extra_roles = set(role_map) - required
            detail = []
            if missing_roles:
                detail.append("missing " + ", ".join(sorted(missing_roles)))
            if extra_roles:
                detail.append("unknown " + ", ".join(sorted(extra_roles)))
            raise TensorContractError(
                "NV12 role mapping is incomplete (" + "; ".join(detail) + ").")
        if set(role_map.values()) != set(names):
            raise TensorContractError(
                "NV12 role mapping must cover exactly the runtime input names.")
    if any(name not in names for name in role_map.values()):
        raise TensorContractError("NV12 role mapping references an unknown input name.")
    primary_name = role_map["image"] if packed else role_map["y"]
    if primary_name not in shapes:
        raise TensorContractError(f"Runtime reports input {primary_name!r} without a shape.")
    height, width, primary_layout = _shape_for_input(
        shapes[primary_name], packed=packed, override=override, label=primary_name)
    _validate_nv12_geometry(height, width, primary_name)
    if override is not None and (height, width) != override:
        raise TensorContractError("input_shape_override conflicts with model metadata.")

    layouts = {primary_name: primary_layout}
    if not packed:
        uv_name = role_map["uv"]
        if uv_name not in shapes:
            raise TensorContractError(f"Runtime reports input {uv_name!r} without a shape.")
        uv_shape = shapes[uv_name]
        if uv_shape != (1, height // 2, width // 2, 2):
            raise TensorContractError(
                f"Split NV12 chroma input {uv_name!r} must have shape "
                f"(1, {height // 2}, {width // 2}, 2), got {uv_shape}.")
        layouts[uv_name] = "NHWC"

    consumed = {name: dtype_map.get(name) for name in names}
    return InputBinding(model_name=str(model_name), roles=role_map, shapes=shapes,
                        dtypes=consumed, packed=packed, height=height, width=width,
                        layouts=layouts)


@dataclass(frozen=True)
class Quantization:
    """Explicit affine quantization parameters supplied by model metadata."""

    scale: float
    zero_point: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale == 0:
            raise TensorContractError("quantization scale must be finite and non-zero.")
        if not np.isfinite(self.zero_point):
            raise TensorContractError("quantization zero_point must be finite.")

    def apply(self, array: np.ndarray) -> np.ndarray:
        return (np.asarray(array, dtype=np.float32) - self.zero_point) * self.scale


def as_quantization(value: Any) -> Optional[Quantization]:
    """Parse an explicit quantization object or mapping."""
    if value is None:
        return None
    if isinstance(value, Quantization):
        return value
    if isinstance(value, Mapping):
        scale = value.get("scale")
        zero = value.get("zero_point", 0.0)
        if np.ndim(scale) != 0 or np.ndim(zero) != 0:
            raise TensorContractError("Per-channel quantization is not declared by this contract.")
        if scale is None:
            raise TensorContractError("Quantization metadata must provide a scalar 'scale'.")
        return Quantization(float(scale), float(zero))
    raise TensorContractError(
        "Quantization metadata must be Quantization or a mapping with 'scale' "
        "and optional 'zero_point'.")


def pack_nv12_single(binding: InputBinding,
                     y_plane: np.ndarray,
                     uv_plane: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Pack planes for a binding whose reviewed input protocol is packed NV12."""
    if not isinstance(binding, InputBinding):
        raise TensorContractError("pack_nv12_single expects an InputBinding.")
    if not binding.packed:
        raise TensorContractError("The bound model requires separate Y/UV planes.")
    return binding.build(y_plane, uv_plane)


def pack_nv12_planes(binding: InputBinding,
                     y_plane: np.ndarray,
                     uv_plane: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Pack planes for a binding whose reviewed input protocol is split NV12."""
    if not isinstance(binding, InputBinding):
        raise TensorContractError("pack_nv12_planes expects an InputBinding.")
    if binding.packed:
        raise TensorContractError("The bound model requires one packed NV12 input.")
    return binding.build(y_plane, uv_plane)


@dataclass(frozen=True)
class OutputBinding:
    """Map physical runtime outputs to named semantic roles."""

    model_name: str
    role_to_name: Mapping[str, str]
    shapes: Mapping[str, Tuple[int, ...]]
    dtypes: Mapping[str, Optional[np.dtype]]
    expected_shapes: Mapping[str, Tuple[int, ...]]
    channels: Mapping[str, int]
    layouts: Mapping[str, str] = field(default_factory=dict)
    quantization: Mapping[str, Optional[Quantization]] = field(default_factory=dict)
    runtime_order: Tuple[str, ...] = ()

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.role_to_name.values())

    def output_name(self, kind: str, stride: int) -> str:
        role = f"{kind}_{int(stride)}"
        try:
            return self.role_to_name[role]
        except KeyError as exc:
            raise TensorContractError(f"No output role {role!r} is bound.") from exc

    def _unwrap(self, outputs: Any) -> Any:
        if isinstance(outputs, Mapping) and self.model_name in outputs:
            return outputs[self.model_name]
        return outputs

    def read(self, outputs: Any) -> Dict[str, np.ndarray]:
        """Validate runtime output tensors and return role-keyed arrays."""
        values = self._unwrap(outputs)
        if isinstance(values, Mapping):
            by_name = values
        elif isinstance(values, (tuple, list)):
            order = self.runtime_order or tuple(self.names)
            if len(values) != len(order):
                raise TensorContractError(
                    f"Runtime returned {len(values)} outputs; expected {len(order)}.")
            by_name = dict(zip(order, values))
        else:
            raise TensorContractError("Runtime outputs must be a mapping or sequence.")
        unexpected = set(by_name) - set(self.names)
        if unexpected:
            raise TensorContractError(
                "Runtime returned unbound output(s): " + ", ".join(sorted(map(str, unexpected))))

        result: Dict[str, np.ndarray] = {}
        for role, name in self.role_to_name.items():
            if name not in by_name:
                raise TensorContractError(
                    f"Runtime output {name!r} for role {role!r} is missing.")
            value = np.asarray(by_name[name])
            declared_dtype = self.dtypes.get(name)
            if declared_dtype is not None and value.dtype != declared_dtype:
                raise TensorContractError(
                    f"Output {name!r} reports dtype {value.dtype}; metadata declares "
                    f"{declared_dtype}.")
            expected_shape = self.expected_shapes[role]
            if tuple(value.shape) != expected_shape:
                raise TensorContractError(
                    f"Output {name!r} for role {role!r} has shape {tuple(value.shape)}; "
                    f"expected {expected_shape}.")
            if not np.issubdtype(value.dtype, np.number):
                raise TensorContractError(
                    f"Output {name!r} for role {role!r} must be numeric, got {value.dtype}.")
            if not np.all(np.isfinite(value)):
                raise TensorContractError(
                    f"Output {name!r} for role {role!r} contains NaN or infinity.")
            quant = self.quantization.get(role)
            if quant is not None:
                value = quant.apply(value)
            elif not np.issubdtype(value.dtype, np.floating):
                raise TensorContractError(
                    f"Output {name!r} is {value.dtype}; no explicit quantization is "
                    "declared for this semantic tensor.")
            result[role] = _normalise_output_layout(
                value, self.layouts.get(role, "NHWC"))
        return result


def _normalise_output_layout(value: np.ndarray,
                             layout: str) -> np.ndarray:
    """Return output as batch-one NHWC while preserving semantic values."""
    layout = str(layout).upper()
    array = np.asarray(value)
    if layout == "NHWC":
        return array
    raise TensorContractError(
        f"Unsupported output layout {layout!r}; this contract requires NHWC.")


def read_output(binding: OutputBinding, outputs: Any) -> Dict[str, np.ndarray]:
    """Functional alias for :meth:`OutputBinding.read`."""
    return binding.read(outputs)


__all__ = [
    "TensorContractError",
    "normalize_shape",
    "normalize_dtype",
    "element_count",
    "InputBinding",
    "bind_nv12_inputs",
    "Quantization",
    "as_quantization",
    "pack_nv12_single",
    "pack_nv12_planes",
    "OutputBinding",
    "read_output",
]
