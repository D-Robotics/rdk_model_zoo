# Copyright (c) 2025-2026 D-Robotics Corporation
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
"""Physical transport, tensor binding and runtime metadata for YOLO tasks.

This is the shared backend every YOLO task file builds on: validated NV12
input/output tensor adapters and contracts, the logical DFL/LTRB/
segmentation/pose/OBB/classification contract dataclasses, the lazy
``ModelRunner`` that loads one artifact through ``utils.py_utils.runtime.RuntimeSession``,
the strict metadata-derived ``Nv12InputAdapter``, and the lazy board-runtime
bridge.  Nothing here decodes task outputs, renders results or parses
arguments; the task files (``detect.py``/``segment.py``/``pose.py``/
``obb.py``/``classify.py``) own the numeric pipelines and ``cli.py`` owns
platform/asset selection and presentation.
"""

from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.py_utils.model_runner import _default_runtime_factory
from samples.vision.ultralytics_yolo.runtime.python.cli import PlatformProfile

# ====================================================================
# Validated tensor adapters (input binding, dtype/shape normalisation, raw output reading).
# ====================================================================

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
    from utils.py_utils.runtime_meta import canonicalise_dtype
    canonical = canonicalise_dtype(value)
    if canonical in {"float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"}:
        return np.dtype(canonical)
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
                     label: str, allow_packed_nhwc: bool = False) -> Tuple[int, int, str]:
    """Derive ``(height, width, layout)`` from one input shape."""
    shape = normalize_shape(shape, f"input {label!r} shape")
    if len(shape) == 4 and shape[0] != 1:
        raise TensorContractError("Only batch-one NV12 inputs are supported.")

    if packed:
        if len(shape) == 4 and shape[1] == 3:
            return shape[2], shape[3], "NCHW"
        if allow_packed_nhwc and len(shape) == 4 and shape[3] == 3:
            return shape[1], shape[2], "NHWC"
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
                     input_shape_override: Optional[Sequence[int]] = None,
                     allow_packed_nhwc: bool = False) -> InputBinding:
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
        shapes[primary_name], packed=packed, override=override, label=primary_name,
        allow_packed_nhwc=allow_packed_nhwc)
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


def require_floating_output(dtype, descriptor, name):
    """Accept runtime-dequantized floating outputs only; never implement SCALE here."""
    kind = None
    if descriptor is not None:
        if isinstance(descriptor, Mapping):
            kind = descriptor.get("quant_type", "SCALE")
        else:
            kind = getattr(descriptor, "quant_type", "SCALE")
        kind = str(getattr(kind, "name", kind))
    if dtype is None or not np.issubdtype(dtype, np.floating) or kind not in (None, "NONE", "0"):
        raise TensorContractError(
            f"Output {name!r} requires an already dequantized floating-point tensor "
            "with NONE/no quantization metadata. Select the maintained Ultralytics "
            "floating-output artifact; manual output dequantization is not supported.")


@dataclass(frozen=True)
class RawOutputs(Mapping[str, np.ndarray]):
    """Role-keyed raw arrays plus their binding; values/layout remain SDK-native.

    The immutable mapping borrows array buffers until the next SDK call. Finish
    postprocessing before reusing the runner, or explicitly copy arrays to retain
    them. This is not a promise of SDK thread safety or concurrent ownership.
    """
    arrays: Mapping[str, np.ndarray]
    binding: Any

    def __post_init__(self):
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))

    def __getitem__(self, role):
        return self.arrays[role]

    def __iter__(self):
        return iter(self.arrays)

    def __len__(self):
        return len(self.arrays)


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

    def read_raw(self, outputs: Any) -> RawOutputs:
        """Validate physical arrays and bind roles without numeric/layout transforms."""
        if isinstance(outputs, RawOutputs):
            if outputs.binding is not self:
                raise TensorContractError("Raw outputs belong to a different model binding.")
            if set(outputs) != set(self.role_to_name):
                raise TensorContractError("Raw output roles do not match the binding.")
            outputs = {name: outputs[role] for role, name in self.role_to_name.items()}
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
            if not np.issubdtype(value.dtype, np.floating):
                raise TensorContractError(
                    f"Output {name!r} must already be floating, got {value.dtype}.")
            result[role] = value
        return RawOutputs(result, self)

    def read(self, outputs: Any) -> Dict[str, np.ndarray]:
        """Explicit postprocess adapter: validate floating arrays and normalize layout.

        Kept for direct decoder/binding callers. ModelRunner uses read_raw instead.
        A RawOutputs carrier prevents semantic role names from bypassing declared
        physical validation and prevents ambiguity with injected semantic maps.
        """
        raw = self.read_raw(outputs)
        if raw.binding is not self:
            raise TensorContractError("Raw outputs belong to a different model binding.")
        return {role: _normalise_output_layout(value, self.layouts.get(role, "NHWC"))
                for role, value in raw.items()}


def _normalise_output_layout(value: np.ndarray,
                             layout: str) -> np.ndarray:
    """Return output as batch-one NHWC while preserving semantic values."""
    layout = str(layout).upper()
    array = np.asarray(value)
    if layout == "NHWC":
        return array
    if layout == "NCHW" and array.ndim == 4:
        return array.transpose(0, 2, 3, 1)
    raise TensorContractError(
        f"Unsupported output layout {layout!r}; this contract requires NHWC.")


def read_output(binding: OutputBinding, outputs: Any) -> Dict[str, np.ndarray]:
    """Functional alias for :meth:`OutputBinding.read`."""
    return binding.read(outputs)

# ====================================================================
# Runtime metadata view and the logical tensor contracts.
# ====================================================================

class BindingError(ValueError):
    """Raised when an artifact cannot be safely bound to the task."""


def _role_key(kind: str, stride: int) -> str:
    return f"{str(kind).lower()}_{int(stride)}"


def _normalise_roles(roles: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Copy the reviewed flat semantic-role map without guessing spellings."""
    if not roles:
        return {}
    result: Dict[str, str] = {}
    for key, value in roles.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise BindingError("Detection output_roles must map string roles to string tensor names.")
        result[key] = value
    return result


@dataclass(frozen=True, init=False)
class DFLDetectionContract:
    """Logical DFL detection semantics for the source-supported YOLO heads.

    ``output_roles`` maps semantic role names such as ``cls_8`` and ``box_8``
    to physical runtime names.  Role names are intentionally exact: an alias
    or nested shorthand is rejected instead of being interpreted as a new
    protocol.
    """

    task: str
    classes: int
    reg_bins: int
    strides: Tuple[int, ...]
    classification: str
    box_distribution: str
    nms: str
    input_roles: Mapping[str, str]
    output_roles: Mapping[str, str]
    output_layouts: Mapping[str, str]

    def __init__(self,
                 classes: int = 80,
                 reg_bins: int = 16,
                 strides: Sequence[int] = (8, 16, 32),
                 classification: str = "logits",
                 box_distribution: str = "logits",
                 nms: str = "classwise",
                 input_roles: Optional[Mapping[str, str]] = None,
                 output_roles: Optional[Mapping[str, str]] = None,
                 output_layouts: Optional[Mapping[str, str]] = None,
                 task: str = "detect") -> None:
        classes = int(classes)
        reg_bins = int(reg_bins)
        strides = tuple(int(stride) for stride in strides)
        if classes <= 0:
            raise BindingError("DFL contract classes must be positive.")
        if reg_bins <= 0:
            raise BindingError("DFL contract reg_bins must be positive.")
        if not strides or any(stride <= 0 for stride in strides) or len(set(strides)) != len(strides):
            raise BindingError("DFL contract strides must be distinct positive values.")
        classification = str(classification).lower()
        box_distribution = str(box_distribution).lower()
        nms = str(nms).lower()
        if classification not in {"logits", "probabilities"}:
            raise BindingError("classification must be 'logits' or 'probabilities'.")
        if box_distribution not in {"logits", "probabilities"}:
            raise BindingError("box_distribution must be 'logits' or 'probabilities'.")
        if nms not in {"none", "classwise", "agnostic"}:
            raise BindingError("nms must be 'none', 'classwise', or 'agnostic'.")
        object.__setattr__(self, "task", str(task))
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "reg_bins", reg_bins)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "box_distribution", box_distribution)
        object.__setattr__(self, "nms", nms)
        object.__setattr__(self, "input_roles", {
            str(key): str(value) for key, value in (input_roles or {}).items()
        })
        object.__setattr__(self, "output_roles", _normalise_roles(output_roles))
        layouts = output_layouts or {
            _role_key(kind, stride): "NHWC"
            for stride in strides
            for kind in ("cls", "box")
        }
        object.__setattr__(self, "output_layouts", {
            str(key): str(value) for key, value in layouts.items()
        })

    @property
    def required_roles(self) -> Tuple[str, ...]:
        return tuple(role for stride in self.strides
                     for role in (_role_key("cls", stride), _role_key("box", stride)))


@dataclass(frozen=True, init=False)
class DFLPoseContract(DFLDetectionContract):
    """Single-person-class DFL pose heads with 17 COCO (x,y,logit) triplets."""

    nkpt: int

    def __init__(self, reg_bins=16, strides=(8, 16, 32), nkpt=17,
                 input_roles=None, output_roles=None, output_layouts=None):
        if int(reg_bins) != 16 or tuple(strides) != (8, 16, 32) or int(nkpt) != 17:
            raise BindingError("DFL pose requires 16 bins, strides 8/16/32 and 17 COCO keypoints.")
        super().__init__(classes=1, reg_bins=reg_bins, strides=strides, task="pose",
                         input_roles=input_roles, output_roles=output_roles,
                         output_layouts=output_layouts)
        object.__setattr__(self, "nkpt", int(nkpt))

    @property
    def required_roles(self):
        return tuple(_role_key(kind, stride) for stride in self.strides
                     for kind in ("cls", "box", "kpts"))


@dataclass(frozen=True, init=False)
class DFLSegmentationContract(DFLDetectionContract):
    """Source YOLOv8/9/11 DFL heads plus 32 mask coefficients and stride-4 prototypes.

    Head tensors are NHWC. Prototypes may be NHWC or NCHW; quantization
    is not implemented here: outputs must already be floating. Ambiguous shapes need an
    explicit reviewed output_roles map, never runtime enumeration order.
    """

    mces_num: int

    def __init__(self, classes=80, reg_bins=16, strides=(8, 16, 32),
                 mces_num=32, input_roles=None, output_roles=None,
                 output_layouts=None):
        if int(reg_bins) != 16 or tuple(strides) != (8, 16, 32) or int(mces_num) != 32:
            raise BindingError("DFL segmentation requires 16 bins, strides 8/16/32 and 32 mask coefficients.")
        super().__init__(classes=classes, reg_bins=reg_bins, strides=strides,
                         task="segment", input_roles=input_roles, output_roles=output_roles,
                         output_layouts=output_layouts)
        object.__setattr__(self, "mces_num", int(mces_num))

    @property
    def required_roles(self):
        return tuple(_role_key(kind, stride) for stride in self.strides
                     for kind in ("cls", "box", "mces")) + ("protos",)


@dataclass(frozen=True, init=False)
class LTRBDetectionContract:
    """Logical direct-offset semantics for the reviewed YOLO26 detect head.

    YOLO26 exposes one class-logit tensor and one four-channel LTRB tensor at
    each stride.  The contract intentionally has no DFL-bin or quantization
    setting: the observed artifacts are floating point tensors whose box
    values are already direct grid-cell distances.
    """

    task: str
    classes: int
    strides: Tuple[int, ...]
    classification: str
    box_encoding: str
    nms: str
    input_roles: Mapping[str, str]
    output_roles: Mapping[str, str]
    output_layouts: Mapping[str, str]

    def __init__(self,
                 classes: int = 80,
                 strides: Sequence[int] = (8, 16, 32),
                 classification: str = "logits",
                 box_encoding: str = "ltrb",
                 nms: str = "classwise",
                 input_roles: Optional[Mapping[str, str]] = None,
                 output_roles: Optional[Mapping[str, str]] = None,
                 output_layouts: Optional[Mapping[str, str]] = None,
                 task: str = "detect") -> None:
        classes = int(classes)
        strides = tuple(int(stride) for stride in strides)
        classification = str(classification).lower()
        box_encoding = str(box_encoding).lower()
        nms = str(nms).lower()
        task = str(task).lower()
        if classes <= 0:
            raise BindingError("LTRB contract classes must be positive.")
        if not strides or any(stride <= 0 for stride in strides) or len(set(strides)) != len(strides):
            raise BindingError("LTRB contract strides must be distinct positive values.")
        if strides != (8, 16, 32):
            raise BindingError("LTRB contract supports only strides 8, 16 and 32.")
        if classification != "logits":
            raise BindingError("LTRB classification must be 'logits'.")
        if box_encoding != "ltrb":
            raise BindingError("LTRB box_encoding must be 'ltrb'.")
        if nms != "classwise":
            raise BindingError("LTRB NMS must be 'classwise'.")
        if task != "detect":
            raise BindingError("LTRB contract task must be 'detect'.")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "box_encoding", box_encoding)
        object.__setattr__(self, "nms", nms)
        object.__setattr__(self, "input_roles", {
            str(key): str(value) for key, value in (input_roles or {}).items()
        })
        object.__setattr__(self, "output_roles", _normalise_roles(output_roles))
        layouts = output_layouts or {
            _role_key(kind, stride): "NHWC"
            for stride in strides
            for kind in ("cls", "box")
        }
        object.__setattr__(self, "output_layouts", {
            str(key): str(value) for key, value in layouts.items()
        })

    @property
    def box_channels(self) -> int:
        return 4

    @property
    def protocol(self) -> str:
        return "LTRB"

    @property
    def required_roles(self) -> Tuple[str, ...]:
        return tuple(role for stride in self.strides
                     for role in (_role_key("cls", stride), _role_key("box", stride)))


@dataclass(frozen=True, init=False)
class LTRBPoseContract(LTRBDetectionContract):
    """YOLO26: direct LTRB distances and COCO-17 grid-relative (x,y,logit)."""

    nkpt: int

    def __init__(self, strides=(8, 16, 32), input_roles=None,
                 output_roles=None, output_layouts=None):
        super().__init__(classes=1, strides=strides, input_roles=input_roles,
                         output_roles=output_roles, output_layouts=output_layouts)
        object.__setattr__(self, "task", "pose")
        object.__setattr__(self, "nkpt", 17)

    @property
    def required_roles(self):
        return tuple(_role_key(kind, stride) for stride in self.strides
                     for kind in ("cls", "box", "kpts"))


@dataclass(frozen=True, init=False)
class LTRBSegmentationContract(LTRBDetectionContract):
    """YOLO26 direct LTRB heads with 32 coefficients and stride-4 prototypes."""

    mces_num: int

    def __init__(self, classes=80, strides=(8, 16, 32), input_roles=None,
                 output_roles=None, output_layouts=None):
        super().__init__(classes=classes, strides=strides, input_roles=input_roles,
                         output_roles=output_roles, output_layouts=output_layouts)
        object.__setattr__(self, "task", "segment")
        object.__setattr__(self, "mces_num", 32)

    @property
    def required_roles(self):
        return tuple(_role_key(kind, stride) for stride in self.strides
                     for kind in ("cls", "box", "mces")) + ("protos",)


@dataclass(frozen=True, init=False)
class LTRBOBBContract(LTRBDetectionContract):
    """YOLO26 direct rotated distances plus one angle in radians per anchor."""

    def __init__(self, classes=15, strides=(8, 16, 32), input_roles=None,
                 output_roles=None, output_layouts=None):
        super().__init__(classes=classes, strides=strides, input_roles=input_roles,
                         output_roles=output_roles, output_layouts=output_layouts)
        object.__setattr__(self, "task", "obb")

    @property
    def required_roles(self):
        return tuple(_role_key(kind, stride) for stride in self.strides
                     for kind in ("cls", "box", "angle"))


@dataclass(frozen=True)
class ClassificationContract:
    """One floating logit vector; singleton physical axes carry no extra samples."""

    classes: int = 1000
    task: str = "classify"
    input_roles: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.classes, bool) or not isinstance(self.classes, int) or self.classes < 2:
            raise BindingError("Classification classes must be an integer greater than one.")
        if self.task != "classify":
            raise BindingError("ClassificationContract task must be 'classify'.")


@dataclass(frozen=True)
class ModelSelection:
    """Read-only selection passed from an entrypoint to the runner factory."""

    model_path: str
    target: Optional[str] = None
    task: str = "detect"
    contract: Optional[Any] = None
    input_shape: Optional[Tuple[int, int]] = None
    family: Optional[str] = None
    artifact_id: Optional[str] = None
    platform: Any = None

    def __post_init__(self) -> None:
        if not self.model_path:
            raise BindingError("model_path is required for a model selection.")
        if self.input_shape is not None:
            if len(self.input_shape) != 2 or any(int(value) <= 0 for value in self.input_shape):
                raise BindingError("input_shape must contain positive (height, width).")
            object.__setattr__(self, "input_shape", (int(self.input_shape[0]),
                                                       int(self.input_shape[1])))
        if self.contract is not None:
            if getattr(self.contract, "task", None) == "classify":
                if not all(hasattr(self.contract, name) for name in ("classes", "input_roles")):
                    raise BindingError("Classification contract requires classes and input_roles.")
                return
            required = ("classes", "strides", "required_roles")
            if getattr(self.contract, "box_encoding", None) == "ltrb":
                required += ("box_channels",)
            else:
                required += ("reg_bins",)
            if not all(hasattr(self.contract, name) for name in required):
                raise BindingError(
                    "selection.contract does not expose the selected detection protocol fields.")

    @property
    def profile(self) -> Any:
        return self.platform if self.platform is not None else self.target


def default_dfl_contract(classes: int = 80,
                         reg_bins: int = 16,
                         strides: Sequence[int] = (8, 16, 32)) -> DFLDetectionContract:
    """Return source-supported DFL semantics without fabricating asset names."""
    return DFLDetectionContract(classes=classes, reg_bins=reg_bins, strides=strides)


def resolve_selection(model_path: str,
                      target: Optional[str] = None,
                      task: str = "detect",
                      contract: Optional[Any] = None,
                      input_shape: Optional[Sequence[int]] = None,
                      family: Optional[str] = None,
                      artifact_id: Optional[str] = None,
                      platform: Any = None) -> ModelSelection:
    """Resolve a local model selection without importing the SDK or networking."""
    if contract is None and task == "detect":
        contract = default_dfl_contract()
    shape = None if input_shape is None else (int(input_shape[0]), int(input_shape[1]))
    return ModelSelection(model_path=model_path, target=target, task=task,
                          contract=contract, input_shape=shape, family=family,
                          artifact_id=artifact_id, platform=platform)


@dataclass(frozen=True)
class ModelBinding:
    """Validated input/output adapters for one loaded runtime model."""

    selection: ModelSelection
    contract: Any
    metadata: Any
    input_adapter: InputBinding
    output_adapter: OutputBinding

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def input_names(self) -> Tuple[str, ...]:
        return self.input_adapter.input_names

    @property
    def output_roles(self) -> Mapping[str, str]:
        return self.output_adapter.role_to_name

    @property
    def output_names(self) -> Tuple[str, ...]:
        return tuple(self.metadata.output_names)

    def output_name(self, kind: str, stride: int) -> str:
        return self.output_adapter.output_name(kind, stride)

    def grid_shape(self, stride: int) -> Tuple[int, int]:
        try:
            return self.input_adapter.expected_anchor_sizes((int(stride),))[0]
        except TensorContractError as exc:
            raise BindingError(str(exc)) from exc

    def read_raw_outputs(self, outputs: Any):
        """Validate and bind raw output roles without numeric transformation."""
        try:
            return self.output_adapter.read_raw(outputs)
        except TensorContractError as exc:
            raise BindingError(str(exc)) from exc

    def read_outputs(self, outputs: Any) -> Dict[str, np.ndarray]:
        try:
            return self.output_adapter.read(outputs)
        except TensorContractError as exc:
            raise BindingError(str(exc)) from exc


def _selection_profile(selection: ModelSelection) -> Any:
    profile = selection.profile
    if profile is None:
        raise BindingError("A concrete target profile is required to bind NV12 inputs.")
    if isinstance(profile, str):
        try:
            from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
            return resolve_platform(profile)
        except Exception as exc:  # board identity is handled by the caller
            raise BindingError(f"Unknown target profile {profile!r}.") from exc
    return profile


def _expected_grid(adapter: InputBinding, stride: int) -> Tuple[int, int]:
    try:
        return adapter.expected_anchor_sizes((stride,))[0]
    except TensorContractError as exc:
        raise BindingError(str(exc)) from exc


def _role_shape_descriptor(shape: Tuple[int, ...],
                           grid: Tuple[int, int],
                           channels: int,
                           name: str, allow_nchw: bool = False) -> Tuple[str, Tuple[int, ...]]:
    """Validate a physical shape and return its layout."""
    gh, gw = grid
    if shape == (1, gh, gw, channels):
        return "NHWC", shape
    if allow_nchw and shape == (1, channels, gh, gw):
        return "NCHW", shape
    raise BindingError(
        f"Output {name!r} shape {shape} is incompatible with grid {gh}x{gw} and "
        f"{channels} channels; this contract requires NHWC.")


def _runtime_quantization(metadata: Any, name: str) -> Any:
    values = getattr(metadata, "output_quantization", {}) or {}
    return values.get(name) if isinstance(values, Mapping) else None


def _bind_classification_output(contract, metadata):
    """Reject extra outputs, batches, spatial maps, integer logits and missing metadata."""
    names = tuple(metadata.output_names)
    if len(names) != 1:
        raise BindingError("Classification requires exactly one logit output.")
    name = names[0]
    if name not in metadata.output_shapes or name not in metadata.output_dtypes:
        raise BindingError("Classification requires output shape and dtype metadata.")
    try:
        shape = normalize_shape(metadata.output_shapes[name], "classification output")
        dtype = normalize_dtype(metadata.output_dtypes[name])
        require_floating_output(dtype, _runtime_quantization(metadata, name), name)
    except TensorContractError as exc:
        raise BindingError(str(exc)) from exc
    if (not 1 <= len(shape) <= 4 or (len(shape) > 1 and shape[0] != 1)
            or tuple(n for n in shape if n != 1) != (contract.classes,)):
        raise BindingError(f"Classification output must contain one {contract.classes}-class vector, got {shape}.")
    return OutputBinding(model_name=metadata.model_name, role_to_name={"logits": name},
                         shapes={name: shape}, dtypes={name: dtype},
                         expected_shapes={"logits": shape}, channels={"logits": contract.classes},
                         layouts={"logits": "NHWC"}, runtime_order=names)


def _bind_output_roles(selection: ModelSelection,
                       contract: Any,
                       metadata: Any,
                       adapter: InputBinding) -> OutputBinding:
    protocol = str(getattr(contract, "protocol", "DFL"))
    box_channels = int(getattr(
        contract, "box_channels", 4 * int(getattr(contract, "reg_bins", 16))))
    specs = []
    kinds = [("cls", contract.classes), ("box", box_channels)]
    if contract.task == "segment":
        kinds.append(("mces", contract.mces_num))
    elif contract.task == "pose":
        kinds.append(("kpts", 3 * contract.nkpt))
    elif contract.task == "obb":
        kinds.append(("angle", 1))
    for stride in contract.strides:
        grid = _expected_grid(adapter, stride)
        specs.extend((_role_key(kind, stride), grid, channels, False)
                     for kind, channels in kinds)
    if contract.task == "segment":
        specs.append(("protos", _expected_grid(adapter, 4), contract.mces_num, True))
    names = tuple(str(name) for name in metadata.output_names)
    if len(set(names)) != len(names):
        raise BindingError("Runtime output names must be unique.")
    if len(names) != len(contract.required_roles):
        raise BindingError(
            f"{protocol} detection contract requires {len(contract.required_roles)} outputs; "
            f"runtime reports {len(names)}.")
    shapes = {str(name): normalize_shape(shape, f"output {name!r} shape")
              for name, shape in (getattr(metadata, "output_shapes", {}) or {}).items()}
    dtypes = {str(name): normalize_dtype(value)
              for name, value in (getattr(metadata, "output_dtypes", {}) or {}).items()}
    has_shapes = all(name in shapes for name in names)
    has_dtypes = all(name in dtypes for name in names)
    if not has_shapes:
        missing = [name for name in names if name not in shapes]
        raise BindingError(
            f"Runtime output shape metadata is missing for: {', '.join(missing)}.")
    if not has_dtypes:
        missing = [name for name in names if name not in dtypes]
        raise BindingError(
            f"Runtime output dtype metadata is missing for: {', '.join(missing)}.")

    role_to_name = _normalise_roles(contract.output_roles)
    required = set(contract.required_roles)
    if role_to_name:
        if set(role_to_name) != required:
            missing = sorted(required - set(role_to_name))
            extra = sorted(set(role_to_name) - required)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if extra:
                detail.append(f"unknown {', '.join(extra)}")
            raise BindingError("Output role mapping is incomplete (" + "; ".join(detail) + ").")
        if len(set(role_to_name.values())) != len(role_to_name):
            raise BindingError(
                f"Each {protocol} semantic output role must name a distinct tensor.")
        unknown = sorted(set(role_to_name.values()) - set(names))
        if unknown:
            raise BindingError(
                f"Output role mapping references runtime name(s) not reported: {', '.join(unknown)}.")
    else:
        # Compiler names are opaque, so their physical names are matched to
        # the selected protocol by complete shape metadata.  Every assignment
        # must be unique; runtime enumeration order is never consulted.
        for role, grid, channels, allow_nchw in specs:
            candidates = []
            for name in names:
                try:
                    _role_shape_descriptor(shapes[name], grid, channels, name, allow_nchw)
                except BindingError:
                    continue
                candidates.append(name)
            if len(candidates) != 1:
                if not candidates:
                    raise BindingError(
                        f"No uniquely identifiable tensor for {protocol} role {role!r}; "
                        f"expected grid {grid[0]}x{grid[1]} with {channels} channels.")
                raise BindingError(
                    f"Multiple tensors match {protocol} role {role!r}: {', '.join(candidates)}.")
            role_to_name[role] = candidates[0]
        if len(set(role_to_name.values())) != len(specs):
            raise BindingError("Each semantic output role must name a distinct tensor.")

    expected_shapes: Dict[str, Tuple[int, ...]] = {}
    layouts: Dict[str, str] = {}
    channels_map: Dict[str, int] = {}
    for role, grid, channels, allow_nchw in specs:
        name = role_to_name[role]
        channels_map[role] = channels
        layout, physical_shape = _role_shape_descriptor(
            shapes[name], grid, channels, name, allow_nchw)
        declared_layout = contract.output_layouts.get(role)
        if declared_layout is not None and str(declared_layout).upper() != layout:
            raise BindingError(
                f"Output role {role!r} declares {declared_layout} but runtime shape "
                f"{shapes.get(name)} is {layout}.")
        layouts[role] = layout
        expected_shapes[role] = physical_shape
        dtype = dtypes.get(name)
        try:
            require_floating_output(dtype, _runtime_quantization(metadata, name), name)
        except TensorContractError as exc:
            raise BindingError(str(exc)) from exc

    return OutputBinding(
        model_name=str(metadata.model_name),
        role_to_name=role_to_name,
        shapes=shapes,
        dtypes=dtypes,
        expected_shapes=expected_shapes,
        channels=channels_map,
        layouts=layouts,
        runtime_order=names,
    )


def bind_model(selection: ModelSelection,
               metadata: Any) -> ModelBinding:
    """Validate a selection against actual model metadata and build adapters."""
    if not isinstance(selection, ModelSelection):
        # Top-level imports remain a supported test/library spelling while the
        # implementation itself uses the package-qualified module.  Coerce the
        # value at this seam instead of coupling every module to duplicate types.
        if not all(hasattr(selection, name)
                   for name in ("model_path", "task", "contract")):
            raise BindingError("bind_model expects a ModelSelection instance.")
        selection = ModelSelection(
            model_path=selection.model_path,
            target=getattr(selection, "target", None),
            task=selection.task,
            contract=selection.contract,
            input_shape=getattr(selection, "input_shape", None),
            family=getattr(selection, "family", None),
            artifact_id=getattr(selection, "artifact_id", None),
            platform=getattr(selection, "platform", None),
        )
    contract = selection.contract or default_dfl_contract()
    if contract.task != selection.task:
        raise BindingError(
            f"Selection task {selection.task!r} conflicts with contract task {contract.task!r}.")
    if not hasattr(metadata, "model_name"):
        raise BindingError("Runtime metadata must provide model_name.")
    profile = _selection_profile(selection)
    try:
        input_adapter = bind_nv12_inputs(
            profile,
            metadata.model_name,
            metadata.input_names,
            metadata.input_shapes,
            getattr(metadata, "input_dtypes", {}),
            roles=contract.input_roles or None,
            input_shape_override=selection.input_shape,
            allow_packed_nhwc=getattr(contract, "allow_packed_nhwc", False),
        )
    except TensorContractError as exc:
        raise BindingError(str(exc)) from exc
    output_adapter = (_bind_classification_output(contract, metadata)
                      if contract.task == "classify" else
                      _bind_output_roles(selection, contract, metadata, input_adapter))
    return ModelBinding(selection=selection, contract=contract, metadata=metadata,
                        input_adapter=input_adapter, output_adapter=output_adapter)


@dataclass(frozen=True)
class RuntimeMetadata:
    """Small normalized view of the runtime descriptors used by binding."""

    model_name: str
    input_names: Tuple[str, ...]
    input_shapes: Mapping[str, Tuple[int, ...]]
    output_names: Tuple[str, ...]
    output_shapes: Mapping[str, Tuple[int, ...]]
    input_dtypes: Mapping[str, Optional[np.dtype]] = field(default_factory=dict)
    output_dtypes: Mapping[str, Optional[np.dtype]] = field(default_factory=dict)
    output_quantization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", str(self.model_name))
        object.__setattr__(self, "input_names", tuple(str(name) for name in self.input_names))
        object.__setattr__(self, "output_names", tuple(str(name) for name in self.output_names))
        object.__setattr__(self, "input_shapes", {
            str(name): normalize_shape(shape, f"input {name!r} shape")
            for name, shape in self.input_shapes.items()
        })
        object.__setattr__(self, "output_shapes", {
            str(name): normalize_shape(shape, f"output {name!r} shape")
            for name, shape in (self.output_shapes or {}).items()
        })
        object.__setattr__(self, "input_dtypes", {
            str(name): normalize_dtype(value)
            for name, value in (self.input_dtypes or {}).items()
        })
        object.__setattr__(self, "output_dtypes", {
            str(name): normalize_dtype(value)
            for name, value in (self.output_dtypes or {}).items()
        })

    @property
    def complete(self) -> bool:
        return (set(self.input_names) <= set(self.input_shapes) and
                set(self.output_names) <= set(self.output_shapes) and
                set(self.input_names) <= set(self.input_dtypes) and
                set(self.output_names) <= set(self.output_dtypes))

    @classmethod
    def from_runtime(cls, runtime: Any, model_name: Optional[str] = None) -> "RuntimeMetadata":
        """Extract the descriptor attributes exposed by current RDK runtimes."""
        names = list(getattr(runtime, "model_names", ()) or ())
        if model_name is None:
            model_name = str(names[0]) if names else None
        if model_name is None:
            raise BindingError("Runtime reports no model name.")

        def per_model(attribute: str, default: Any):
            value = getattr(runtime, attribute, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        def per_tensor(attribute: str, default: Any):
            value = per_model(attribute, default)
            return value if isinstance(value, Mapping) else {}

        input_names = tuple(str(name) for name in (per_model("input_names", ()) or ()))
        output_names = tuple(str(name) for name in (per_model("output_names", ()) or ()))
        input_shapes = per_tensor("input_shapes", {})
        output_shapes = per_tensor("output_shapes", {})
        input_dtypes = per_tensor("input_dtypes", {})
        if not input_dtypes:
            input_dtypes = per_tensor("input_dtype", {})
        output_dtypes = per_tensor("output_dtypes", {})
        if not output_dtypes:
            output_dtypes = per_tensor("output_dtype", {})
        quant = per_tensor("output_quants", {})
        if not quant:
            quant = per_tensor("output_quantization", {})
        if not quant:
            quant = per_tensor("output_quant_infos", {})
        if not quant:
            quant = per_tensor("quant_infos", {})
        return cls(model_name=str(model_name), input_names=input_names,
                   input_shapes=input_shapes or {}, output_names=output_names,
                   output_shapes=output_shapes or {}, input_dtypes=input_dtypes or {},
                   output_dtypes=output_dtypes or {}, output_quantization=quant or {})

# ====================================================================
# Lazy model runner: one artifact, validated binding, raw execution.
# ====================================================================

class RunnerError(RuntimeError):
    """Raised when the runtime cannot be loaded or called."""


def _selection_from_config(config: Any) -> ModelSelection:
    """Build a selection from the legacy detector configuration object."""
    profile = getattr(config, "platform", None)
    target = getattr(profile, "key", None) if profile is not None else None
    contract = getattr(config, "contract", None)
    if contract is None:
        contract = default_dfl_contract(
            classes=int(getattr(config, "classes_num", 80)),
            reg_bins=int(getattr(config, "reg", 16)),
            strides=tuple(getattr(config, "strides", (8, 16, 32))),
        )
    return ModelSelection(
        model_path=str(config.model_path),
        target=target,
        platform=profile,
        task="detect",
        contract=contract,
        input_shape=getattr(config, "input_shape", None),
    )


class ModelRunner:
    """Load and call one compiled model with its validated binding."""

    def __init__(self,
                 model: Any,
                 binding: ModelBinding,
                 metadata: Optional[RuntimeMetadata] = None) -> None:
        self.model = model
        self.binding = binding
        self.metadata = metadata or binding.metadata
        self.model_name = self.metadata.model_name
        self.input_adapter = binding.input_adapter
        self.output_adapter = binding.output_adapter
        self.input_names = tuple(self.input_adapter.input_names)
        self.output_names = tuple(self.metadata.output_names)
        self.input_shapes = dict(self.metadata.input_shapes)
        self.output_shapes = dict(self.metadata.output_shapes)
        self.input_dtypes = dict(self.metadata.input_dtypes)
        self.output_dtypes = dict(self.metadata.output_dtypes)
        self.input_height = self.input_adapter.input_height
        self.input_width = self.input_adapter.input_width
        self.input_size = (self.input_height, self.input_width)

    @property
    def contract(self) -> Any:
        return self.binding.contract

    @classmethod
    def from_selection(cls,
                       selection: ModelSelection,
                       runtime_loader: Optional[Callable[[], Any]] = None) -> "ModelRunner":
        """Load a selected artifact, inspect it, and bind its actual metadata."""
        if not isinstance(selection, ModelSelection):
            if not all(hasattr(selection, name)
                       for name in ("model_path", "task", "contract")):
                raise RunnerError("ModelRunner.from_selection expects ModelSelection.")
            selection = ModelSelection(
                model_path=selection.model_path,
                target=getattr(selection, "target", None),
                task=selection.task,
                contract=selection.contract,
                input_shape=getattr(selection, "input_shape", None),
                family=getattr(selection, "family", None),
                artifact_id=getattr(selection, "artifact_id", None),
                platform=getattr(selection, "platform", None),
            )
        if runtime_loader is None:
            # Direct library use is an execution path too.  The canonical CLI
            # performs the same check before model selection; the shared SDK
            # session repeats the exact-target identity gate here so a caller
            # cannot load a board model on an unknown or mismatched host by
            # bypassing that entrypoint, then imports the SDK and constructs
            # the model.
            from utils.py_utils.runtime import RuntimeSession

            requested = selection.target
            if requested is None:
                requested = getattr(selection.platform, "key", selection.platform)
            session = RuntimeSession(str(selection.model_path), target=requested)
            try:
                session.load()
            except Exception as exc:
                raise RunnerError(str(exc)) from exc
            model = session.runtime
        else:
            # ``runtime_loader`` stays the documented host-test seam: it
            # injects the SDK module directly and carries no claim about
            # local hardware.
            try:
                runtime_module = runtime_loader()
                model = runtime_module.HB_HBMRuntime(selection.model_path)
            except Exception as exc:
                if isinstance(exc, RunnerError):
                    raise
                raise RunnerError(f"Unable to load model {selection.model_path!r}: {exc}") from exc
        try:
            metadata = RuntimeMetadata.from_runtime(model)
        except Exception as exc:
            raise RunnerError(f"Runtime metadata is not readable: {exc}") from exc
        try:
            binding = bind_model(selection, metadata)
        except BindingError as exc:
            raise RunnerError(str(exc)) from exc
        return cls(model, binding, metadata)

    @classmethod
    def from_config(cls,
                    config: Any,
                    runtime_loader: Optional[Callable[[], Any]] = None) -> "ModelRunner":
        return cls.from_selection(_selection_from_config(config), runtime_loader)

    def prepare_input(self, y_plane: Any, uv_plane: Any):
        """Pack preprocessed NV12 planes through the bound adapter."""
        return self.input_adapter.build(y_plane, uv_plane)

    def __call__(self, prepared_input: Mapping[str, Any]):
        """Execute once and return role-keyed raw arrays without changing their values."""
        try:
            raw_outputs = self.model.run(prepared_input)
        except Exception as exc:
            raise RunnerError(f"Model execution failed: {exc}") from exc
        try:
            return self.binding.read_raw_outputs(raw_outputs)
        except BindingError as exc:
            raise RunnerError(str(exc)) from exc

    def run(self, prepared_input: Mapping[str, Any]):
        """Compatibility alias for the callable runner interface."""
        return self(prepared_input)

    def forward(self, prepared_input: Mapping[str, Any]):
        """Compatibility alias used by older task wrappers."""
        return self(prepared_input)

    def set_scheduling_params(self, **kwargs: Any) -> None:
        """Forward only explicitly requested scheduling parameters."""
        values = {}
        if "priority" in kwargs and kwargs["priority"] is not None:
            values["priority"] = {self.model_name: kwargs["priority"]}
        if "bpu_cores" in kwargs and kwargs["bpu_cores"] is not None:
            values["bpu_cores"] = {self.model_name: kwargs["bpu_cores"]}
        if not values:
            return
        method = getattr(self.model, "set_scheduling_params", None)
        if method is None:
            raise RunnerError(
                "The selected runtime does not support explicit scheduling parameters.")
        method(**values)


def build_runner(selection: Any,
                 runtime_loader: Optional[Callable[[], Any]] = None) -> ModelRunner:
    """Factory accepting either a ``ModelSelection`` or legacy config."""
    if isinstance(selection, ModelSelection) or all(
            hasattr(selection, name) for name in ("model_path", "task", "contract")):
        return ModelRunner.from_selection(selection, runtime_loader)
    return ModelRunner.from_config(selection, runtime_loader)

# ====================================================================
# Metadata-derived NV12 input adapter (packed X5 / split S protocols).
# ====================================================================

_LAYOUT_NCHW = "NCHW"


_LAYOUT_NHWC = "NHWC"


_KNOWN_LAYOUTS = {
    _LAYOUT_NCHW: 3,
    _LAYOUT_NHWC: 1,
}


class UnsupportedInputError(ValueError):
    """Raised when model input metadata cannot be described exactly."""


@dataclass(frozen=True)
class InputGeometry:
    """Describe the geometry of one model input tensor.

    Attributes:
        height: Input height in pixels.
        width: Input width in pixels.
        layout: `"NCHW"`, `"NHWC"`, or `"flat"` when the tensor is reported as
            a two-dimensional buffer.
        shape: The tensor shape as reported by the runtime.
    """

    height: int
    width: int
    layout: str
    shape: Tuple[int, ...]

    @property
    def pixels(self) -> int:
        """Return the number of luma samples in one frame."""
        return self.height * self.width


def _int_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    """Normalise a runtime shape into a tuple of Python integers."""
    return tuple(int(dim) for dim in shape)


def _infer_plane_geometry(shape: Tuple[int, ...],
                          default_hw: Optional[Tuple[int, int]],
                          label: str) -> InputGeometry:
    """Derive the pixel geometry of a single-plane NV12 input.

    Args:
        shape: Tensor shape as reported by the runtime.
        default_hw: `(height, width)` assumed when the shape is a flat buffer
            whose element count matches it.
        label: Input name, used in error messages.

    Returns:
        The inferred `InputGeometry`.

    Raises:
        UnsupportedInputError: If the geometry cannot be derived.
    """
    if len(shape) == 4:
        # NCHW: (N, C, H, W) with a single channel.
        if shape[1] == 1:
            return InputGeometry(shape[2], shape[3], _LAYOUT_NCHW, shape)
        # NHWC: (N, H, W, C) with a single channel.
        if shape[3] == 1:
            return InputGeometry(shape[1], shape[2], _LAYOUT_NHWC, shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports shape {shape}, which is not a single-plane "
            f"NV12 tensor. Expected (1, 1, H, W) or (1, H, W, 1).")
    if len(shape) == 2:
        count = element_count(shape)
        if default_hw is not None and default_hw[0] * default_hw[1] == count:
            return InputGeometry(default_hw[0], default_hw[1], "flat", shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports flat shape {shape} ({count} elements), so "
            f"its height and width cannot be derived. Pass --input-shape HxW "
            f"matching the compiled model.")
    if len(shape) == 3 and shape[2] == 1:
        return InputGeometry(shape[0], shape[1], _LAYOUT_NHWC, shape)
    raise UnsupportedInputError(
        f"Input {label!r} reports shape {shape}, which is not a supported "
        f"single-plane NV12 tensor layout.")


def _infer_packed_geometry(shape: Tuple[int, ...],
                           default_hw: Optional[Tuple[int, int]],
                           label: str) -> InputGeometry:
    """Derive the pixel geometry of a packed NV12 input.

    A packed NV12 tensor carries one luma plane plus one interleaved chroma
    plane, so its element count is `H * W * 3 / 2`. Some runtimes report the
    same buffer as an `H x W x 3` view instead.

    Args:
        shape: Tensor shape as reported by the runtime.
        default_hw: `(height, width)` assumed when the shape carries no usable
            spatial dimensions.
        label: Input name, used in error messages.

    Returns:
        The inferred `InputGeometry`.

    Raises:
        UnsupportedInputError: If the geometry cannot be derived.
    """
    if len(shape) == 4:
        if shape[1] == 3:
            return InputGeometry(shape[2], shape[3], _LAYOUT_NCHW, shape)
        if shape[3] in (3, 4):
            return InputGeometry(shape[1], shape[2], _LAYOUT_NHWC, shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports shape {shape}, which is not a packed NV12 "
            f"tensor. Expected (1, 3, H, W) or (1, H, W, 3).")
    count = element_count(shape)
    if default_hw is not None:
        expected = default_hw[0] * default_hw[1] * 3 // 2
        if expected == count:
            return InputGeometry(default_hw[0], default_hw[1], "flat", shape)
    raise UnsupportedInputError(
        f"Input {label!r} reports shape {shape} ({count} elements), which does "
        f"not describe a square packed NV12 frame. Pass --input-shape HxW.")


def _validate_square(geometry: InputGeometry, label: str) -> None:
    """Reject non-square inputs, which the shared decoder cannot handle.

    The DFL decoder builds an anchor grid from a single spatial size, so a
    rectangular input would silently decode against the wrong grid. It is
    rejected rather than approximated.

    Args:
        geometry: Geometry to validate.
        label: Input name, used in error messages.

    Raises:
        UnsupportedInputError: If the input is not square.
    """
    if min(geometry.height, geometry.width) <= 0 or geometry.height % 2 or geometry.width % 2:
        raise UnsupportedInputError("NV12 dimensions must be positive and even.")
    if geometry.height != geometry.width:
        raise UnsupportedInputError(
            f"Input {label!r} is {geometry.height}x{geometry.width}. This sample "
            f"supports square inputs only, because the DFL anchor grid is built "
            f"from a single spatial size.")


@dataclass
class Nv12InputAdapter:
    """Bind NV12 planes to a model input in the platform's protocol.

    Attributes:
        profile: Platform whose protocol is used.
        model_name: Model name used as the outer key of the input dictionary.
        input_names: Input tensor names, in runtime order.
        geometry: Geometry inferred from the first input tensor.
        consumed_elements: Total element count the adapter binds.
    """

    profile: PlatformProfile
    model_name: str
    input_names: List[str]
    geometry: InputGeometry
    consumed_elements: int

    @property
    def input_height(self) -> int:
        """Return the model input height in pixels."""
        return self.geometry.height

    @property
    def input_width(self) -> int:
        """Return the model input width in pixels."""
        return self.geometry.width

    @property
    def is_packed(self) -> bool:
        """Return True when NV12 is bound as one packed tensor."""
        return self.profile.is_packed_input

    def expected_anchor_sizes(self, strides: Sequence[int]) -> List[int]:
        """Return the feature-map grid size of each detection scale.

        Args:
            strides: Downsampling stride of each detection scale.

        Returns:
            The grid size of each scale, derived from the input height.
        """
        if len(strides) != 3 or any(int(s) <= 0 or self.geometry.height % int(s) for s in strides):
            raise UnsupportedInputError("Three positive strides must divide the model input size.")
        return [self.geometry.height // int(stride) for stride in strides]

    def describe(self) -> str:
        """Return a one-line description of the bound input protocol.

        Returns:
            A human-readable summary used in logs and in `--list-models`.
        """
        return (f"{self.profile.key}: {self.profile.input_protocol} NV12, "
                f"{self.geometry.height}x{self.geometry.width}, "
                f"layout {self.geometry.layout}")

    @classmethod
    def from_metadata(cls,
                      profile: PlatformProfile,
                      model_name: str,
                      input_names: Sequence[str],
                      input_shapes: Dict[str, Sequence[int]],
                      input_shape_override: Optional[Tuple[int, int]] = None
                      ) -> "Nv12InputAdapter":
        """Build an adapter from the input metadata a runtime reports.

        Args:
            profile: Platform whose protocol is used.
            model_name: Model name used as the outer key of the input dict.
            input_names: Input tensor names, in the order the runtime reports.
            input_shapes: Tensor shapes keyed by input name.
            input_shape_override: Explicit `(height, width)` used when the
                reported shapes carry no usable spatial metadata.

        Returns:
            A validated `Nv12InputAdapter`.

        Raises:
            UnsupportedInputError: If the input count, tensor layout or
                element count does not match the platform protocol.
        """
        names = [str(name) for name in input_names]
        expected_count = 1 if profile.is_packed_input else 2
        if len(names) != expected_count:
            raise UnsupportedInputError(
                f"{profile.key} models take {expected_count} NV12 input "
                f"tensor(s), but {model_name!r} reports {len(names)}: "
                f"{', '.join(names) or '(none)'}. The model was compiled for a "
                f"different input protocol.")
        for name in names:
            if name not in input_shapes:
                raise UnsupportedInputError(
                    f"Runtime reports input {name!r} for {model_name!r} without "
                    f"a shape.")

        default_hw = input_shape_override
        primary = _int_tuple(input_shapes[names[0]])
        if any(int(d) <= 0 for name in names for d in input_shapes[name]):
            raise UnsupportedInputError("Input shapes must be positive and static.")
        if len(primary) == 4 and primary[0] != 1:
            raise UnsupportedInputError("Only batch 1 NV12 inputs are supported.")

        if profile.is_packed_input:
            geometry = _infer_packed_geometry(primary, default_hw, names[0])
            _validate_square(geometry, names[0])
            if input_shape_override and tuple(input_shape_override) != (geometry.height, geometry.width):
                raise UnsupportedInputError("--input-shape conflicts with model metadata.")
            expected_payload = geometry.pixels * 3 // 2
            declared = element_count(primary)
            if declared not in (expected_payload, geometry.pixels * 3):
                raise UnsupportedInputError(
                    f"Input {names[0]!r} declares {declared} elements, but a "
                    f"{geometry.height}x{geometry.width} packed NV12 frame needs "
                    f"{expected_payload}. The model input resolution does not "
                    f"match the shape this sample derived.")
            return cls(profile, model_name, names, geometry, expected_payload)

        geometry = _infer_plane_geometry(primary, default_hw, names[0])
        _validate_square(geometry, names[0])
        if input_shape_override and tuple(input_shape_override) != (geometry.height, geometry.width):
            raise UnsupportedInputError("--input-shape conflicts with model metadata.")
        if primary != (1, geometry.height, geometry.width, 1):
            raise UnsupportedInputError("Split NV12 requires NHWC Y (1,H,W,1).")
        if tuple(input_shapes[names[1]]) != (1, geometry.height//2, geometry.width//2, 2):
            raise UnsupportedInputError("Split NV12 requires UV (1,H/2,W/2,2).")
        luma_shape = _int_tuple(input_shapes[names[1]])
        luma_elements = element_count(primary)
        chroma_elements = element_count(luma_shape)
        if luma_elements != geometry.pixels:
            raise UnsupportedInputError(
                f"Input {names[0]!r} declares {luma_elements} elements, but a "
                f"{geometry.height}x{geometry.width} luma plane needs "
                f"{geometry.pixels}.")
        if chroma_elements * 2 != luma_elements:
            raise UnsupportedInputError(
                f"Input {names[1]!r} declares {chroma_elements} elements, but "
                f"NV12 chroma must hold exactly half the luma samples "
                f"({luma_elements // 2}).")
        return cls(profile, model_name, names, geometry, luma_elements + chroma_elements)

    def build(self, y_plane: np.ndarray, uv_plane: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Bind NV12 planes into the runtime input dictionary.

        Args:
            y_plane: Luma plane with `H * W` elements.
            uv_plane: Interleaved chroma plane with `H * W / 2` elements.

        Returns:
            A nested input dictionary in the form `{model_name: {name: tensor}}`.

        Raises:
            UnsupportedInputError: If the supplied planes do not match the
                geometry the adapter validated at construction time.
        """
        pixels = self.geometry.pixels
        if y_plane.size != pixels:
            raise UnsupportedInputError(
                f"Luma plane holds {y_plane.size} elements, expected {pixels} "
                f"for a {self.geometry.height}x{self.geometry.width} input.")
        if uv_plane.size * 2 != pixels:
            raise UnsupportedInputError(
                f"Chroma plane holds {uv_plane.size} elements, expected "
                f"{pixels // 2}.")

        if self.is_packed:
            packed = np.concatenate(
                [y_plane.reshape(-1), uv_plane.reshape(-1)]).astype(np.uint8)
            return {self.model_name: {self.input_names[0]: packed}}

        return {
            self.model_name: {
                self.input_names[0]: np.ascontiguousarray(y_plane, dtype=np.uint8).reshape(1, self.input_height, self.input_width, 1),
                self.input_names[1]: np.ascontiguousarray(uv_plane, dtype=np.uint8).reshape(1, self.input_height//2, self.input_width//2, 2),
            }
        }

# ====================================================================
# Lazy board-runtime bridge.
# ====================================================================

class BoardRuntimeUnavailableError(RuntimeError):
    """Raised when the board runtime is required but not installed."""


def load_hbm_runtime():
    """Import and return the board runtime module.

    Returns:
        The imported `hbm_runtime` module.

    Raises:
        BoardRuntimeUnavailableError: If the module is not installed. The
            message names the missing package and never triggers an install.
    """
    try:
        import hbm_runtime  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise BoardRuntimeUnavailableError(
            "hbm_runtime is not installed. It is provided by the RDK system "
            "image and is required only for on-board inference. Inspecting "
            "model names, printing --help and download dry runs work without "
            "it.") from exc
    return hbm_runtime

__all__ = [
    "BoardRuntimeUnavailableError",
    "BindingError",
    "ClassificationContract",
    "DFLDetectionContract",
    "DFLPoseContract",
    "DFLSegmentationContract",
    "InputBinding",
    "InputGeometry",
    "LTRBDetectionContract",
    "LTRBOBBContract",
    "LTRBPoseContract",
    "LTRBSegmentationContract",
    "ModelBinding",
    "ModelRunner",
    "ModelSelection",
    "Nv12InputAdapter",
    "OutputBinding",
    "RawOutputs",
    "RuntimeMetadata",
    "RunnerError",
    "TensorContractError",
    "UnsupportedInputError",
    "bind_model",
    "bind_nv12_inputs",
    "build_runner",
    "default_dfl_contract",
    "element_count",
    "load_hbm_runtime",
    "normalize_dtype",
    "normalize_shape",
    "pack_nv12_planes",
    "pack_nv12_single",
    "read_output",
    "require_floating_output",
    "resolve_selection",
]
