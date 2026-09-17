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

"""Bind a selected YOLO artifact to its observed runtime tensor contract.

The binding is deliberately finite and executable-code based.  It contains no
asset catalogue or download logic.  A caller may provide the output names that
were reviewed for a particular artifact; otherwise a DFL binding is discovered
only when runtime shapes make every role unambiguous.  A file name or an opaque
output position is never treated as proof of a detection protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.tensor_io import (
    InputBinding,
    OutputBinding,
    TensorContractError,
    as_quantization,
    bind_nv12_inputs,
    normalize_dtype,
    normalize_shape,
)


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
    quantization: Mapping[str, Any]

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
                 quantization: Optional[Mapping[str, Any]] = None,
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
        object.__setattr__(self, "quantization", {
            str(key): value for key, value in (quantization or {}).items()
        })

    @property
    def required_roles(self) -> Tuple[str, ...]:
        return tuple(role for stride in self.strides
                     for role in (_role_key("cls", stride), _role_key("box", stride)))


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
            from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
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
                           name: str) -> Tuple[str, Tuple[int, ...]]:
    """Validate a physical shape and return its layout."""
    gh, gw = grid
    if shape == (1, gh, gw, channels):
        return "NHWC", shape
    raise BindingError(
        f"Output {name!r} shape {shape} is incompatible with grid {gh}x{gw} and "
        f"{channels} channels; this contract requires NHWC.")


def _runtime_quantization(metadata: Any, name: str) -> Any:
    values = getattr(metadata, "output_quantization", {}) or {}
    return values.get(name) if isinstance(values, Mapping) else None


def _bind_output_roles(selection: ModelSelection,
                       contract: Any,
                       metadata: Any,
                       adapter: InputBinding) -> OutputBinding:
    protocol = str(getattr(contract, "protocol", "DFL"))
    box_channels = int(getattr(
        contract, "box_channels", 4 * int(getattr(contract, "reg_bins", 16))))
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
        for stride in contract.strides:
            grid = _expected_grid(adapter, stride)
            for kind, channels in (("cls", contract.classes),
                                   ("box", box_channels)):
                role = _role_key(kind, stride)
                candidates = []
                for name in names:
                    try:
                        _role_shape_descriptor(shapes[name], grid, channels, name)
                    except BindingError:
                        continue
                    candidates.append(name)
                if len(candidates) != 1:
                    if len(candidates) == 0:
                        raise BindingError(
                            f"No uniquely identifiable tensor for {protocol} role {role!r}; "
                            f"expected grid {grid[0]}x{grid[1]} with {channels} channels.")
                    raise BindingError(
                        f"Multiple tensors match {protocol} role {role!r}: {', '.join(candidates)}.")
                role_to_name[role] = candidates[0]

    expected_shapes: Dict[str, Tuple[int, ...]] = {}
    layouts: Dict[str, str] = {}
    channels_map: Dict[str, int] = {}
    quant_map: Dict[str, Any] = {}
    for stride in contract.strides:
        grid = _expected_grid(adapter, stride)
        for kind, channels in (("cls", contract.classes),
                               ("box", box_channels)):
            role = _role_key(kind, stride)
            name = role_to_name[role]
            channels_map[role] = channels
            layout, physical_shape = _role_shape_descriptor(
                shapes[name], grid, channels, name)
            declared_layout = contract.output_layouts.get(role)
            if declared_layout is not None and str(declared_layout).upper() != layout:
                raise BindingError(
                    f"Output role {role!r} declares {declared_layout} but runtime shape "
                    f"{shapes.get(name)} is {layout}.")
            layouts[role] = layout
            expected_shapes[role] = physical_shape
            dtype = dtypes.get(name)
            quantization = getattr(contract, "quantization", {}) or {}
            quant_value = quantization.get(role,
                                           quantization.get(name))
            if quant_value is None:
                quant_value = _runtime_quantization(metadata, name)
            if protocol == "LTRB" and quant_value is not None:
                raise BindingError(
                    f"LTRB output {name!r} must be the observed floating tensor; "
                    "quantization is not part of this contract.")
            try:
                quant = as_quantization(quant_value)
            except TensorContractError as exc:
                raise BindingError(str(exc)) from exc
            quant_map[role] = quant
            if dtype is None:
                raise BindingError(f"Runtime output {name!r} has no dtype metadata.")
            if np.issubdtype(dtype, np.integer) and quant is None:
                raise BindingError(
                    f"Output {name!r} is integer {dtype}; no explicit quantization "
                    "parameters were provided.")
            elif not np.issubdtype(dtype, np.floating) and quant is None:
                raise BindingError(
                    f"Output {name!r} dtype {dtype} is not a floating semantic tensor.")

    return OutputBinding(
        model_name=str(metadata.model_name),
        role_to_name=role_to_name,
        shapes=shapes,
        dtypes=dtypes,
        expected_shapes=expected_shapes,
        channels=channels_map,
        layouts=layouts,
        quantization=quant_map,
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
        )
    except TensorContractError as exc:
        raise BindingError(str(exc)) from exc
    output_adapter = _bind_output_roles(selection, contract, metadata,
                                        input_adapter)
    return ModelBinding(selection=selection, contract=contract, metadata=metadata,
                        input_adapter=input_adapter, output_adapter=output_adapter)


__all__ = [
    "BindingError",
    "DFLDetectionContract",
    "LTRBDetectionContract",
    "ModelSelection",
    "ModelBinding",
    "RuntimeMetadata",
    "default_dfl_contract",
    "resolve_selection",
    "bind_model",
]


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
        quant = per_tensor("output_quantization", {})
        if not quant:
            quant = per_tensor("output_quant_infos", {})
        if not quant:
            quant = per_tensor("quant_infos", {})
        return cls(model_name=str(model_name), input_names=input_names,
                   input_shapes=input_shapes or {}, output_names=output_names,
                   output_shapes=output_shapes or {}, input_dtypes=input_dtypes or {},
                   output_dtypes=output_dtypes or {}, output_quantization=quant or {})
