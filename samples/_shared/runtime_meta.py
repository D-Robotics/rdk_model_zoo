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

"""Runtime metadata reading shared by the migrated samples (Phase 1.5 H3).

The class reads only public attributes exposed by ``hbm_runtime`` objects and
never imports a board SDK.  It supports runtimes that host several models in
one artifact (an explicit model selection is required in that case) and any
number of output tensors; single-output contracts stay a binding-level rule.

``output_quants`` carries the raw per-output quantization descriptors
(``{output_name: quant_info}``) exactly as the runtime exposes them through
``output_quants``/``model.output_quants[model_name]`` on both the X5 and S
toolchains.  The values are deliberately not float-coerced: their structure
(scale, zero_point, axis, quant_type) belongs to the dequantization chain in
:mod:`samples._shared.quantization`, and a descriptor that rides along an F32
output must stay visible in the binding snapshot instead of being silently
dropped (the raw_f32 path gates on dtype and never applies it).

Those raw descriptors are SDK objects that refuse to be copied, so evidence
writers must not run them through :func:`dataclasses.asdict` (it
``copy.deepcopy``\ s every leaf and the board ``QuantParams`` type raises
``TypeError`` when pickled — board evidence 2026-09-24).  Use
:func:`metadata_evidence` to project a ``RuntimeMetadata`` into JSON-ready
evidence values without copying or mutating anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping


class MetadataMismatchError(ValueError):
    """Runtime metadata cannot satisfy a known sample contract."""


def canonicalise_dtype(dtype: Any) -> str | None:
    """Normalise one runtime dtype token; unknown tokens are kept verbatim."""

    if dtype is None:
        return None
    raw = str(getattr(dtype, "name", dtype)).lower()
    if raw in {"f32", "float", "float32", "hbdnndatatype.f32"} or raw.endswith(".f32"):
        return "float32"
    # The board SDK exposes signed outputs as hbDNNDataType.S8/S16/S32 enum
    # members (S100 evidence: <hbDNNDataType.S32: 8>), so their enum names are
    # accepted next to the i*/int* spellings already handled below.
    if raw in {"i8", "s8", "int8", "hbdnndatatype.int8"} or raw.endswith((".int8", ".s8")):
        return "int8"
    if raw in {"u8", "uint8", "hbdnndatatype.u8"} or raw.endswith(".u8"):
        return "uint8"
    if raw in {"i16", "s16", "int16", "hbdnndatatype.int16"} or raw.endswith((".int16", ".s16")):
        return "int16"
    if raw in {"i32", "s32", "int32", "hbdnndatatype.int32"} or raw.endswith((".int32", ".s32")):
        return "int32"
    if raw in {"f16", "float16", "hbdnndatatype.f16"} or raw.endswith(".f16"):
        return "float16"
    if raw in {"nv12", "hbdnndatatype.nv12"} or raw.endswith(".nv12"):
        return "nv12"
    return raw


@dataclass(frozen=True)
class RuntimeMetadata:
    """Public tensor facts observed for one selected runtime model.

    ``model_names`` lists every model hosted by the runtime artifact while
    ``model_name`` is the selected one.  ``from_runtime`` no longer assumes
    ``model_names[0]``: a multi-model runtime requires an explicit selection.
    """

    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_dtypes: Mapping[str, str]
    model_names: tuple[str, ...] = ()
    input_strides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    output_strides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    output_quants: Mapping[str, Any] = field(default_factory=dict)
    output_semantics: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeMetadata":
        """Create metadata from a flat or one-model nested mapping."""

        names_value = values.get("model_names") or ()
        model_names = tuple(str(name) for name in names_value)
        model_name = str(values.get("model_name") or values.get("name") or "model")
        if model_names and model_name not in model_names:
            raise MetadataMismatchError(
                f"Selected model {model_name!r} is not among the runtime's "
                f"models {model_names!r}."
            )
        if not model_names:
            model_names = (model_name,)

        def model_field(name: str, default: Any) -> Any:
            value = values.get(name, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        semantics = model_field("output_semantics", None)
        output_names = tuple(str(value) for value in model_field("output_names", ()))
        if isinstance(semantics, Mapping):
            # Multi-output semantics stay per output name; callers that need
            # one value resolve it against their bound output.
            semantics = {str(key): value for key, value in semantics.items()}
        elif semantics is not None:
            semantics = str(semantics).lower()
        return cls(
            model_name=model_name,
            input_names=tuple(str(value) for value in model_field("input_names", ())),
            input_shapes=_normalise_shapes(model_field("input_shapes", {})),
            output_names=output_names,
            output_shapes=_normalise_shapes(model_field("output_shapes", {})),
            input_dtypes=_normalise_dtypes(model_field("input_dtypes", {})),
            output_dtypes=_normalise_dtypes(model_field("output_dtypes", {})),
            model_names=model_names,
            input_strides=_normalise_shapes(model_field("input_strides", {})),
            output_strides=_normalise_shapes(model_field("output_strides", {})),
            output_quants=_normalise_quants(model_field("output_quants", {})),
            output_semantics=semantics,
        )

    @classmethod
    def from_runtime(
        cls, runtime: Any, model_name: str | None = None
    ) -> "RuntimeMetadata":
        """Read public metadata attributes exposed by ``hbm_runtime``.

        ``model_name`` selects one model of a multi-model artifact.  Without a
        selection exactly one model must be exposed; several models are never
        silently reduced to ``model_names[0]``.
        """

        names = getattr(runtime, "model_names", None)
        if not names:
            raise MetadataMismatchError("Runtime did not expose model_names.")
        model_names = tuple(str(name) for name in names)
        if model_name is None:
            if len(model_names) != 1:
                raise MetadataMismatchError(
                    "Runtime hosts several models "
                    f"{model_names!r}; pass model_name to select one."
                )
            selected = model_names[0]
        else:
            selected = str(model_name)
            if selected not in model_names:
                raise MetadataMismatchError(
                    f"Model {selected!r} is not among the runtime's models "
                    f"{model_names!r}."
                )

        def runtime_field(name: str, default: Any) -> Any:
            value = getattr(runtime, name, default)
            if isinstance(value, Mapping) and selected in value:
                return value[selected]
            return value

        return cls.from_mapping(
            {
                "model_names": model_names,
                "model_name": selected,
                "input_names": runtime_field("input_names", ()),
                "input_shapes": runtime_field("input_shapes", {}),
                "output_names": runtime_field("output_names", ()),
                "output_shapes": runtime_field("output_shapes", {}),
                "input_dtypes": runtime_field("input_dtypes", {}),
                "output_dtypes": runtime_field("output_dtypes", {}),
                "input_strides": runtime_field("input_strides", {}),
                "output_strides": runtime_field("output_strides", {}),
                "output_quants": runtime_field("output_quants", {}),
                "output_semantics": runtime_field("output_semantics", None),
            }
        )


def _normalise_shapes(values: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for name, shape in values.items():
        try:
            result[str(name)] = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            result[str(name)] = ()
    return result


def _normalise_dtypes(values: Any) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, str] = {}
    for name, dtype in values.items():
        canonical = canonicalise_dtype(dtype)
        result[str(name)] = canonical if canonical is not None else str(dtype)
    return result


def _normalise_quants(values: Any) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}
    return {str(name): info for name, info in values.items()}


def _evidence_value(value: Any) -> Any:
    """Project one metadata value onto JSON-serialisable primitives.

    SDK objects are read attribute by attribute and never copied, and unknown
    objects raise instead of being stringified, so evidence can never
    silently degrade into ``str(...)`` text.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _evidence_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            entry.name: _evidence_value(getattr(value, entry.name))
            for entry in fields(value)
        }
    if hasattr(value, "quant_type"):
        # Raw runtime quantization descriptor (hbm_runtime QuantParams and
        # alike): keep every reported fact verbatim instead of copying the
        # object.  quant_type is reduced to its enum-like name exactly like
        # the dequantization chain reads it; scale/zero_point may be scalars
        # or per-channel arrays; further public attributes ride along so an
        # SDK extension cannot be dropped from evidence silently.
        projected: dict[str, Any] = {
            "quant_type": str(getattr(value.quant_type, "name", value.quant_type)),
            "scale": _evidence_value(getattr(value, "scale", None)),
            "zero_point": _evidence_value(getattr(value, "zero_point", None)),
            "axis": _evidence_value(getattr(value, "axis", None)),
        }
        for name in sorted(getattr(value, "__dict__", {})):
            if not name.startswith("_") and name not in projected:
                projected[name] = _evidence_value(getattr(value, name))
        return projected
    raise TypeError(f"Unsupported evidence value {type(value).__name__}.")


def metadata_evidence(metadata: Any) -> dict[str, Any]:
    """Project runtime metadata into JSON-serialisable evidence values.

    ``asdict(RuntimeMetadata.from_runtime(runtime))`` fails on real boards:
    :func:`dataclasses.asdict` deep-copies leaf values and the SDK's
    ``QuantParams`` forbids pickling (X5 board evidence 2026-09-24).  This
    projection keeps every tensor fact — model names, input/output names,
    shapes, dtypes, strides and the complete per-output quant descriptors
    (``quant_type``, ``scale``, ``zero_point``, ``axis``, plus any further
    public attributes the SDK reports) — while never copying or mutating the
    metadata object or its SDK descriptors.  A ``RuntimeMetadata`` (or any
    dataclass) and plain mappings are both accepted, so host-test seams can
    pass either form.
    """

    if is_dataclass(metadata) and not isinstance(metadata, type):
        return {
            entry.name: _evidence_value(getattr(metadata, entry.name))
            for entry in fields(metadata)
        }
    if isinstance(metadata, Mapping):
        return _evidence_value(metadata)
    raise TypeError(
        f"Unsupported metadata object {type(metadata).__name__}; expected a "
        "RuntimeMetadata or a mapping."
    )


__all__ = ["MetadataMismatchError", "RuntimeMetadata", "canonicalise_dtype", "metadata_evidence"]
