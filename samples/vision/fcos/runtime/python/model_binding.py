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

"""FCOS asset and tensor contract for the X5 publication.

The binding reads the existing ``docs/release/x5/models.yaml`` rows through
the shared asset reader.  It never infers an artifact identity from an
arbitrary filename, and it binds all fifteen FCOS tensors by observed shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import RuntimeMetadata, canonicalise_dtype

SAMPLE = "fcos"
SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("x5",)
STRIDES = (8, 16, 32, 64, 128)
CLASSES = 80
_VARIANT_FILES = {
    "fcos_efficientnetb0_detect_512x512_bayese_nv12.bin": "efficientnetb0",
    "fcos_efficientnetb2_detect_768x768_bayese_nv12.bin": "efficientnetb2",
    "fcos_efficientnetb3_detect_896x896_bayese_nv12.bin": "efficientnetb3",
}
_VARIANT_SIZES = {"efficientnetb0": 512, "efficientnetb2": 768, "efficientnetb3": 896}
_ALLOWED_OUTPUT_DTYPES = {"float16", "float32", "int8", "uint8", "int16", "int32"}


class BindingError(ValueError):
    """A requested selection or runtime metadata violates the FCOS contract."""


@dataclass(frozen=True)
class AssetRecord:
    """One exact manifest-backed FCOS artifact."""

    target: str
    sample_id: str
    variant: str
    filename: str
    model_format: str
    url: str | None
    sha256: str | None

    @property
    def asset_id(self) -> str:
        """Return the exact qualified manifest identity."""
        return f"x5:{self.sample_id}:{self.filename}"

    @property
    def reference(self) -> str:
        """Alias for callers shared with other model samples."""
        return self.asset_id


@dataclass(frozen=True)
class FCOSContract:
    """Static geometry and output semantics for one published variant.

    FCOS follows the fixed source's ``dequantize_outputs`` behavior: a SCALE
    descriptor is applied to every observed dtype, including F32.  The runtime
    metadata must therefore carry a complete descriptor even for a float
    output; the binding never guesses a raw-float path from dtype alone.
    """

    variant: str
    input_height: int
    input_width: int
    classes_num: int = CLASSES
    strides: tuple[int, ...] = STRIDES
    resize_type: int = 0
    conf_thres: float = 0.5
    iou_thres: float = 0.6
    output_transform: str = "dequant"


@dataclass(frozen=True)
class ModelSelection:
    """Resolved artifact identity and its source-proven FCOS contract."""

    asset_id: str
    target: str
    variant: str
    model_path: Path
    contract: FCOSContract
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Observed runtime metadata bound to one exact FCOS artifact."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    output_dtypes: Mapping[str, str]
    output_quants: Mapping[str, Any]
    cls_output_names: tuple[str, ...]
    box_output_names: tuple[str, ...]
    center_output_names: tuple[str, ...]

    @property
    def contract(self) -> FCOSContract:
        """Return the static contract used for binding."""
        return self.selection.contract

    @property
    def input_width(self) -> int:
        return self.contract.input_width

    @property
    def input_height(self) -> int:
        return self.contract.input_height

    def validate_inputs(self, tensors: Mapping[str, np.ndarray]) -> None:
        """Validate packed NV12 tensors without changing their identity."""
        if tuple(tensors) != self.input_names:
            raise BindingError(f"Input names {tuple(tensors)!r} != {self.input_names!r}.")
        value = tensors[self.input_names[0]]
        if not isinstance(value, np.ndarray):
            raise BindingError("Packed NV12 input must be a NumPy array.")
        expected = self.input_height * self.input_width * 3 // 2
        if value.ndim != 1 or value.size != expected or value.dtype != np.uint8:
            raise BindingError(
                f"Packed NV12 input must be uint8 shape ({expected},), got "
                f"{value.shape} {value.dtype}."
            )
        if not value.flags.c_contiguous:
            raise BindingError("Packed NV12 input must be contiguous.")

    def validate_outputs(self, outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate raw output containers and return the same ndarray objects.

        Matching is by exact name set, never by insertion order: the board
        ``hbm_runtime`` ``run()`` mapping does not preserve
        ``metadata.output_names`` order while the fifteen names themselves
        are identical (X5 evidence 2026-09-24).  Missing and extra names are
        both rejected, and every bound name is checked against the binding's
        shape, dtype, and finiteness.  The mapping and each ndarray stay
        caller-owned and identity-stable; roles are resolved only from the
        binding's own name tuples, never by iterating the caller's dict.
        """
        if not isinstance(outputs, Mapping):
            raise BindingError("Runtime output must be a name→ndarray mapping.")
        observed = set(outputs)
        bound = set(self.output_names)
        missing = sorted(bound - observed)
        extra = sorted(observed - bound)
        if missing or extra:
            raise BindingError(
                f"Output names must match the binding exactly; "
                f"missing={missing}, unexpected={extra}."
            )
        for name in self.output_names:
            value = outputs[name]
            if not isinstance(value, np.ndarray):
                raise BindingError(f"Output {name!r} must be a NumPy array.")
            if tuple(value.shape) != self.output_shapes[name]:
                raise BindingError(
                    f"Output {name!r} shape {value.shape} != {self.output_shapes[name]}."
                )
            if canonicalise_dtype(value.dtype) != self.output_dtypes[name]:
                raise BindingError(
                    f"Output {name!r} dtype {value.dtype} != {self.output_dtypes[name]}."
                )
            if not np.all(np.isfinite(value.astype(np.float32, copy=False))):
                raise BindingError(f"Output {name!r} contains non-finite values.")
        # The raw output mapping and each ndarray remain caller-owned and
        # identity-stable; validation is deliberately observational.
        return outputs  # type: ignore[return-value]


def _records() -> tuple[AssetRecord, ...]:
    result = []
    for asset in list_assets("x5", SAMPLE):
        try:
            variant = _VARIANT_FILES[asset.filename]
        except KeyError as exc:
            raise BindingError(f"Unexpected FCOS manifest asset: {asset.filename!r}.") from exc
        result.append(AssetRecord("x5", SAMPLE, variant, asset.filename, asset.format, asset.url, asset.sha256))
    if {record.variant for record in result} != set(_VARIANT_SIZES):
        raise BindingError("FCOS manifest does not publish the required three variants.")
    return tuple(result)


def list_available_assets(target: str | None = None) -> tuple[AssetRecord, ...]:
    """List exact FCOS asset references for X5; ``auto`` is host-listing only."""
    if target not in (None, "auto", "x5"):
        raise BindingError(f"FCOS is published for x5 only, not {target!r}.")
    return _records()


def resolve_selection(
    target: str = "x5",
    *,
    asset_id: str | None = None,
    variant: str | None = None,
    model_path: str | Path | None = None,
) -> ModelSelection:
    """Resolve one exact manifest row and reject unqualified external paths."""
    if target == "auto":
        raise BindingError("FCOS execution requires explicit --target x5.")
    if target != "x5":
        raise BindingError(f"Unsupported FCOS target {target!r}; only x5 is published.")
    records = _records()
    if asset_id is None:
        requested_variant = "efficientnetb0" if variant is None else variant
        matches = [record for record in records if record.variant == requested_variant]
        if len(matches) != 1:
            raise BindingError("Select one FCOS variant with --variant or --asset-id.")
    else:
        matches = [record for record in records if record.asset_id == asset_id]
        if variant is not None:
            matches = [record for record in matches if record.variant == variant]
        if len(matches) != 1:
            raise BindingError(f"Unknown or mismatched FCOS asset_id {asset_id!r}.")
    record = matches[0]
    if model_path is not None and asset_id is None:
        raise BindingError("--model-path requires the exact --asset-id reference.")
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / record.filename
    size = _VARIANT_SIZES[record.variant]
    return ModelSelection(record.asset_id, target, record.variant, path, FCOSContract(record.variant, size, size), model_path is not None)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Bind all fifteen observed tensors to the selected FCOS contract.

    Output names are not guessed.  Their roles are resolved only by the exact
    source shape family for each stride and channel count; every output keeps
    its runtime quantization descriptor for post-processing.
    """
    records = {record.asset_id: record for record in _records()}
    if selection.asset_id not in records or records[selection.asset_id].variant != selection.variant:
        raise BindingError("Selection does not match a current FCOS manifest asset.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if tuple(facts.model_names) != (facts.model_name,):
        raise BindingError("FCOS artifact must expose exactly one selected runtime model.")
    if len(facts.input_names) != 1:
        raise BindingError("FCOS X5 expects one packed input tensor.")
    input_name = facts.input_names[0]
    if tuple(facts.input_shapes.get(input_name, ())) != (1, 3, selection.contract.input_height, selection.contract.input_width):
        raise BindingError("Runtime input shape does not match the selected FCOS variant.")
    if canonicalise_dtype(facts.input_dtypes.get(input_name)) != "nv12":
        raise BindingError("FCOS input metadata must report NV12.")
    if len(facts.output_names) != 15 or set(facts.output_names) != set(facts.output_shapes):
        raise BindingError("FCOS metadata must expose exactly fifteen named outputs.")
    cls: list[str] = []
    box: list[str] = []
    center: list[str] = []
    used: set[str] = set()
    size = selection.contract.input_height
    for names, channels in ((cls, 80), (box, 4), (center, 1)):
        for stride in selection.contract.strides:
            expected = (1, size // stride, size // stride, channels)
            matches = [name for name in facts.output_names if name not in used and tuple(facts.output_shapes.get(name, ())) == expected]
            if len(matches) != 1:
                raise BindingError(f"Expected exactly one FCOS output with shape {expected}, found {matches}.")
            name = matches[0]
            used.add(name)
            names.append(name)
    if used != set(facts.output_names):
        raise BindingError("FCOS metadata contains an unclassified output tensor.")
    for name in facts.output_names:
        dtype = canonicalise_dtype(facts.output_dtypes.get(name))
        if dtype not in _ALLOWED_OUTPUT_DTYPES:
            raise BindingError(f"Unsupported FCOS output dtype for {name!r}: {dtype!r}.")
        if name not in facts.output_quants or not _is_quant_descriptor(facts.output_quants[name]):
            raise BindingError(f"FCOS output {name!r} is missing its quantization descriptor.")
    return ModelBinding(
        selection=selection,
        metadata=facts,
        model_name=facts.model_name,
        input_names=facts.input_names,
        input_shapes=facts.input_shapes,
        output_names=facts.output_names,
        output_shapes=facts.output_shapes,
        output_dtypes={name: canonicalise_dtype(facts.output_dtypes[name]) or "" for name in facts.output_names},
        output_quants=facts.output_quants,
        cls_output_names=tuple(cls),
        box_output_names=tuple(box),
        center_output_names=tuple(center),
    )


def _is_quant_descriptor(value: Any) -> bool:
    """Accept only runtime-like descriptors that source dequant can inspect."""
    return value is not None and all(hasattr(value, key) for key in ("quant_type", "scale", "zero_point"))


__all__ = ["AssetRecord", "BindingError", "FCOSContract", "ModelBinding", "ModelSelection", "RuntimeMetadata", "bind_model", "list_available_assets", "resolve_selection"]
