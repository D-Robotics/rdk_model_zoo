# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2's exact publication and runtime tensor contract.

This module owns the facts that differ from the shared classification binding:
one model with two feature outputs, RGB float input, and per-output numeric
transforms.  It never infers a contract from an arbitrary filename.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.quantization import validate_output_transform
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError, canonicalise_dtype

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("s100", "s100p", "s600")
MARCHES = {"s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}
OUTPUTS = ("cls_feat", "patch_feat")
OUTPUT_SHAPES = {
    "cls_feat": (1, 384),
    "patch_feat": (1, 256, 384),
}
NUMERIC_DTYPES = {"float32", "int8", "uint8", "int16", "int32"}


@dataclass(frozen=True)
class ModelSelection:
    """One exact manifest asset selected for one concrete target."""

    asset: Asset
    target: str
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Metadata-validated single-model, dual-output runtime contract."""

    selection: ModelSelection
    model_name: str
    input_name: str
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    output_dtypes: Mapping[str, str]
    output_quants: Mapping[str, Any]
    output_transforms: Mapping[str, str]


def _expected_filename(target: str) -> str:
    march = MARCHES[target]
    suffix = {"nash-e": "nashe", "nash-m": "nashm", "nash-p": "nashp"}[march]
    return f"{march}/dinov2_vits14_224_int16_{suffix}.hbm"


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """Return the three exact HBM rows from the S publication manifest."""

    if target not in (None, "auto", *SUPPORTED_TARGETS):
        if target in ("x5",):
            return ()
        raise ValueError(f"Unknown target: {target}")
    assets = list_assets("s", "dinov2")
    expected = {_expected_filename(key) for key in SUPPORTED_TARGETS}
    actual = {asset.filename for asset in assets}
    if actual != expected or any(asset.format != "hbm" for asset in assets):
        raise ValueError("DINOv2 publication changed; review its finite contracts first.")
    if target in (None, "auto"):
        return tuple(asset for asset in assets if asset.filename in expected)
    filename = _expected_filename(target)
    return tuple(asset for asset in assets if asset.filename == filename)


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: str | None = None,
    model_path: str | Path | None = None,
    soc_name: str | None = None,
    board_type: str | None = None,
) -> ModelSelection:
    """Resolve an exact published asset without filename or target fallback."""

    resolved = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if resolved not in SUPPORTED_TARGETS:
        raise ValueError(f"No published DINOv2 support for {resolved}.")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model-path requires the exact manifest asset-id.")

    assets = list_available_assets(resolved)
    if asset_id is None:
        matches = assets
    else:
        matches = tuple(asset for asset in assets if asset.reference == asset_id)
    if len(matches) != 1:
        available = ", ".join(asset.reference for asset in assets)
        raise ValueError(f"Unknown DINOv2 asset-id {asset_id!r}; available: {available}")
    asset = matches[0]
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / asset.filename
    return ModelSelection(asset, resolved, path, model_path is not None)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate actual runtime metadata against the two-output DINO contract."""

    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path,
    )
    if expected.asset != selection.asset:
        raise ValueError("Selection publication facts do not match the manifest.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if facts.input_names != ("input",) or facts.input_shapes.get("input") != (1, 3, 224, 224):
        raise MetadataMismatchError("DINOv2 input must be input F32 (1,3,224,224).")
    if facts.input_dtypes.get("input") != "float32":
        raise MetadataMismatchError("DINOv2 input dtype must be float32.")
    if facts.output_names != OUTPUTS:
        raise MetadataMismatchError(f"DINOv2 outputs must be {OUTPUTS}; got {facts.output_names}.")

    dtypes: dict[str, str] = {}
    transforms: dict[str, str] = {}
    for name in OUTPUTS:
        shape = facts.output_shapes.get(name)
        if shape != OUTPUT_SHAPES[name]:
            raise MetadataMismatchError(f"DINOv2 {name} shape must be {OUTPUT_SHAPES[name]}; got {shape}.")
        dtype = canonicalise_dtype(facts.output_dtypes.get(name))
        if dtype not in NUMERIC_DTYPES:
            raise MetadataMismatchError(f"DINOv2 {name} dtype is unsupported: {dtype!r}.")
        dtypes[name] = dtype
        if dtype == "float32":
            transforms[name] = validate_output_transform("raw_f32")
        else:
            if name not in facts.output_quants:
                raise MetadataMismatchError(f"DINOv2 {name} integer output has no quantization descriptor.")
            transforms[name] = validate_output_transform("dequant")
    return ModelBinding(
        selection=selection,
        model_name=facts.model_name,
        input_name="input",
        input_shape=(1, 3, 224, 224),
        output_names=OUTPUTS,
        output_shapes=dict(facts.output_shapes),
        output_dtypes=dtypes,
        output_quants=dict(facts.output_quants),
        output_transforms=transforms,
    )


__all__ = [
    "MARCHES", "ModelBinding", "ModelSelection", "OUTPUTS", "OUTPUT_SHAPES",
    "SAMPLE_DIR", "SUPPORTED_TARGETS", "bind_model", "list_available_assets",
    "resolve_selection",
]
