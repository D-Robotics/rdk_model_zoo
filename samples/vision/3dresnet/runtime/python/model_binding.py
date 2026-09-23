# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exact publication and tensor binding for the S100 R3D-18 artifact."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.cls_binding import score_vector_shape
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata, canonicalise_dtype

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("s100",)
ASSET_FILENAME = "s100/r3d_18.hbm"
ASSET_ID = "s:3dresnet:s100/r3d_18.hbm"
INPUT_SHAPE = (1, 3, 16, 112, 112)
CLASS_COUNT = 400


@dataclass(frozen=True)
class ModelSelection:
    asset: Asset
    target: str
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    model_name: str
    input_name: str
    input_shape: tuple[int, ...]
    output_name: str
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    output_transform: str = "raw_f32"


def _published_asset() -> Asset:
    assets = tuple(asset for asset in list_assets("s", "3dresnet") if asset.filename == ASSET_FILENAME)
    if len(assets) != 1 or assets[0].format != "hbm":
        raise ValueError("The published 3DResNet S100 HBM asset is missing or changed.")
    return assets[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    if target not in (None, "auto", *SUPPORTED_TARGETS):
        return ()
    asset = _published_asset()
    if target in (None, "auto", "s100"):
        return (asset,)
    return ()


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: str | None = None,
    model_path: str | Path | None = None,
    soc_name: str | None = None,
    board_type: str | None = None,
) -> ModelSelection:
    resolved = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if resolved not in SUPPORTED_TARGETS:
        raise ValueError(f"No published 3DResNet support for {resolved}.")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model-path requires the exact manifest asset-id.")
    asset = _published_asset()
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Unknown 3DResNet asset-id {asset_id!r}; expected {asset.reference!r}.")
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / asset.filename
    return ModelSelection(asset, resolved, path, model_path is not None)


def _score_vector_shape(shape: tuple[int, ...], class_count: int = CLASS_COUNT) -> bool:
    return score_vector_shape(shape, class_count)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    if selection.asset.reference != ASSET_ID or selection.target != "s100":
        raise ValueError("3DResNet binding received an unpublished selection.")
    published = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path,
    )
    if published.asset != selection.asset:
        raise ValueError("3DResNet selection publication facts do not match the manifest.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if len(facts.input_names) != 1 or len(facts.output_names) != 1:
        raise MetadataMismatchError("R3D-18 requires exactly one input and one output tensor.")
    input_name = facts.input_names[0]
    output_name = facts.output_names[0]
    if facts.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(
            f"R3D-18 input {input_name!r} must have shape {INPUT_SHAPE}; "
            f"got {facts.input_shapes.get(input_name)}."
        )
    if canonicalise_dtype(facts.input_dtypes.get(input_name)) != "float32":
        raise MetadataMismatchError("R3D-18 input must be float32.")
    output_shape = facts.output_shapes.get(output_name)
    if output_shape is None or not _score_vector_shape(output_shape):
        raise MetadataMismatchError(
            f"R3D-18 output {output_name!r} must squeeze to 400 scores; got {output_shape}."
        )
    if canonicalise_dtype(facts.output_dtypes.get(output_name)) != "float32":
        raise MetadataMismatchError("R3D-18 output must be float32 scores.")
    return ModelBinding(
        selection=selection,
        model_name=facts.model_name,
        input_name=input_name,
        input_shape=INPUT_SHAPE,
        output_name=output_name,
        output_shape=tuple(output_shape),
        input_dtype="float32",
        output_dtype="float32",
    )


__all__ = [
    "ASSET_ID", "ASSET_FILENAME", "CLASS_COUNT", "INPUT_SHAPE", "ModelBinding",
    "ModelSelection", "SAMPLE_DIR", "SUPPORTED_TARGETS", "bind_model",
    "list_available_assets", "resolve_selection",
]
