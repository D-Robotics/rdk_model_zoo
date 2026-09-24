# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact X5 YOLOWorld asset and tensor metadata binding."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata

SAMPLE_DIR = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: Asset
    model_path: Path

@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    model_name: str
    image_input_name: str
    text_input_name: str
    score_output_name: str
    box_output_name: str


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """Return the one published X5 model asset; no target detection is done."""
    key = (target or "auto").lower()
    if key in {"auto", "x5"}:
        assets = tuple(list_assets("x5", "yoloworld"))
        if len(assets) != 1 or assets[0].filename != "yolo_world.bin":
            raise ValueError("The published YOLOWorld asset facts changed; review this sample.")
        return assets
    if key in {"s100", "s100p", "s600"}:
        return ()
    raise ValueError(f"Unknown target {target!r}.")


def resolve_selection(target: str = "auto", *, model_path: str | Path | None = None,
                      asset_id: str | None = None) -> ModelSelection:
    """Resolve X5 and require exact asset identity for an external model path."""
    concrete = resolve_target(target)
    if concrete != "x5":
        raise ValueError(f"YOLOWorld has a published model only for x5, not {concrete}.")
    asset = list_available_assets(concrete)[0]
    if model_path is not None and asset_id is None:
        raise ValueError("--model-path requires the exact --asset-id.")
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / asset.filename
    return ModelSelection(concrete, asset, path)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | dict[str, Any]) -> ModelBinding:
    """Bind observed metadata to the fixed source protocol without guessing names."""
    # Re-resolve the manifest first: a caller-created ModelSelection (for example
    # a forged same-shape publication row) must not be trusted at this boundary.
    resolved = resolve_selection(
        selection.target, model_path=selection.model_path, asset_id=selection.asset.reference
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
    ):
        raise ValueError("ModelSelection does not match the exact published asset and path.")
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if facts.model_names and facts.model_name not in facts.model_names:
        raise MetadataMismatchError("Selected runtime model is not among model_names.")
    if len(facts.input_names) != 2 or len(facts.output_names) != 2:
        raise MetadataMismatchError("YOLOWorld requires exactly two inputs and two outputs.")
    for name, shape, dtype in (
        (facts.input_names[0], (1, 3, 640, 640), "float32"),
        (facts.input_names[1], (1, 32, 512, 1), "float32"),
    ):
        if facts.input_shapes.get(name) != shape or facts.input_dtypes.get(name) != dtype:
            raise MetadataMismatchError(f"Input {name!r} is not {dtype}[{','.join(map(str, shape))}].")
    expected = {"score": ((1, 8400, 32), (1, 8400, 32, 1)), "box": ((1, 8400, 4), (1, 8400, 4, 1))}
    score_name = box_name = None
    for name in facts.output_names:
        shape = facts.output_shapes.get(name)
        dtype = facts.output_dtypes.get(name)
        if dtype != "float32":
            raise MetadataMismatchError(f"Output {name!r} must preserve native float32, got {dtype!r}.")
        if shape in expected["score"]:
            if score_name is not None:
                raise MetadataMismatchError("More than one score output matches the source protocol.")
            score_name = name
        elif shape in expected["box"]:
            if box_name is not None:
                raise MetadataMismatchError("More than one box output matches the source protocol.")
            box_name = name
        else:
            raise MetadataMismatchError(f"Output {name!r} has unsupported shape {shape!r}.")
    if score_name is None or box_name is None:
        raise MetadataMismatchError("Missing YOLOWorld score or box output.")
    return ModelBinding(selection, facts.model_name, facts.input_names[0], facts.input_names[1], score_name, box_name)
