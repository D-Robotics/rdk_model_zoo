# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for PP-LiteSeg."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for PP-LiteSeg selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated PP-LiteSeg tensor protocol; class-map output semantics are fixed by the source runtime."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


def _asset() -> Asset:
    rows = list_assets("x5", "pp_liteseg")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """List the published X5 asset without board or SDK access."""

    if target in (None, "auto", "x5"):
        return (_asset(),)
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}.")


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: str | None = None,
    model_path: str | Path | None = None,
) -> ModelSelection:
    """Resolve PP-LiteSeg's exact published asset identity without loading it."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("PP-LiteSeg is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source PP-LiteSeg tensor protocol."""

    # Re-resolve caller-created selections: identity and path stay inseparable.
    resolved = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
        or selection.explicit_model_path != resolved.explicit_model_path
    ):
        raise BindingError("ModelSelection does not match the exact manifest asset and path.")

    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("PP-LiteSeg artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("PP-LiteSeg requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    shape = meta.input_shapes.get(input_name, ())
    if shape not in ((1,3,512,1024), (1,512,1024,3), (1,768,1024,1), (768,1024)):
        raise MetadataMismatchError("PP-LiteSeg requires 1024x512 NV12 geometry.")
    if meta.input_dtypes.get(input_name) != "nv12":
        raise MetadataMismatchError("PP-LiteSeg input must be NV12.")
    if meta.output_shapes.get(output_name) != (1,512,1024,1):
        raise MetadataMismatchError("PP-LiteSeg expects a (1,512,1024,1) class map, not logits.")
    if meta.output_dtypes.get(output_name) != "int32":
        raise MetadataMismatchError("PP-LiteSeg class-map output must be int32.")
    return ModelBinding(selection, meta, input_name, output_name)
