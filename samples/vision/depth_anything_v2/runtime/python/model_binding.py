# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for Depth Anything V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for Depth Anything V2 selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "s:depth_anything_v2:s100/depth_any.hbm"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated fixed RGB featuremap and relative-depth tensor protocol."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


def _asset() -> Asset:
    rows = list_assets("s", "depth_anything_v2")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """List the published S100 asset without board or SDK access."""

    if target in (None, "auto", "s100"):
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
    """Resolve Depth Anything V2's exact published asset identity without loading it."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "s100"
    if key != "s100":
        raise BindingError("Depth Anything V2 is published only for target s100.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError(
            "An external model path requires --asset-id s:depth_anything_v2:s100/depth_any.hbm."
        )
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=(
            Path(model_path).expanduser()
            if model_path
            else SAMPLE_DIR / "model" / asset.filename
        ),
        explicit_model_path=model_path is not None,
    )


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Bind the source 518×686 RGB featuremap and relative-depth output.

    Internal int16 quantization in the source guide is not evidence that the
    public output tensor is int16. Accept only the float32 source IO contract.
    """
    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if expected != selection:
        raise BindingError("Selection differs from the exact published asset")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if (
        len(meta.model_names) != 1
        or len(meta.input_names) != 1
        or len(meta.output_names) != 1
    ):
        raise MetadataMismatchError("Expected one model, one input and one output")
    inp, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(inp) != (1, 3, 518, 686)
        or meta.input_dtypes.get(inp) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 RGB NCHW [1,3,518,686]")
    if (
        meta.output_shapes.get(out) != (1, 518, 686)
        or meta.output_dtypes.get(out) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 depth [1,518,686]")
    return ModelBinding(selection, meta, inp, out)
