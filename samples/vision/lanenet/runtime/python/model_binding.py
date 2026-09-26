# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for LaneNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for LaneNet selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "s:lanenet:s100/lanenet256x512.hbm"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated fixed input and named embedding/binary output roles."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    embedding_name: str
    binary_name: str

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


def _asset() -> Asset:
    rows = list_assets("s", "lanenet")
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
    """Resolve LaneNet's exact published asset identity without loading it."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "s100"
    if key != "s100":
        raise BindingError("LaneNet is published only for target s100.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError(
            "An external model path requires --asset-id s:lanenet:s100/lanenet256x512.hbm."
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
    """Bind required roles by exact names; retain declared auxiliary tensors.

    Source prose mentions a third output without identifying it. No name or
    semantic is invented for such outputs; the runner preserves observed raw IO.
    """
    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if expected != selection:
        raise BindingError("Selection differs from exact manifest contract")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if len(meta.model_names) != 1 or len(meta.input_names) != 1:
        raise MetadataMismatchError("Expected exactly one model and one input")
    inp = meta.input_names[0]
    if (
        meta.input_shapes.get(inp) != (1, 3, 256, 512)
        or meta.input_dtypes.get(inp) != "float32"
    ):
        raise MetadataMismatchError("Expected float32 NCHW RGB [1,3,256,512]")
    names = meta.output_names
    required = {"instance_seg_logits", "binary_seg_pred"}
    if len(set(names)) != len(names) or not required.issubset(names):
        raise MetadataMismatchError(
            "Required named embedding and binary outputs are missing or duplicated"
        )
    for name in names:
        shape = meta.output_shapes.get(name, ())
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise MetadataMismatchError(f"Invalid fixed output shape: {name}")
        if meta.output_dtypes.get(name) not in (
            "float16",
            "float32",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
        ):
            raise MetadataMismatchError(f"Unsupported raw output dtype: {name}")
    embedding, binary = "instance_seg_logits", "binary_seg_pred"
    if (
        meta.output_shapes[embedding] != (1, 3, 256, 512)
        or meta.output_dtypes[embedding] != "float32"
    ):
        raise MetadataMismatchError("Embedding requires float32 [1,3,256,512]")
    if (
        meta.output_shapes[binary] not in ((1, 1, 256, 512), (1, 256, 512))
        or meta.output_dtypes[binary] != "int64"
    ):
        raise MetadataMismatchError(
            "Binary prediction requires int64 [1,1,256,512] or [1,256,512]"
        )
    return ModelBinding(selection, meta, inp, embedding, binary)
