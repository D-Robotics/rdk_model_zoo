# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for MODNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for MODNet selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 1, 512, 512)
ASSET_ID = "x5:modnet:modnet_512x512_rgb.bin"


@dataclass(frozen=True)
class ModelSelection:
    """One manual manifest asset and its selected local path."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated MODNet input/output names and source-proven shapes."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


def _asset() -> Asset:
    rows = list_assets("x5", "modnet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """List the manual X5 asset without board or SDK access."""

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
    """Resolve MODNet's exact manual asset identity without loading it."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("MODNet is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:modnet:modnet_512x512_rgb.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source MODNet tensor protocol."""

    # Re-resolve before accepting a caller-created selection.  MODNet's asset is
    # manual, so its manifest identity and explicit path must remain inseparable.
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
        raise MetadataMismatchError("MODNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("MODNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    if meta.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(f"Expected input shape {INPUT_SHAPE}, got {meta.input_shapes.get(input_name)}.")
    if meta.input_dtypes.get(input_name) != "float32":
        raise MetadataMismatchError("MODNet input must be float32.")
    if meta.output_shapes.get(output_name) != OUTPUT_SHAPE:
        raise MetadataMismatchError(f"Expected output shape {OUTPUT_SHAPE}, got {meta.output_shapes.get(output_name)}.")
    if meta.output_dtypes.get(output_name) != "float32":
        raise MetadataMismatchError("MODNet output must be float32.")
    return ModelBinding(selection, meta, input_name, output_name)


__all__ = [
    "ASSET_ID", "BindingError", "INPUT_SHAPE", "ModelBinding", "ModelSelection",
    "OUTPUT_SHAPE", "SAMPLE_DIR", "SUPPORTED_TARGETS", "bind_model",
    "list_available_assets", "resolve_selection", "MetadataMismatchError",
]
