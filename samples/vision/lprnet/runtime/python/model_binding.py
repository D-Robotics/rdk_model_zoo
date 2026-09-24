# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for LPRNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for LPRNet selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
INPUT_SHAPE = (1, 3, 24, 94)
# Accepted logits layouts, each bound exactly as reported and matched
# strictly on every runtime call.  (1, 68, 18, 1) is the measured protocol
# of the one published X5 ``lpr.bin`` (board metadata 2026-09-24).  The 3D
# (1, 68, 18) layout is the old unified-contract/host-fixture shape kept
# for API compatibility with existing host tests and injected runners; no
# published SDK artifact has been observed reporting it.  No other rank or
# axis order is accepted — the binding never reshapes, squeezes, or
# permutes.  Singleton elimination for the CTC decoder happens only in the
# task's post_process stage.
OUTPUT_SHAPES = ((1, 68, 18), (1, 68, 18, 1))
CTC_LOGITS_SHAPE = (68, 18)
ASSET_ID = "x5:lprnet:lpr.bin"


@dataclass(frozen=True)
class ModelSelection:
    """One exact manifest asset and the path selected for execution."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    """Validated input/output names and shapes observed from one model.

    ``output_shape`` is the complete native logits shape reported by the
    bound runtime metadata (see ``OUTPUT_SHAPES``); every runtime call must
    reproduce it exactly.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str
    output_shape: tuple[int, ...]

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


def _asset() -> Asset:
    rows = list_assets("x5", "lprnet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """List the one X5 asset without detecting a board or loading an SDK."""

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
    """Resolve an exact published LPRNet model without touching the runtime."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("LPRNet is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:lprnet:lpr.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate source-proven LPRNet metadata and bind runtime tensor names."""

    # Do not trust a caller-created ModelSelection.  Re-resolving the manifest
    # row also protects the binding boundary from an asset-id/path substitution.
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
        raise MetadataMismatchError("LPRNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("LPRNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    if meta.input_shapes.get(input_name) != INPUT_SHAPE:
        raise MetadataMismatchError(f"Expected input shape {INPUT_SHAPE}, got {meta.input_shapes.get(input_name)}.")
    if meta.input_dtypes.get(input_name) != "float32":
        raise MetadataMismatchError("LPRNet input must be float32.")
    if meta.output_shapes.get(output_name) not in OUTPUT_SHAPES:
        raise MetadataMismatchError(
            f"Expected LPRNet logits metadata shape in {OUTPUT_SHAPES} "
            "(classes=68, timesteps=18; rank 4 is accepted only as "
            f"(1, 68, 18, 1)), got {meta.output_shapes.get(output_name)}."
        )
    if meta.output_dtypes.get(output_name) != "float32":
        raise MetadataMismatchError("LPRNet output must be float32.")
    return ModelBinding(
        selection, meta, input_name, output_name, meta.output_shapes[output_name]
    )


__all__ = [
    "ASSET_ID", "BindingError", "CTC_LOGITS_SHAPE", "INPUT_SHAPE", "ModelBinding",
    "ModelSelection", "OUTPUT_SHAPES", "SAMPLE_DIR", "SUPPORTED_TARGETS", "bind_model",
    "list_available_assets", "resolve_selection", "MetadataMismatchError",
]
