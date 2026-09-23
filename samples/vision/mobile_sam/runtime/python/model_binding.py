"""MobileSAM publication selection delegated to the shared SAM catalog."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from samples._shared import sam_binding as _shared
from samples._shared.runtime_meta import RuntimeMetadata

SAMPLE = "mobile_sam"
SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")

ModelSelection = _shared.ModelSelection
ModelBinding = _shared.ModelBinding


def list_available_assets(target: str | None = None):
    """Return manifest-backed encoder/decoder assets for the selected target."""
    return _shared.list_available_assets(SAMPLE, target)


def resolve_selection(
    target: str = "auto",
    *,
    encoder_model_path: str | Path | None = None,
    decoder_model_path: str | Path | None = None,
    encoder_asset_id: str | None = None,
    decoder_asset_id: str | None = None,
):
    """Resolve one exact published encoder/decoder pair."""
    return _shared.resolve_selection(
        SAMPLE,
        target,
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
        encoder_asset_id=encoder_asset_id,
        decoder_asset_id=decoder_asset_id,
        sample_dir=SAMPLE_DIR,
    )


def bind_model(selection, encoder_metadata: Mapping[str, Any] | Any,
               decoder_metadata: Mapping[str, Any] | Any):
    """Delegate atomic encoder/decoder metadata validation to the shared binding."""
    return _shared.bind_model(selection, encoder_metadata, decoder_metadata)


__all__ = ["ModelBinding", "ModelSelection", "RuntimeMetadata", "SAMPLE_DIR", "SUPPORTED_TARGETS", "bind_model", "list_available_assets", "resolve_selection"]
