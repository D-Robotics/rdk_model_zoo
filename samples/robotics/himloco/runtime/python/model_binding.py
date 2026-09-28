# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact X5 publication and fixed float policy tensor binding."""

from dataclasses import dataclass
from pathlib import Path
from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError

SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:himloco:himloco_go2_bayese_1x270.bin"


@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str = "obs_history"
    output_name: str = "actions"

    @property
    def model_name(self):
        return self.metadata.model_name


def resolve_selection(target="auto", *, model_path=None, asset_id=None):
    """Resolve exact X5 asset; alternate paths require its explicit asset identity."""
    if target == "auto":
        from samples._shared.platforms import detect_target

        target = detect_target()
    if target != "x5":
        raise ValueError("HIMLoco has a published fused model only for x5")
    assets = list_assets("x5", "himloco")
    if len(assets) != 1 or assets[0].reference != ASSET_ID or assets[0].format != "bin":
        raise ValueError("Expected the single published HIMLoco BIN asset")
    asset = assets[0]
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Expected asset-id {asset.reference}")
    if model_path is not None and asset_id is None:
        raise ValueError(
            "An external model path requires the explicit matching asset-id"
        )
    path = (
        Path(model_path).expanduser()
        if model_path is not None
        else SAMPLE_DIR / "model/bayes-e" / asset.filename
    )
    if path.suffix != ".bin":
        raise ValueError("HIMLoco requires an X5 .bin artifact")
    return ModelSelection(target, asset, path, model_path is not None)


def validate_selection(selection):
    expected = resolve_selection(
        selection.target,
        model_path=selection.model_path if selection.explicit_model_path else None,
        asset_id=selection.asset.reference,
    )
    if selection != expected:
        raise ValueError("Selection differs from the declared publication and path")


def bind_model(selection, metadata):
    """Require one model, obs_history F32 [1,270], actions F32 [1,12]."""
    validate_selection(selection)
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("HIMLoco requires exactly one packed model")
    for side, name, shape in (
        ("input", "obs_history", (1, 270)),
        ("output", "actions", (1, 12)),
    ):
        if (
            getattr(meta, side + "_names") != (name,)
            or getattr(meta, side + "_shapes").get(name) != shape
            or getattr(meta, side + "_dtypes").get(name) != "float32"
        ):
            raise MetadataMismatchError(f"Expected {side} {name} float32 {shape}")
    return ModelBinding(selection, meta)
