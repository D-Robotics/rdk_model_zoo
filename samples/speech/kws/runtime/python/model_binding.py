# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact S100 KWS identity and SDK tensor checks; no SDK import."""

from dataclasses import dataclass
from pathlib import Path
from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.quantization import validate_scale_quantization

SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "s:kws:s100/kws.hbm"


@dataclass(frozen=True)
class Selection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self):
        return self.metadata.model_name


def list_available_assets(target="auto"):
    if target not in ("auto", "s100", "s100p", "s600", "x5"):
        raise ValueError(f"Unknown target {target!r}")
    if target not in ("auto", "s100"):
        return ()
    assets = tuple(list_assets("s", "kws"))
    if len(assets) != 1 or assets[0].reference != ASSET_ID:
        raise ValueError("Expected the exact S100 KWS publication")
    return assets


def resolve_selection(target="auto", *, asset_id=None, model_path=None):
    if target == "auto":
        from utils.py_utils.platforms import detect_target

        target = detect_target()
    if target != "s100":
        raise ValueError("KWS has a published asset only for s100")
    asset = list_available_assets("s100")[0]
    if asset_id is not None and asset_id != ASSET_ID:
        raise ValueError(f"Expected asset-id {ASSET_ID}")
    if model_path is not None and asset_id is None:
        raise ValueError(f"An external model path requires --asset-id {ASSET_ID}")
    return Selection(
        target,
        asset,
        (
            Path(model_path).expanduser()
            if model_path
            else SAMPLE_DIR / "model" / asset.filename
        ),
        model_path is not None,
    )


def bind_model(selection, metadata):
    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if selection != expected:
        raise ValueError("Selection does not match the published identity/path")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if (
        meta.model_names != (meta.model_name,)
        or len(meta.input_names) != 1
        or len(meta.output_names) != 1
    ):
        raise MetadataMismatchError("KWS requires exactly one model/input/output")
    name, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(name) != (1, 373, 80)
        or meta.input_dtypes.get(name) != "float32"
    ):
        raise MetadataMismatchError(
            "KWS input must be float32 [1,373,80] for the fixed frontend"
        )
    shape = meta.output_shapes.get(out, ())
    if not shape or shape[0] != 1 or any(type(n) is not int or n <= 0 for n in shape):
        raise MetadataMismatchError(
            "KWS score output requires finite positive dimensions and batch one"
        )
    dtype = meta.output_dtypes.get(out)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError("Unsupported KWS output dtype")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(out), shape)
    return Binding(selection, meta, name, out)
