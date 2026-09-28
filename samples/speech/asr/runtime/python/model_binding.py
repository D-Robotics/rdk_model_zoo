# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact S100/S600 ASR identity and SDK tensor checks; no SDK import."""

from dataclasses import dataclass
from pathlib import Path
from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples._shared.quantization import validate_scale_quantization

SAMPLE_DIR = Path(__file__).resolve().parents[2]
TARGETS = ("s100", "s600")
VOCABULARY_SIZE = 3503


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
    assets = tuple(list_assets("s", "asr"))
    if {a.reference for a in assets} != {f"s:asr:{t}/asr.hbm" for t in TARGETS}:
        raise ValueError("Expected exact S100/S600 ASR publications")
    return tuple(
        a for a in assets if target == "auto" or a.filename.startswith(target + "/")
    )


def resolve_selection(target="auto", *, asset_id=None, model_path=None):
    if target == "auto":
        from samples._shared.platforms import detect_target

        target = detect_target()
    if target not in TARGETS:
        raise ValueError("ASR is published only for s100 and s600")
    asset = list_available_assets(target)[0]
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Expected asset-id {asset.reference}")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model path requires the exact --asset-id")
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
        raise MetadataMismatchError("ASR requires exactly one model/input/output")
    name, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(name) != (1, 30000)
        or meta.input_dtypes.get(name) != "float32"
    ):
        raise MetadataMismatchError(
            "ASR input must be float32 [1,30000] for the fixed frontend"
        )
    shape = meta.output_shapes.get(out, ())
    if (
        len(shape) != 3
        or shape[0] != 1
        or type(shape[1]) is not int
        or shape[1] <= 0
        or shape[2] != VOCABULARY_SIZE
    ):
        raise MetadataMismatchError("ASR logits must be [1,T,3503], T > 0")
    dtype = meta.output_dtypes.get(out)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError("Unsupported ASR output dtype")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(out), shape)
    return Binding(selection, meta, name, out)
