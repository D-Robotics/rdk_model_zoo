# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Published asset identity and mixed-profile depth tensor contracts."""

from dataclasses import dataclass
from pathlib import Path

from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError

SAMPLE_DIR = Path(__file__).resolve().parents[2]
TARGETS = ("x5", "s100", "s100p", "s600")
VARIANTS = ("n", "s", "m", "l", "x")
MARCH = {"s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}
LITE_CALIBRATION = {"l": (1.0, -0.2498779296875), "x": (1.0, -0.316650390625)}


@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool
    variant: str
    profile: str
    converted_model: bool = False


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str
    input_size: int = 768

    @property
    def model_name(self):
        return self.metadata.model_name


def list_available_assets(target=None):
    if target not in (None, "auto", *TARGETS):
        raise ValueError(f"Unknown target {target!r}")
    rows = tuple(list_assets("x5", "yolo26_depth")) + tuple(
        list_assets("s", "yolo26_depth")
    )
    if target in (None, "auto"):
        return rows
    if target == "x5":
        return tuple(a for a in rows if a.filename.endswith(".bin"))
    return tuple(a for a in rows if a.filename.startswith(MARCH[target] + "/"))


def resolve_selection(
    target="auto",
    *,
    variant=None,
    asset_id=None,
    model_path=None,
    converted_model=False,
):
    if converted_model and (model_path is None or asset_id is None):
        raise ValueError(
            "Converted models require --model-path and an exact --asset-id contract reference"
        )
    if asset_id is not None:
        matches = [a for a in list_available_assets() if a.reference == asset_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown YOLO26 Depth asset-id {asset_id!r}")
        asset = matches[0]
        inferred = (
            "x5"
            if asset.filename.endswith(".bin")
            else next(t for t, m in MARCH.items() if asset.filename.startswith(m + "/"))
        )
        inferred_variant = Path(asset.filename).name[len("yolo26")]
        if target not in (None, "auto", inferred) or variant not in (
            None,
            inferred_variant,
        ):
            raise ValueError("target/variant and asset-id select different artifacts")
        target, variant = inferred, inferred_variant
    target = resolve_target(target) if target in (None, "auto") else target
    variant = "n" if variant is None else variant
    if target not in TARGETS or variant not in VARIANTS:
        raise ValueError(f"Unsupported target/variant: {target!r}/{variant!r}")
    profile = "lite" if target != "x5" and variant in LITE_CALIBRATION else "nv12"
    if target == "x5":
        filename = f"yolo26{variant}_depth_bayese_768x768_nv12.bin"
    else:
        march = MARCH[target]
        suffix = march.replace("-", "")
        filename = (
            f"{march}/yolo26{variant}_depth_lite_{suffix}_768x768.hbm"
            if profile == "lite"
            else f"{march}/yolo26{variant}_depth_{suffix}_768x768_nv12.hbm"
        )
    rows = [a for a in list_available_assets(target) if a.filename == filename]
    if len(rows) != 1:
        raise ValueError(f"Expected one published artifact for {target}/{variant}")
    if model_path is not None and asset_id is None:
        raise ValueError("External model paths require the exact --asset-id")
    path = (
        Path(model_path).expanduser()
        if model_path is not None
        else SAMPLE_DIR / "model" / filename
    )
    return ModelSelection(
        target, rows[0], path, model_path is not None, variant, profile, converted_model
    )


def bind_model(selection, metadata):
    resolved = resolve_selection(
        selection.target,
        variant=selection.variant,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
        converted_model=selection.converted_model,
    )
    if selection != resolved:
        raise ValueError(
            "ModelSelection differs from the manifest identity/profile/path"
        )
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
        raise MetadataMismatchError(
            "Depth requires exactly one model, one input and one output"
        )
    inp, out = meta.input_names[0], meta.output_names[0]
    if selection.profile == "lite":
        shapes = ((1, 3, 768, 768),)
        dtype = "float32"
        semantics = ("raw_logit",)
    else:
        shapes = ((1, 3, 768, 768), (1, 768, 768, 3), (1, 1152, 768, 1))
        dtype = "nv12"
        semantics = ("log_depth", "calibrated_log_depth")
    if meta.input_shapes.get(inp) not in shapes or meta.input_dtypes.get(inp) != dtype:
        raise MetadataMismatchError(
            f"{selection.profile} requires 768-square {dtype} input metadata"
        )
    if (
        meta.output_shapes.get(out) not in ((1, 192, 192, 1), (1, 1, 192, 192))
        or meta.output_dtypes.get(out) != "float32"
    ):
        raise MetadataMismatchError(
            "Expected one float32 NHWC/NCHW 192-square depth channel"
        )
    declared = meta.output_semantics
    if isinstance(declared, dict):
        declared = declared.get(out)
    if declared is not None and declared not in semantics:
        raise MetadataMismatchError(
            f"Output semantic {declared!r} conflicts with {selection.profile}"
        )
    return ModelBinding(selection, meta, inp, out)
