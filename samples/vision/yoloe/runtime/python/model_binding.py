# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact YOLOE publication selection, separate from local float conversion identity."""

from dataclasses import dataclass
from pathlib import Path
import re
from samples._shared.assets import Asset, list_assets
from samples._shared.platforms import resolve_target
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLSegmentationContract,
    LTRBSegmentationContract,
    ModelSelection as RuntimeSelection,
)

SAMPLE_DIR = Path(__file__).resolve().parents[2]
DEFAULTS = {"x5": "11s", "s100": "11s", "s100p": "26n"}


@dataclass(frozen=True)
class Selection:
    target: str
    variant: str
    asset: Asset
    model_path: Path
    local_float_sha256: str | None = None

    @property
    def local_float(self):
        return self.local_float_sha256 is not None

    @property
    def published_float(self):
        return self.target == "x5"


class PF11Contract(DFLSegmentationContract):
    """Keep both RGB-shaped X5 descriptor layouts explicitly accepted by its source."""

    allow_packed_nhwc = True

    def __init__(self):
        super().__init__(classes=4585)


class PF26Contract(LTRBSegmentationContract):
    """Direct offsets with deterministic Top-K; no NMS policy is implied."""

    def __init__(self):
        super().__init__(classes=4585)
        object.__setattr__(self, "nms", "none")


def list_models(target=None):
    """Return (target, variant, publication) rows, including unavailable float routes."""
    if target not in (None, "auto", "x5", "s100", "s100p", "s600"):
        raise ValueError(f"Unknown target {target!r}.")
    rows = []
    for group, sample in [("x5", "yoloe"), ("s", "yoloe11_seg"), ("s", "yoloe26_seg")]:
        for asset in list_assets(group, sample):
            match = re.fullmatch(
                r"(?:nash-[em]/)?yoloe_(11[sml]|26[nslmx])_seg_pf_(bayese|nashe|nashm)_640x640_nv12\.(bin|hbm)",
                asset.filename,
            )
            if match is None:
                raise ValueError(
                    f"Unexpected YOLOE publication: {asset.reference}; review its protocol."
                )
            variant, march, ext = match.groups()
            concrete = {"bayese": "x5", "nashe": "s100", "nashm": "s100p"}[march]
            if target in (None, "auto", concrete):
                rows.append((concrete, variant, asset))
    return tuple(rows)


def resolve_selection(
    target="auto",
    *,
    variant=None,
    asset_id=None,
    model_path=None,
    local_float_sha256=None,
):
    """No implicit downloads or local-file trust; a custom float file needs its digest."""
    concrete = resolve_target(target)
    rows = list_models(concrete)
    if asset_id is not None:
        rows = tuple(row for row in rows if row[2].reference == asset_id)
    else:
        variant = variant or DEFAULTS.get(concrete)
    if variant is not None:
        rows = tuple(row for row in rows if row[1] == variant)
    if len(rows) != 1:
        raise ValueError(
            f"No unique YOLOE asset for target={concrete}, variant={variant}, asset_id={asset_id}."
        )
    _, variant, asset = rows[0]
    if local_float_sha256 is not None:
        if (
            model_path is None
            or re.fullmatch("[0-9a-fA-F]{64}", local_float_sha256) is None
        ):
            raise ValueError(
                "A local float model requires --model-path and a 64-digit --local-float-sha256."
            )
        local_float_sha256 = local_float_sha256.lower()
    elif model_path is not None and asset_id is None:
        raise ValueError(
            "--model-path requires the exact --asset-id or --local-float-sha256."
        )
    path = (
        Path(model_path).expanduser()
        if model_path is not None
        else SAMPLE_DIR / "model" / concrete / asset.filename
    )
    return Selection(concrete, variant, asset, path, local_float_sha256)


def validate_selection(selection):
    """Reject caller-forged publication facts as well as target/variant drift."""
    current = resolve_selection(
        selection.target,
        variant=selection.variant,
        asset_id=selection.asset.reference,
        model_path=selection.model_path,
        local_float_sha256=selection.local_float_sha256,
    )
    if current != selection:
        raise ValueError("Selection differs from the active YOLOE manifest.")


def runtime_selection(selection):
    validate_selection(selection)
    contract = PF26Contract() if selection.variant.startswith("26") else PF11Contract()
    return RuntimeSelection(
        str(selection.model_path),
        target=selection.target,
        task="segment",
        contract=contract,
        input_shape=(640, 640),
        family="yoloe",
        artifact_id=selection.asset.reference,
    )
