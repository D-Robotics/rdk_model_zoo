# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE CLI surface: published selection, options, inspection modes and result display.

``main.py`` stays a thin entry that constructs the model and calls ``predict``;
everything around selection and presentation lives here: the published YOLOE
asset matrix and resolver, the parser, the model-free listing/dry-run modes,
the result report and the annotated-image rendering.  Nothing in this module
runs segmentation or loads a board SDK; the task stages and tensor binding
live in ``yoloe.py``.
"""

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import re
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256

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


def build_parser():
    p = argparse.ArgumentParser(
        description="YOLOE-11/26 prompt-free segmentation with native float or SCALE-quantized outputs."
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument(
        "--variant",
        default=None,
        help="11s/m/l or 26n/s/m/l/x; defaults x5/s100=11s, s100p=26n.",
    )
    p.add_argument(
        "--asset-id", default=None, help="Exact original publication identity."
    )
    p.add_argument("--model-path", default=None)
    p.add_argument(
        "--local-float-sha256",
        default=None,
        help="SHA-256 of your separately converted floating-output model; requires --model-path.",
    )
    p.add_argument("--test-img", default=str(SAMPLE_DIR / "test_data/office_desk.jpg"))
    p.add_argument("--label-file", default=str(SAMPLE_DIR / "test_data/classes.names"))
    p.add_argument("--img-save-path", default=str(SAMPLE_DIR / "test_data/result.jpg"))
    p.add_argument("--score-thres", type=float, default=0.25)
    p.add_argument(
        "--nms-thres",
        type=float,
        default=None,
        help="YOLOE-11 default .7; forbidden for 26.",
    )
    p.add_argument("--resize-type", type=int, choices=(0, 1), default=1)
    p.add_argument(
        "--no-morph",
        action="store_true",
        help="Disable source CLI default opening for S YOLOE-11 ROI masks.",
    )
    p.add_argument(
        "--no-contour", action="store_true", help="Disable mask contour outlines."
    )
    p.add_argument("--max-det", type=int, default=300, help="YOLOE-26 only, 1..8400.")
    p.add_argument(
        "--multi-label",
        action="store_true",
        help="YOLOE-26 only; permit multiple classes per anchor.",
    )
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def run_list_models(target) -> int:
    """Print the manifest model matrix for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {
                    "target": t,
                    "variant": v,
                    "asset_id": a.reference,
                    "published_float": t == "x5",
                }
                for t, v, a in list_models(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection, config) -> int:
    """Print the resolved selection and config without loading a model."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "variant": selection.variant,
                "source_asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "published_float": selection.published_float,
                "local_float_sha256": selection.local_float_sha256,
                "runtime_metadata_verified": False,
                "config": asdict(config),
                "status": "metadata validation required",
            },
            indent=2,
        )
    )
    return 0


def validate_scheduling(args) -> None:
    """Reject scheduling values the run would refuse, including during dry-run."""

    if (
        not 0 <= args.priority <= 255
        or not args.bpu_cores
        or any(c < 0 for c in args.bpu_cores)
    ):
        raise ValueError("Invalid scheduling priority/core list.")


def print_result_report(result, selection, args) -> None:
    """Print the JSON detection summary for one finished prediction."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "variant": selection.variant,
                "count": len(result.scores),
                "class_ids": result.class_ids.tolist(),
                "scores": result.scores.tolist(),
                "mask_layout": result.mask_layout,
                "image_saved": str(Path(args.img_save_path).expanduser()),
            },
            indent=2,
        )
    )


@dataclass(frozen=True)
class Config:
    score_thres: float = 0.25
    nms_thres: float | None = None
    resize_type: int = 1
    do_morph: bool = False
    max_det: int = 300
    single_label: bool = True


def validate_config(selection, cfg):
    if not np.isfinite(cfg.score_thres) or not 0 < cfg.score_thres < 1:
        raise ValueError("score_thres must be finite and strictly between 0 and 1.")
    if cfg.resize_type not in (0, 1) or not isinstance(cfg.do_morph, bool):
        raise ValueError("resize_type must be 0/1 and do_morph must be boolean.")
    if (
        isinstance(cfg.max_det, bool)
        or not isinstance(cfg.max_det, Integral)
        or not 1 <= cfg.max_det <= 8400
        or not isinstance(cfg.single_label, bool)
    ):
        raise ValueError(
            "max_det must be an integer in 1..8400; single_label must be boolean."
        )
    if selection.variant.startswith("26"):
        if cfg.nms_thres is not None or cfg.resize_type != 1 or cfg.do_morph:
            raise ValueError(
                "YOLOE-26 uses fixed round/114 letterbox, no NMS and no morphology."
            )
    else:
        if cfg.max_det != 300 or not cfg.single_label:
            raise ValueError("max_det and multi-label apply only to YOLOE-26.")
        if cfg.nms_thres is not None and (
            not np.isfinite(cfg.nms_thres) or not 0 <= cfg.nms_thres <= 1
        ):
            raise ValueError("nms_thres must be finite in [0,1].")
        if selection.target == "x5" and cfg.do_morph:
            raise ValueError("Morphology applies only to S YOLOE-11 ROI masks.")


def load_inputs(image_path, label_path):
    import cv2

    image = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    raw = Path(label_path).expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise ValueError(
            "Vocabulary checksum mismatch; the PF model requires its fixed ordered 4585 classes."
        )
    labels = raw.decode("utf-8").splitlines()
    if len(labels) != 4585:
        raise ValueError("Expected 4585 vocabulary entries.")
    return image, labels


def draw_result(image, result, labels, *, contours=True):
    import cv2
    import numpy as np

    canvas = image.copy()
    for box, score, cls, mask in zip(
        result.boxes, result.scores, result.class_ids, result.masks
    ):
        cls = int(cls)
        if not 0 <= cls < len(labels):
            raise ValueError("Class ID is outside the fixed vocabulary.")
        x1, y1, x2, y2 = box.astype(int)
        color = np.array(
            [(37 * cls + 50) % 256, (67 * cls + 80) % 256, (97 * cls + 110) % 256],
            dtype=np.uint8,
        )
        view = canvas if result.mask_layout == "full" else canvas[y1:y2, x1:x2]
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != view.shape[:2]:
            raise ValueError("Mask geometry differs from its declared layout.")
        view[selected] = (view[selected].astype(np.float32) * 0.6 + color * 0.4).astype(
            np.uint8
        )
        if contours and selected.size:
            curves, _ = cv2.findContours(
                selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(view, curves, -1, tuple(int(v) for v in color), 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2)
        cv2.putText(
            canvas,
            f"{labels[cls]} {float(score):.3f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            tuple(int(v) for v in color),
            1,
        )
    return canvas


def save_result(path, image, result, labels, *, contours=True):
    import cv2

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), draw_result(image, result, labels, contours=contours)
    ):
        raise OSError(f"Cannot save result: {destination}")
