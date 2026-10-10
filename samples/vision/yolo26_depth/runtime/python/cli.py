# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI surface: published selection, options, path checks and evidence IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything around selection and presentation lives here: the published asset
identity/listing/resolver, the parser, the model-free listing/dry-run modes
and the canonical output/report writing.  Nothing in this module runs
inference or measures latency; the tensor contracts and task stages live in
``yolo26_depth.py``.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target

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

#: Canonical files every run writes into its fresh output directory.
CANONICAL_OUTPUTS = (
    "log_depth.npy",
    "depth_native.npy",
    "raw_logit.npy",
    "depth.png",
    "overlay.png",
    "report.json",
)


def build_parser():
    """Build the depth CLI parser with target, asset and input options."""

    p = argparse.ArgumentParser(
        description="YOLO26 relative depth on X5/S100/S100P/S600"
    )
    p.add_argument("--target", choices=("auto",) + TARGETS, default="auto")
    p.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Default n; exact asset-id may infer another variant",
    )
    p.add_argument("--asset-id")
    p.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=Path,
        help="External published asset path; requires exact asset-id",
    )
    p.add_argument(
        "--converted-model",
        action="store_true",
        help="Explicit custom artifact using the asset-id tensor contract; no publisher hash claim",
    )
    p.add_argument(
        "--test-img",
        "--input",
        dest="test_img",
        type=Path,
        default=SAMPLE_DIR / "test_data/bus.jpg",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/yolo26_depth"),
        help="New directory; existing paths are refused",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--priority",
        type=int,
        default=None,
        help="S default 0; X5 leaves SDK default unless set",
    )
    p.add_argument(
        "--bpu-cores",
        type=int,
        nargs="+",
        default=None,
        help="S default [0]; X5 leaves SDK default unless set",
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return p


def run_list_models(target) -> int:
    """Print the manifest-backed assets for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {"asset_id": a.reference, "url": a.url, "sha256": a.sha256}
                for a in list_available_assets(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection, args) -> int:
    """Print the resolved selection without loading a model or SDK."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "variant": selection.variant,
                "profile": selection.profile,
                "asset_id": (
                    None if selection.converted_model else selection.asset.reference
                ),
                "contract_reference": selection.asset.reference,
                "artifact_origin": (
                    "user-converted" if selection.converted_model else "published-manifest"
                ),
                "model_path": str(selection.model_path),
                "output_semantics": (
                    "raw_logit" if selection.profile == "lite" else "calibrated_log_depth"
                ),
                "sdk_loaded": False,
                "downloaded": False,
            },
            indent=2,
        )
    )
    return 0


def read_bgr_image(path: Path):
    """Read one BGR image; decode failures name the exact path."""

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode input image: {path}")
    return image


def colorize_depth(depth):
    """Source 2nd/98th percentile normalization and inverted TURBO palette.

    Nonempty finite floating-point H×W arrays are required. Runtime already
    rejects nonfinite inference; rendering never hides invalid model output.
    """
    import cv2
    import numpy as np

    if (
        not isinstance(depth, np.ndarray)
        or depth.ndim != 2
        or not depth.size
        or depth.dtype.kind != "f"
        or not np.isfinite(depth).all()
    ):
        raise ValueError("Expected nonempty finite floating-point HxW depth")
    low, high = np.percentile(depth, (2, 98))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)


def save_depth_evidence(
    output: Path,
    *,
    selection,
    runner,
    binding,
    image_path: Path,
    image,
    details,
    priority,
    bpu_cores,
) -> None:
    """Write log/native/raw tensors, displays and the provenance report."""

    import cv2
    import numpy as np

    from utils.py_utils.assets import sha256_file
    from utils.py_utils.runtime_meta import metadata_evidence

    result = details.result
    color = colorize_depth(result.depth_native)
    overlay = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)
    report = {
        "schema_version": "1.0",
        "target": selection.target,
        "variant": selection.variant,
        "profile": selection.profile,
        "asset_id": (
            None if selection.converted_model else selection.asset.reference
        ),
        "contract_reference": selection.asset.reference,
        "artifact_origin": (
            "user-converted" if selection.converted_model else "published-manifest"
        ),
        "model_path": str(selection.model_path),
        "model_sha256": sha256_file(selection.model_path),
        "publisher_sha256": (
            None if selection.converted_model else selection.asset.sha256
        ),
        "input": str(image_path),
        "input_sha256": sha256_file(image_path),
        "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
        "metadata": metadata_evidence(binding.metadata),
        "log_depth_shape": list(result.log_depth.shape),
        "depth_native_shape": list(result.depth_native.shape),
        "warmup": details.warmup,
        "latency_ms": details.latency_ms,
        "latency_scope": "one forward including transport validation and owned output copy; not BPU-only",
        "priority": priority,
        "bpu_cores": bpu_cores,
        "geometry": vars(result.context),
        "depth_units": "relative; not calibrated metres",
    }
    if result.raw_logit is not None:

        a, b = LITE_CALIBRATION[selection.variant]
        report["calibration"] = {"cal_a": a, "cal_b": b, "clip": [-4, 5]}
        report["raw_logit_shape"] = list(result.raw_logit.shape)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "log_depth.npy", result.log_depth, allow_pickle=False)
    np.save(output / "depth_native.npy", result.depth_native, allow_pickle=False)
    if result.raw_logit is not None:
        np.save(output / "raw_logit.npy", result.raw_logit, allow_pickle=False)
    for name, value in (("depth.png", color), ("overlay.png", overlay)):
        if not cv2.imwrite(str(output / name), value):
            raise OSError(f"Could not write {output/name}")
    text = json.dumps(report, indent=2)
    (output / "report.json").write_text(text + "\n")
    print(text)
