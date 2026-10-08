# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Planning CLI surface: option declarations, path checks and evidence IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything presentational — the parser, the model-free listing/dry-run modes
and the canonical output/report writing — lives here.  Nothing in this module
plans trajectories or runs inference.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

from dataclasses import dataclass
from pathlib import Path

from utils.py_utils.assets import list_assets
from utils.py_utils.platforms import resolve_target

SAMPLE_DIR = Path(__file__).resolve().parents[2]
TARGETS = ("s100p", "s600")


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (s100p or s600).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: object
    model_path: Path
    explicit_model_path: bool = False


def list_available_assets(target=None):
    """List published DiffusionDrive assets for a target.

    Args:
        target: Concrete target filter; ``auto``/None lists all publications.

    Returns:
        tuple: Published assets in manifest order.

    Raises:
        ValueError: The target is unknown.
    """
    rows = tuple(list_assets("s", "diffusiondrive"))
    if target in (None, "auto"):
        return rows
    if target not in ("x5", "s100", "s100p", "s600"):
        raise ValueError(f"Unknown target {target}")
    return tuple(a for a in rows if a.filename.startswith(target + "/"))


def resolve_selection(target="auto", *, asset_id=None, model_path=None):
    """Resolve one exact published DiffusionDrive planning asset.

    Args:
        target: ``auto`` resolves the executing board; s100p/s600 publish.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        ValueError: The target or asset combination is invalid.
    """
    rows = list_available_assets()
    if asset_id is not None:
        rows = tuple(a for a in rows if a.reference == asset_id)
        if len(rows) != 1:
            raise ValueError("Unknown DiffusionDrive asset identity")
        asset_target = rows[0].filename.split("/")[0]
        if target == "auto":
            target = asset_target
    selected = resolve_target(target)
    if selected not in TARGETS:
        raise ValueError("DiffusionDrive assets support S100P and S600 only")
    rows = tuple(a for a in rows if a.filename.startswith(selected + "/"))
    if len(rows) != 1:
        raise ValueError("Target and asset identity must match exactly")
    if model_path is not None and asset_id is None:
        raise ValueError("External model path requires exact --asset-id")
    asset = rows[0]
    return ModelSelection(
        selected,
        asset,
        (
            Path(model_path).expanduser()
            if model_path is not None
            else SAMPLE_DIR / "model" / asset.filename
        ),
        model_path is not None,
    )

#: Canonical files every run writes into its fresh output directory.
CANONICAL_OUTPUTS = (
    "physical_inputs.npz",
    "raw_outputs.npz",
    "outputs.npz",
    "result.png",
    "report.json",
)


def add_runtime_arguments(p):
    p.add_argument(
        "--target",
        "--platform",
        dest="target",
        choices=("auto", "x5", "s100", "s100p", "s600"),
        default="auto",
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--agent-score-thres", type=float, default=0.5)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")


def build_parser():
    p = argparse.ArgumentParser(
        description="Run one deterministic planning example with raw IO and full provenance."
    )
    add_runtime_arguments(p)
    p.add_argument(
        "--input-npz", type=Path, default=SAMPLE_DIR / "test_data/reference_inputs.npz"
    )
    p.add_argument("--output", type=Path, default=Path("outputs/diffusiondrive"))
    p.add_argument(
        "--output-npz",
        type=Path,
        help="Optional additional decoded archive; canonical outputs.npz is always retained",
    )
    p.add_argument(
        "--img-save-path",
        "--output-image",
        dest="img_save_path",
        type=Path,
        help="Optional additional visualization",
    )
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
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "input_npz": str(args.input_npz),
                "sdk_loaded": False,
                "downloaded": False,
            },
            indent=2,
        )
    )
    return 0


def resolve_extras(args):
    """Resolve optional additional destinations keyed by their output kind."""

    return {
        key: p.expanduser().resolve()
        for key, p in (("decoded", args.output_npz), ("image", args.img_save_path))
        if p is not None
    }


def validate_extra_destinations(extras) -> None:
    """Additional outputs need image extensions and non-canonical paths."""

    if "image" in extras and extras["image"].suffix.lower() not in (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    ):
        raise ValueError("Additional image requires PNG/JPEG/BMP extension")


def save_planning_evidence(
    output: Path,
    extras: dict,
    *,
    selection,
    runner,
    binding,
    args,
    input_path: Path,
    features,
    details,
    started: str,
    finished: str,
) -> None:
    """Write trajectories, raw IO, visualization and the provenance report."""

    from datetime import datetime, timezone

    import cv2
    import numpy as np

    from utils.py_utils.assets import sha256_file
    from utils.py_utils.runtime_meta import metadata_evidence
    from samples.vision.diffusiondrive.runtime.python.visualization import (
        render_result,
    )

    canvas = render_result(features, details.result, selection.target.upper())
    report = {
        "schema_version": "1.0",
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path.resolve()),
        "model_sha256": sha256_file(selection.model_path),
        "publisher_sha256": selection.asset.sha256,
        "input_npz": str(input_path),
        "input_sha256": sha256_file(input_path),
        "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
        "runtime_metadata": metadata_evidence(binding.metadata),
        "scheduling": {"priority": args.priority, "bpu_cores": args.bpu_cores},
        "agent_score_threshold": args.agent_score_thres,
        "started_utc": started,
        "finished_inference_utc": finished,
        "noise": "fixed caller-supplied tensor; no regeneration",
        "latency": "not measured",
        "actuation": False,
        "additional_outputs": {k: str(p) for k, p in extras.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "physical_inputs.npz", **details.physical)
    np.savez(output / "raw_outputs.npz", **details.raw)
    np.savez(output / "outputs.npz", **details.result)
    if not cv2.imwrite(str(output / "result.png"), canvas):
        raise OSError("Failed to save visualization")
    for key, path in extras.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "decoded":
            shutil.copyfile(output / "outputs.npz", path)
        elif not cv2.imwrite(str(path), canvas):
            raise OSError(f"Failed to save {path}")
    report["output_sha256"] = {
        name: sha256_file(output / name) for name in CANONICAL_OUTPUTS[:-1]
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
