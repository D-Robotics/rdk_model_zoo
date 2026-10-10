# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""LaneNet CLI surface: option declarations, path checks and evidence IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything presentational — the parser, the model-free listing/dry-run modes,
destination validation, the display renderers and the canonical output/report
writing — lives here. Nothing in this module runs inference or clusters lanes.
"""

import argparse
import json
from pathlib import Path
import sys

from dataclasses import dataclass
from typing import Optional

from utils.py_utils.assets import Asset, list_assets


class BindingError(ValueError):
    """A LaneNet selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "s:lanenet:s100/lanenet256x512.hbm"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (``s100`` for LaneNet).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def _asset() -> Asset:
    rows = list_assets("s", "lanenet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: Optional[str] = None) -> tuple:
    """List the published S100 asset without board or SDK access.

    Args:
        target: Concrete target filter; other targets publish nothing.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        BindingError: The target is unknown.
    """
    if target in (None, "auto", "s100"):
        return (_asset(),)
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}.")


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    model_path: "str | Path | None" = None,
) -> ModelSelection:
    """Resolve LaneNet's exact published asset identity without loading it.

    Args:
        target: ``auto``/``s100``; other targets publish nothing.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        BindingError: The target, asset, or path combination is invalid.
    """
    key = (target or "auto").lower()
    if key == "auto":
        key = "s100"
    if key != "s100":
        raise BindingError("LaneNet is published only for target s100.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError(
            "An external model path requires --asset-id s:lanenet:s100/lanenet256x512.hbm."
        )
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=(
            Path(model_path).expanduser()
            if model_path
            else SAMPLE_DIR / "model" / asset.filename
        ),
        explicit_model_path=model_path is not None,
    )

#: Canonical files every run writes into its fresh output directory.
CANONICAL_OUTPUTS = (
    "raw_outputs.npz",
    "embedding.npy",
    "binary.npy",
    "instance_pred.png",
    "binary_pred.png",
    "report.json",
)


def build_parser():
    """Build the CLI parser with target, asset and input options."""

    p = argparse.ArgumentParser(
        description="LaneNet embeddings and binary labels; S100 published asset only"
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--test-img", type=Path, default=SAMPLE_DIR / "test_data/lane.jpg")
    p.add_argument("--output", type=Path, default=Path("outputs/lanenet"))
    p.add_argument(
        "--instance-save-path",
        type=Path,
        help="Optional additional embedding display, not clustered lane IDs",
    )
    p.add_argument(
        "--binary-save-path", type=Path, help="Optional additional binary display"
    )
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
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


def run_dry_run(selection) -> int:
    """Print the resolved selection without loading a model or SDK."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "sdk_loaded": False,
                "downloaded": False,
                "clustering_performed": False,
            },
            indent=2,
        )
    )
    return 0


def validate_display_destinations(output: Path, extras: dict) -> None:
    """Additional displays must be new, distinct and outside canonical names."""

    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    reserved = {(output / name).resolve() for name in CANONICAL_OUTPUTS}
    destinations = [p.resolve() for p in extras.values()]
    if len(set(destinations)) != len(destinations) or any(
        p in reserved for p in destinations
    ):
        raise ValueError(
            "Additional displays must have distinct paths outside canonical output filenames"
        )
    for path in extras.values():
        if path.exists():
            raise FileExistsError(f"Use a new additional image path: {path}")


def embedding_image(embedding):
    """Finite float32 CHW3 → HWC uint8 clip/round; channel order is preserved.

    This unifies Python with native clipping/saturating rounding and deliberately
    replaces source Python truncation/wrap. It is a display, not instance labels.
    """
    import numpy as np

    value = np.asarray(embedding)
    if (
        value.ndim != 3
        or value.shape[0] != 3
        or value.size == 0
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected finite float32 CHW embedding with three channels")
    return np.rint(np.clip(value, 0, 1) * 255).transpose(1, 2, 0).astype(np.uint8)


def binary_image(binary):
    """2D discrete labels → 0/255 display; reject invalid labels."""
    import numpy as np

    value = np.asarray(binary)
    if (
        value.ndim != 2
        or value.size == 0
        or value.dtype.kind not in "iu"
        or not np.isin(value, (0, 1)).all()
    ):
        raise ValueError("Expected 2D integer labels 0/1")
    return value.astype(np.uint8) * 255


def read_bgr_image(path: Path):
    """Read one BGR image; decode failures name the exact path."""

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_lane_evidence(
    output: Path,
    extras: dict,
    *,
    selection,
    runner,
    binding,
    args,
    image_path: Path,
    image,
    details,
) -> None:
    """Write raw outputs, owned arrays, displays and the provenance report."""

    import cv2
    import numpy as np

    from utils.py_utils.assets import sha256_file
    from utils.py_utils.runtime_meta import metadata_evidence

    result = details.result
    displays = {
        "instance": embedding_image(result.embedding),
        "binary": binary_image(result.binary),
    }
    # Fixed archive keys avoid arbitrary SDK names colliding with np.savez
    # keyword parameters. The complete name-to-key relation stays explicit.
    keys = {name: f"output_{index}" for index, name in enumerate(binding.metadata.output_names)}
    report = {
        "schema_version": "1.0",
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path),
        "model_sha256": sha256_file(selection.model_path),
        "publisher_sha256": selection.asset.sha256,
        "input": str(image_path),
        "input_sha256": sha256_file(image_path),
        "input_shape": list(image.shape),
        "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
        "runtime_metadata": metadata_evidence(binding.metadata),
        "raw_tensor_keys": keys,
        "embedding_shape": list(result.embedding.shape),
        "binary_shape": list(result.binary.shape),
        "clustering_performed": False,
        "output_grid": "model 256x512; not original image size",
        "embedding_display": "clip to [0,1], multiply255, round ties-to-even, preserve channel order",
        "binary_display": "validated 0/1 labels times255",
        "priority": args.priority,
        "bpu_cores": args.bpu_cores,
        "additional_images": {name: str(path) for name, path in extras.items()},
        "latency": "not measured",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez(
        output / "raw_outputs.npz",
        **{keys[name]: value for name, value in details.raw.items()},
    )
    np.save(output / "embedding.npy", result.embedding)
    np.save(output / "binary.npy", result.binary)
    writes = [
        (output / "instance_pred.png", displays["instance"]),
        (output / "binary_pred.png", displays["binary"]),
    ] + [(path, displays[name]) for name, path in extras.items()]
    for path, value in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value):
            raise OSError(f"Failed to save image: {path}")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
