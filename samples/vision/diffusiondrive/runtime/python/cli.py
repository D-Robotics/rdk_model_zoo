# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Planning CLI surface: option declarations, path checks and evidence IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything around selection and delivery lives here — the parser, the
model-free listing/dry-run modes, strict NPZ feature loading, destination
validation, the source display geometry/palette and the canonical
output/report writing. Nothing in this module plans trajectories or runs
inference; the task stages and affine transforms live in
``diffusiondrive.py``.
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

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


# ======================================================================
# Strict NPZ loading and non-overwriting output paths, independent of
# inference.
# ======================================================================

def load_npz(path):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive")
    with archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate NPZ names are ambiguous")
        return {name: archive[name].copy() for name in archive.files}


def load_features(path):
    from samples.vision.diffusiondrive.runtime.python.diffusiondrive import INPUT_SHAPES

    values = load_npz(path)
    if set(values) != set(INPUT_SHAPES):
        raise ValueError("Feature archive requires camera/lidar/status/noise only")
    for name, shape in INPUT_SHAPES.items():
        value = values[name]
        if (
            value.shape != shape
            or value.dtype != np.dtype("float32")
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid logical feature {name}: expected finite float32 {shape}"
            )
    return values


def validate_destinations(output, extras):
    output = Path(output).expanduser().resolve()
    paths = [Path(p).expanduser().resolve() for p in extras]
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    reserved = {
        output / name
        for name in (
            "raw_outputs.npz",
            "physical_inputs.npz",
            "outputs.npz",
            "result.png",
            "report.json",
            "batch-report.json",
        )
    }
    if len(set(paths)) != len(paths):
        raise ValueError("Additional output paths must be distinct")
    for p in paths:
        if p.exists() or p == output or p in output.parents or p in reserved:
            raise ValueError(f"Additional output path exists or conflicts: {p}")
    return output


# ======================================================================
# Source display geometry and palette, separated from inference math.
# ======================================================================

BEV_PIXEL_SIZE = 0.25


BEV_CLASS_NAMES = (
    "background",
    "road",
    "walkway",
    "centerline",
    "static",
    "vehicle",
    "pedestrian",
)


BEV_PALETTE_BGR = np.asarray(
    [
        [255, 255, 255],
        [185, 185, 185],
        [167, 205, 232],
        [0, 215, 255],
        [182, 89, 155],
        [60, 76, 231],
        [219, 152, 52],
    ],
    dtype=np.uint8,
)


def _xy_to_pixel(x: float, y: float) -> tuple[int, int]:
    """Convert ego-local metric coordinates into the 256x256 LiDAR raster.

    Args:
        x: Forward coordinate in meters.
        y: Left coordinate in meters.

    Returns:
        OpenCV pixel coordinate as ``(column, row)``.
    """

    return int(round(y / BEV_PIXEL_SIZE + 128.0)), int(
        round(x / BEV_PIXEL_SIZE + 128.0)
    )


def _agent_polygon(state: np.ndarray) -> np.ndarray:
    """Convert one ``[x, y, heading, length, width]`` state to raster corners.

    Args:
        state: Agent state in ego-local metric coordinates.

    Returns:
        Four OpenCV polygon points as an int32 array.
    """

    x, y, heading, length, width = map(float, state)
    forward = np.array([np.cos(heading), np.sin(heading)]) * length / 2.0
    lateral = np.array([-np.sin(heading), np.cos(heading)]) * width / 2.0
    center = np.array([x, y])
    corners = [
        center + forward + lateral,
        center + forward - lateral,
        center - forward - lateral,
        center - forward + lateral,
    ]
    return np.asarray(
        [_xy_to_pixel(float(point[0]), float(point[1])) for point in corners],
        dtype=np.int32,
    )


def render_result(
    features: Dict[str, np.ndarray],
    result: Dict[str, np.ndarray],
    platform_name: str = "RDK S",
) -> np.ndarray:
    """Render camera, BEV semantic, LiDAR, trajectory, and agent visualization.

    Args:
        features: Original float input feature dictionary.
        result: Post-processed result returned by ``DiffusionDrive.predict``.
        platform_name: Board name shown in the visualization title.

    Returns:
        Owned BGR uint8 canvas; the caller controls saving.
    """
    import cv2

    camera = np.clip(features["camera"][0].transpose(1, 2, 0), 0.0, 1.0)
    camera_bgr = cv2.cvtColor(
        np.rint(camera * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    semantic = BEV_PALETTE_BGR[result["bev_labels"][0]]
    semantic = cv2.rotate(semantic, cv2.ROTATE_180)
    semantic = cv2.resize(semantic, (512, 256), interpolation=cv2.INTER_NEAREST)

    density = np.clip(features["lidar"][0, 0], 0.0, 1.0)
    gray = np.rint(255.0 * (1.0 - density)).astype(np.uint8)
    lidar = np.repeat(gray[..., None], 3, axis=-1)
    cv2.polylines(
        lidar,
        [_agent_polygon(np.array([0.0, 0.0, 0.0, 5.2, 2.0]))],
        True,
        (235, 99, 36),
        2,
    )
    for state in result["agent_states"][0, result["agent_mask"][0]]:
        cv2.polylines(lidar, [_agent_polygon(state)], True, (68, 68, 239), 2)
    trajectory = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), result["trajectory"][0, :, :2]], axis=0
    )
    points = np.asarray(
        [_xy_to_pixel(float(x), float(y)) for x, y in trajectory], dtype=np.int32
    )
    cv2.polylines(lidar, [points], False, (0, 122, 255), 3)
    for point in points[1:]:
        cv2.circle(lidar, tuple(point), 3, (0, 122, 255), -1)
    lidar = lidar[128:256]
    lidar = cv2.rotate(lidar, cv2.ROTATE_180)
    lidar = cv2.resize(lidar, (512, 256), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((648, 1024, 3), (36, 29, 25), dtype=np.uint8)
    canvas[32:288] = camera_bgr
    canvas[320:576, :512] = semantic
    canvas[320:576, 512:] = lidar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"{platform_name} DiffusionDrive - camera input",
        (12, 21),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Predicted BEV semantics",
        (12, 309),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    count = int(result["agent_mask"].sum())
    cv2.putText(
        canvas,
        f"LiDAR + trajectory (orange) + agents (red, count={count})",
        (524, 309),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    x_offset = 12
    for class_name, color in zip(BEV_CLASS_NAMES, BEV_PALETTE_BGR):
        cv2.rectangle(
            canvas,
            (x_offset, 591),
            (x_offset + 13, 604),
            tuple(int(value) for value in color),
            -1,
        )
        cv2.putText(
            canvas,
            class_name,
            (x_offset + 18, 603),
            font,
            0.38,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        x_offset += 139
    cv2.putText(
        canvas,
        "Forward is up; ego vehicle is blue.",
        (12, 634),
        font,
        0.45,
        (210, 205, 195),
        1,
        cv2.LINE_AA,
    )
    return canvas


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
