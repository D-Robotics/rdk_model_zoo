# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit source sample selections, preserving manifest identities and URLs.

These records route to the shared task implementation, not to a substitute model.
A source filename containing ``nashe`` under ``s600/`` stays an S600 publication
reference; it does not certify its unobserved physical HBM march or output ABI.
"""

from dataclasses import dataclass
from pathlib import Path

from samples._shared.assets import Asset, resolve_asset
from yolo_assets import UnsupportedAssetError


@dataclass(frozen=True)
class StandaloneSelection:
    asset: Asset
    target: str
    family: str
    task: str
    size: str
    nms_thres: float


def standalone_selections(profile):
    """Return the finite ten source records, filtered by exact target identity."""
    if profile.family != "s":
        return ()
    rows = []
    if profile.key in ("s100", "s600"):
        for sample, task, nms in [
            ("yolo11", "detect", 0.45),
            ("yolo11_pose", "pose", 0.7),
            ("yolo11_seg", "seg", 0.7),
        ]:
            name = f"{profile.key}/yolo11n_{task}_nashe_640x640_nv12.hbm"
            rows.append((sample, name, "yolo11", task, "n", nms))
    if profile.key == "s100":
        for size in "nslx":
            rows.append(
                (
                    "yolov13_imoonlab",
                    f"s100/yolo13{size}_detect_nashe_640x640_nv12.hbm",
                    "yolov13",
                    "detect",
                    size,
                    0.45,
                )
            )
    try:
        return tuple(
            StandaloneSelection(
                resolve_asset(f"s:{sample}:{filename}"),
                profile.key,
                family,
                task,
                size,
                nms,
            )
            for sample, filename, family, task, size, nms in rows
        )
    except ValueError as exc:
        raise UnsupportedAssetError(str(exc)) from exc


def select_standalone(profile, reference):
    """Match an exact published source identity without target fallback."""
    matches = [
        s for s in standalone_selections(profile) if s.asset.reference == reference
    ]
    if len(matches) != 1:
        raise UnsupportedAssetError(
            "Source asset is not registered for the selected target."
        )
    return matches[0]


def asset_local_path(model_root, profile, asset):
    """Keep standalone files distinct from the family download destinations."""
    if asset.sample_id in ("yolo11", "yolo11_pose", "yolo11_seg", "yolov13_imoonlab"):
        select_standalone(profile, asset.reference)
        return str(Path(model_root) / "standalone" / asset.sample_id / asset.filename)
    from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
        model_directory,
    )

    return str(
        Path(model_directory(str(model_root), profile)) / Path(asset.filename).name
    )
