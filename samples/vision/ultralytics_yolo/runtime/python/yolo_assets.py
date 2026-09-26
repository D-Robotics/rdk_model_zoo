# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve Ultralytics YOLO model assets for a selected platform.

Asset filename tokens are not uniform across platforms or families. S100/S100P
v8/v11 classifiers retain `640x640` manifest identities with `224x224` public
URLs. Tokens are compatibility identifiers; actual input geometry comes from
runtime metadata. The suffix encodes the march and S artifacts use per-march
directories. This module derives names and resolves exact active manifest entries.

Nothing here downloads anything; Manifest resolution is separated from I/O so the
downloader can run as a dry run on a host without board runtime.

Typical Usage:
    >>> from yolo_platform import resolve_platform
    >>> from yolo_assets import model_url
    >>> model_url(resolve_platform("x5"), "yolo11", "detect").endswith("yolo11n_detect_bayese_640x640_nv12.bin")
    True
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from yolo_platform import (
    PlatformProfile,
    model_directory,
)

#: Task handled by the NMS-free YOLOv10 detection path.
TASK_DETECT = "detect"

#: Instance segmentation task.
TASK_SEG = "seg"

#: Pose estimation task.
TASK_POSE = "pose"

#: Image classification task.
TASK_CLS = "cls"

#: Every task name accepted on the command line.
SUPPORTED_TASKS: Tuple[str, ...] = (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS, "obb")

#: Default family used when none is requested.
DEFAULT_FAMILY = "yolo11"

#: Default task used when none is requested.
DEFAULT_TASK = TASK_DETECT

#: Mapping of the task name to the token embedded in published filenames.
TASK_TOKENS: Dict[str, str] = {
    "obb": "obb",
    TASK_DETECT: "detect",
    TASK_SEG: "seg",
    TASK_POSE: "pose",
    TASK_CLS: "cls",
}


class UnsupportedAssetError(ValueError):
    """Raised when a platform does not publish the requested model asset."""


@dataclass(frozen=True)
class FamilySpec:
    """Describe the assets published for one model family on one platform.

    Attributes:
        family: Canonical family name, for example `"yolov8"`.
        token: Prefix of the published filename, for example `"yolov5nu"`.
        tasks: Tasks the platform publishes for this family.
        sizes: Model scales the platform publishes for this family.
        default_size: Scale used when none is requested.
        task_sizes: Per-task scale restrictions that narrow `sizes`.
        size_platforms: Per-scale platform restrictions. A scale listed here is
            only published on the named platform keys.
        notes: Human-readable caveats surfaced in documentation.
    """

    family: str
    token: str
    tasks: Tuple[str, ...]
    sizes: Tuple[str, ...]
    default_size: str
    task_sizes: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    size_platforms: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    notes: str = ""


def _spec(family: str,
          tasks: Tuple[str, ...],
          sizes: Tuple[str, ...],
          default_size: str,
          task_sizes: Optional[Dict[str, Tuple[str, ...]]] = None,
          size_platforms: Optional[Dict[str, Tuple[str, ...]]] = None,
          notes: str = "") -> FamilySpec:
    """Build a `FamilySpec`, deriving the published filename token.

    Args:
        family: Canonical family name.
        tasks: Tasks the platform publishes for this family.
        sizes: Model scales the platform publishes for this family.
        default_size: Scale used when none is requested.
        task_sizes: Optional per-task scale restrictions.
        size_platforms: Optional per-scale platform restrictions.
        notes: Optional human-readable caveat.

    Returns:
        The constructed `FamilySpec`.
    """
    # YOLOv5u publishes a `u` suffix directly after the scale letter, so it is
    # part of the filename token rather than a separate segment.
    token = "yolov5" if family == "yolov5u" else family
    return FamilySpec(
        family=family,
        token=token,
        tasks=tasks,
        sizes=sizes,
        default_size=default_size,
        task_sizes=task_sizes or {},
        size_platforms=size_platforms or {},
        notes=notes,
    )


_COMMON_SMALL_TO_LARGE = ("n", "s", "m", "l", "x")
_YOLOV9_SIZES = ("t", "s", "m", "c", "e")

#: Families published on RDK X5.
X5_FAMILIES: Dict[str, FamilySpec] = {
    "yolo26": _spec("yolo26", SUPPORTED_TASKS, _COMMON_SMALL_TO_LARGE, "n"),
    "yolov5u": _spec("yolov5u", (TASK_DETECT,), _COMMON_SMALL_TO_LARGE, "n"),
    "yolov8": _spec("yolov8", (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS),
                    _COMMON_SMALL_TO_LARGE, "n"),
    "yolov9": _spec("yolov9", (TASK_DETECT, TASK_SEG), _YOLOV9_SIZES, "t",
                    task_sizes={TASK_SEG: ("c", "e")}),
    "yolov10": _spec("yolov10", (TASK_DETECT,), ("n", "s", "m", "b", "l", "x"), "n",
                     notes="X5 retains DFL + NMS."),
    "yolo11": _spec("yolo11", (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS),
                    _COMMON_SMALL_TO_LARGE, "n"),
    "yolo12": _spec("yolo12", (TASK_DETECT,), _COMMON_SMALL_TO_LARGE, "n"),
    "yolov13": _spec("yolov13", (TASK_DETECT,), ("n", "s", "l", "x"), "n"),
}

#: Families published on the RDK S series.
S_FAMILIES: Dict[str, FamilySpec] = {
    "yolo26": _spec("yolo26", SUPPORTED_TASKS, _COMMON_SMALL_TO_LARGE, "n"),
    "yolov5u": _spec("yolov5u", (TASK_DETECT,), _COMMON_SMALL_TO_LARGE, "n"),
    "yolov8": _spec("yolov8", (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS),
                    _COMMON_SMALL_TO_LARGE, "n"),
    "yolov9": _spec("yolov9", (TASK_DETECT, TASK_SEG), _YOLOV9_SIZES, "s",
                    task_sizes={TASK_SEG: ("c", "e")},
                    size_platforms={"t": ("s100", "s100p")},
                    notes="YOLOv9 segmentation is published on S100/S100P only."),
    "yolov10": _spec("yolov10", (TASK_DETECT,), ("n", "s", "m", "b", "l", "x"), "n",
                     notes="YOLOv10 is NMS-free."),
    "yolo11": _spec("yolo11", (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS),
                    _COMMON_SMALL_TO_LARGE, "n"),
    "yolo12": _spec("yolo12", (TASK_DETECT,), _COMMON_SMALL_TO_LARGE, "n"),
}

#: Compatibility filename tokens, not authoritative input geometry. The active
#: S100/S100P manifest keeps legacy 640 identities but its URLs use 224 names;
#: both URL spellings were available in the 2026-09-26 HEAD audit. S600 and
#: YOLO26 identities already use 224. Preserve exact manifest references and
#: derive execution geometry from runtime metadata, never these tokens.
CLASSIFICATION_LOW_RESOLUTION_FAMILIES = frozenset({"yolo26"})
CLASSIFICATION_LOW_RESOLUTION_TARGETS = frozenset({"s600"})

#: Input resolution used by every non-classification artifact on both
#: platforms.
DEFAULT_RESOLUTION = "640x640"


def family_registry(profile: PlatformProfile) -> Dict[str, FamilySpec]:
    """Return the family registry of a platform.

    Args:
        profile: Platform whose published families are requested.

    Returns:
        A mapping of canonical family name to `FamilySpec`.
    """
    if profile.family == "x5":
        return X5_FAMILIES
    if profile.key == "s100":
        return {**S_FAMILIES, "yolov13": FamilySpec(
            "yolov13", "yolo13", (TASK_DETECT,), ("n", "s", "l", "x"), "n",
            notes="iMoonLab source artifacts, S100 only; board verification separate.")}
    return S_FAMILIES


def available_families(profile: PlatformProfile) -> Tuple[str, ...]:
    """Return the family names a platform publishes.

    Args:
        profile: Platform whose families are requested.

    Returns:
        A tuple of canonical family names, in display order.
    """
    return tuple(family_registry(profile))


def classification_resolution(profile: PlatformProfile, family: str) -> str:
    """Return the compatibility filename resolution token of one family.

    Args:
        profile: Platform that publishes the classification artifacts.
        family: Model family name.

    Returns:
        A resolution string such as `"640x640"` or `"224x224"`.
    """
    if family in CLASSIFICATION_LOW_RESOLUTION_FAMILIES or profile.key in CLASSIFICATION_LOW_RESOLUTION_TARGETS:
        return "224x224"
    return DEFAULT_RESOLUTION


def asset_resolution(profile: PlatformProfile, family: str, task: str) -> str:
    """Return the filename resolution token, not observed runtime geometry.

    Args:
        profile: Platform that publishes the asset.
        family: Model family name.
        task: Task name.

    Returns:
        A resolution string embedded in the published filename.
    """
    if task == TASK_CLS:
        return classification_resolution(profile, family)
    return DEFAULT_RESOLUTION


def resolve_size(profile: PlatformProfile,
                 family: str,
                 task: str,
                 size: Optional[str] = None) -> str:
    """Validate and default the model scale of an asset.

    Args:
        profile: Platform that publishes the asset.
        family: Model family name.
        task: Task name.
        size: Requested model scale, or `None` to use the family default.

    Returns:
        The validated model scale.

    Raises:
        UnsupportedAssetError: If the family, task or scale is not published
            by the platform.
    """
    spec = _require_family(profile, family)
    if task not in spec.tasks:
        raise UnsupportedAssetError(
            f"{profile.key} does not publish {family} for task {task!r}. "
            f"Published tasks: {', '.join(spec.tasks)}.")
    if profile.key == "s600" and family == "yolov9" and task == TASK_SEG:
        raise UnsupportedAssetError("S600 publishes no YOLOv9 segmentation asset.")
    allowed = spec.task_sizes.get(task, spec.sizes)
    chosen = (size or (spec.default_size if spec.default_size in allowed else allowed[0])).strip().lower()
    if chosen not in allowed:
        raise UnsupportedAssetError(
            f"{profile.key} does not publish {family} size {chosen!r} for task "
            f"{task!r}. Published sizes: {', '.join(allowed)}.")
    restricted = spec.size_platforms.get(chosen)
    if restricted is not None and profile.key not in restricted:
        raise UnsupportedAssetError(
            f"{profile.key} does not publish {family} size {chosen!r}. "
            f"Published on: {', '.join(restricted)}.")
    return chosen


def _require_family(profile: PlatformProfile, family: str) -> FamilySpec:
    """Look up a family in a platform registry.

    Args:
        profile: Platform that publishes the family.
        family: Requested family name.

    Returns:
        The matching `FamilySpec`.

    Raises:
        UnsupportedAssetError: If the platform publishes no such family.
    """
    registry = family_registry(profile)
    spec = registry.get((family or "").strip().lower())
    if spec is None:
        raise UnsupportedAssetError(
            f"{profile.key} publishes no {family!r} family. Published families: "
            f"{', '.join(registry)}.")
    return spec


def model_filename(profile: PlatformProfile,
                   family: str,
                   task: str,
                   size: Optional[str] = None) -> str:
    """Build the existing manifest filename, including compatibility identities.

    Args:
        profile: Platform that publishes the asset.
        family: Model family name.
        task: Task name.
        size: Model scale, or `None` to use the family default.

    Returns:
        The published filename, for example
        `yolo11n_detect_bayese_640x640_nv12.bin`.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    spec = _require_family(profile, family)
    scale = resolve_size(profile, family, task, size)
    token = f"{spec.token}{scale}u" if spec.family == "yolov5u" else f"{spec.token}{scale}"
    resolution = asset_resolution(profile, family, task)
    return (f"{token}_{TASK_TOKENS[task]}_{profile.model_suffix}_"
            f"{resolution}_nv12{profile.model_format}")


def model_url(profile: PlatformProfile,
              family: str,
              task: str,
              size: Optional[str] = None) -> str:
    """Build the download URL of a model asset.

    Args:
        profile: Platform that publishes the asset.
        family: Model family name.
        task: Task name.
        size: Model scale, or `None` to use the family default.

    Returns:
        The absolute download URL.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    asset = manifest_asset(profile, family, task, size)
    if not asset.url:
        raise UnsupportedAssetError(f'No download URL recorded for {asset.reference}.')
    return asset.url


def manifest_asset(profile: PlatformProfile, family: str, task: str,
                   size: Optional[str] = None):
    """Read the published record selected by the existing finite family policy.

    Filenames remain compatibility selections; URLs and publisher hashes have
    one authority in the existing platform manifest.
    """
    from samples._shared.assets import resolve_asset
    filename = model_filename(profile, family, task, size)
    if profile.family == "s" and family == "yolov13":
        return resolve_asset(f's:yolov13_imoonlab:s100/{filename}')
    if profile.model_subdir:
        filename = profile.model_subdir + '/' + filename
    sample = 'ultralytics_yolo26' if family == 'yolo26' else 'ultralytics_yolo'
    try:
        return resolve_asset(f'{profile.family}:{sample}:{filename}')
    except ValueError as exc:
        raise UnsupportedAssetError(str(exc)) from exc


def model_path(model_root: str,
               profile: PlatformProfile,
               family: str,
               task: str,
               size: Optional[str] = None) -> str:
    """Build the local filesystem path of a model asset.

    Args:
        model_root: The sample `model/` directory.
        profile: Platform that publishes the asset.
        family: Model family name.
        task: Task name.
        size: Model scale, or `None` to use the family default.

    Returns:
        The absolute or relative path the asset is stored at.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    from standalone_assets import asset_local_path
    return asset_local_path(model_root, profile, manifest_asset(profile, family, task, size))


def is_nms_free(profile: PlatformProfile, family: str) -> bool:
    """Report whether a family uses the NMS-free detection head.

    Args:
        profile: Platform that publishes the family.
        family: Model family name.

    Returns:
        True for S-series YOLOv10 only; X5 retains its NMS decoder.

    Raises:
        UnsupportedAssetError: If the platform publishes no such family.
    """
    return profile.family == "s" and _require_family(profile, family).family == "yolov10"


def family_from_filename(profile: PlatformProfile,
                         filename: str) -> Optional[str]:
    """Identify the model family a published asset filename belongs to.

    Published asset names start with the family token followed by the model
    scale, for example `yolo11n_detect_bayese_640x640_nv12.bin`. The token is
    matched against the registry of the selected platform, longest token first
    so that `yolov10` is not mistaken for `yolov1`.

    Args:
        profile: Platform whose published names are matched.
        filename: File name, with or without a directory part.

    Returns:
        The canonical family name, or `None` when the name matches no family
        the platform publishes.
    """
    import os

    base = os.path.basename(filename or "").lower()
    for family, spec in sorted(family_registry(profile).items(),
                               key=lambda item: len(item[1].token),
                               reverse=True):
        if base.startswith(spec.token):
            return family
    return None


def family_listing(profile: PlatformProfile) -> List[Dict[str, object]]:
    """Summarise every asset a platform publishes.

    Args:
        profile: Platform whose assets are summarised.

    Returns:
        A list of dictionaries, one per published family, containing the
        family name, published tasks, published per-task sizes, the default
        size, and any caveat. Used by `--list-models` and by the tests.
    """
    listing: List[Dict[str, object]] = []
    for name, spec in family_registry(profile).items():
        tasks = {}
        for task in spec.tasks:
            sizes = []
            for size in spec.task_sizes.get(task, spec.sizes):
                try:
                    resolve_size(profile, name, task, size)
                except UnsupportedAssetError:
                    continue
                sizes.append(size)
            if sizes:
                tasks[task] = sizes
        listing.append({
            "family": name,
            "tasks": tasks,
            "default_size": spec.default_size,
            "default_task": TASK_DETECT if TASK_DETECT in spec.tasks else spec.tasks[0],
            "notes": spec.notes,
        })
    return listing
