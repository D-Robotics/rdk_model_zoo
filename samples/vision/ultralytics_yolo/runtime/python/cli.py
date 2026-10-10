# Copyright (c) 2025-2026 D-Robotics Corporation
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
"""Ultralytics YOLO command surface: platforms, published assets and dispatch.

``cli.py`` concentrates everything the entry needs around selection and
presentation: the platform profiles (X5 packed-NV12 ``.bin`` vs the S-series
split-NV12 ``.hbm`` toolchains), the published family/task/size registry with
manifest-derived filenames and URLs, the task dispatch that maps one parsed
argument set to its task model class and configuration, the documented
resize/NMS defaults, the parser, the model-free ``--list-models``/``--dry-run``
modes, label loading and result rendering.  Nothing here loads the board SDK
or imports a task module at import time; task stages live in
``detect.py``/``segment.py``/``pose.py``/``obb.py``/``classify.py`` and the
physical binding/transport in ``backend.py``.  ``main.py`` owns the sys.path
setup that makes these modules importable.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np  # noqa: F401 - re-exported surface used by callers

from utils.py_utils.assets import Asset, list_assets

# ====================================================================
# Deployment platforms.
# ====================================================================

SOC_NAME_PATH = "/sys/class/boardinfo/soc_name"


BOARD_TYPE_PATH = "/sys/class/boardinfo/board_type"


INPUT_PROTOCOL_PACKED = "packed"


INPUT_PROTOCOL_SPLIT = "split"


MODEL_FORMAT_BIN = ".bin"


MODEL_FORMAT_HBM = ".hbm"


class UnsupportedPlatformError(ValueError):
    """Raised when a requested or detected platform is not supported."""


@dataclass(frozen=True)
class PlatformProfile:
    """Describe one deployment platform.

    Attributes:
        key: Canonical platform name accepted on the command line.
        family: Broad platform family, `"x5"` or `"s"`.
        soc_names: SoC names that map onto this platform during host detection.
        march: BPU architecture name used by the compiler toolchain.
        model_suffix: Suffix embedded in published model filenames.
        model_format: Published artifact file extension.
        model_subdir: Sub-directory that holds the artifacts, relative to the
            sample `model/` directory. Empty when artifacts are stored flat.
        url_soc_dir: SoC directory segment of the download URL.
        url_base: Toolchain image segment of the download URL.
        input_protocol: NV12 tensor protocol, `"packed"` or `"split"`.
        default_input_hw: Input resolution assumed when the runtime reports a
            tensor shape that does not encode height and width separately.
        detect_nms_thres: Default IoU threshold for the detection tasks. The
            two published sample trees document different defaults, so the
            platform supplies it instead of the wrapper hard-coding one.
        cls_resize_type: Classification CLI resize default: X5 letterbox (1), S stretch (0).
        supports_cpp: Whether this sample ships a C++ runtime for the platform.
    """

    key: str
    family: str
    soc_names: Tuple[str, ...]
    march: str
    model_suffix: str
    model_format: str
    model_subdir: str
    url_soc_dir: str
    url_base: str
    input_protocol: str
    default_input_hw: Tuple[int, int]
    detect_nms_thres: float
    cls_resize_type: int
    supports_cpp: bool

    @property
    def nms_thres(self) -> float:
        """Return the documented default IoU threshold for detection."""
        return self.detect_nms_thres

    @property
    def is_packed_input(self) -> bool:
        """Return True when NV12 is bound as a single packed tensor."""
        return self.input_protocol == INPUT_PROTOCOL_PACKED


_ARCHIVE_ROOT = "https://archive.d-robotics.cc/downloads/rdk_model_zoo"


PLATFORMS: Dict[str, PlatformProfile] = {
    "x5": PlatformProfile(
        key="x5",
        family="x5",
        soc_names=("x5",),
        march="bayes-e",
        model_suffix="bayese",
        model_format=MODEL_FORMAT_BIN,
        model_subdir="",
        url_soc_dir="rdk_x5",
        url_base="ultralytics_YOLO",
        input_protocol=INPUT_PROTOCOL_PACKED,
        default_input_hw=(640, 640),
        detect_nms_thres=0.70,
        cls_resize_type=1,
        supports_cpp=True,
    ),
    "s100": PlatformProfile(
        key="s100",
        family="s",
        soc_names=("s100",),
        march="nash-e",
        model_suffix="nashe",
        model_format=MODEL_FORMAT_HBM,
        model_subdir="nash-e",
        url_soc_dir="rdk_s100",
        url_base="Ultralytics_YOLO_OE_3.7.0",
        input_protocol=INPUT_PROTOCOL_SPLIT,
        default_input_hw=(640, 640),
        detect_nms_thres=0.45,
        cls_resize_type=0,
        supports_cpp=False,
    ),
    "s100p": PlatformProfile(
        key="s100p",
        family="s",
        soc_names=("s100p",),
        march="nash-m",
        model_suffix="nashm",
        model_format=MODEL_FORMAT_HBM,
        model_subdir="nash-m",
        url_soc_dir="rdk_s100",
        url_base="Ultralytics_YOLO_OE_3.7.0",
        input_protocol=INPUT_PROTOCOL_SPLIT,
        default_input_hw=(640, 640),
        detect_nms_thres=0.45,
        cls_resize_type=0,
        supports_cpp=False,
    ),
    "s600": PlatformProfile(
        key="s600",
        family="s",
        soc_names=("s600",),
        march="nash-p",
        model_suffix="nashp",
        model_format=MODEL_FORMAT_HBM,
        model_subdir="nash-p",
        url_soc_dir="rdk_s600",
        url_base="Ultralytics_YOLO_OE_3.7.0",
        input_protocol=INPUT_PROTOCOL_SPLIT,
        default_input_hw=(640, 640),
        detect_nms_thres=0.45,
        cls_resize_type=0,
        supports_cpp=False,
    ),
}


def available_platforms() -> Tuple[str, ...]:
    """Return the canonical platform names, in display order.

    Returns:
        A tuple of accepted `--platform` values.
    """
    return tuple(PLATFORMS)


def model_base_url(profile: PlatformProfile) -> str:
    """Build the download base URL of a platform.

    Args:
        profile: Platform whose artifact directory is requested.

    Returns:
        The base URL that published model filenames are appended to.
    """
    if profile.family == "x5":
        return "/".join((_ARCHIVE_ROOT, profile.url_soc_dir, profile.url_base))
    return "/".join((
        _ARCHIVE_ROOT, profile.url_soc_dir, profile.url_base, profile.march))


def read_board_info(path: str) -> Optional[str]:
    """Read a single-line board information file.

    Args:
        path: Absolute path of the board information file.

    Returns:
        The stripped file contents, or `None` when the file is absent or
        empty. A missing file is a normal condition on a development host.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def detect_host_platform() -> Optional[str]:
    """Detect the platform of the board this process runs on.

    The SoC name is authoritative. RDK S boards additionally publish a board
    type that distinguishes the `p` variants when the SoC name alone does not.

    Returns:
        A canonical platform name, or `None` when no supported board is
        detected. An unknown SoC returns `None` and never a fallback.
    """
    from utils.py_utils.platforms import detect_target
    return detect_target()


def match_platform(soc_name: str, board_type: Optional[str] = None) -> Optional[str]:
    """Map a reported SoC name onto a supported platform.

    Args:
        soc_name: Lower-cased SoC name as reported by the board.
        board_type: Optional board variant string. It is only consulted when
            the SoC name does not already identify a platform, and it only
            ever selects the `p` (performance) variant of the same SoC.

    Returns:
        A canonical platform name, or `None` when the SoC is not supported.
    """
    from utils.py_utils.platforms import match_target
    return match_target(soc_name, board_type)


def resolve_platform(platform: Optional[str] = None,
                     soc_name: Optional[str] = None,
                     board_type: Optional[str] = None) -> PlatformProfile:
    """Select a platform profile, preferring an explicit request.

    Args:
        platform: Explicit platform name. When given it is validated and
            returned as-is; host information is not consulted.
        soc_name: SoC name to use instead of reading the board. Only used when
            `platform` is not given.
        board_type: Board variant used to refine `soc_name`.

    Returns:
        The selected `PlatformProfile`.

    Raises:
        UnsupportedPlatformError: If the explicit platform is unknown, or if
            no explicit platform was given and the host cannot be identified.
    """
    if platform is not None:
        key = platform.strip().lower()
        profile = PLATFORMS.get(key)
        if profile is None:
            raise UnsupportedPlatformError(
                f"Unsupported platform {platform!r}. Supported platforms: "
                f"{', '.join(available_platforms())}.")
        return profile

    if soc_name is not None:
        detected = match_platform(soc_name, board_type)
    else:
        detected = detect_host_platform()
    if detected is None:
        reported = read_board_info(SOC_NAME_PATH)
        detail = f"detected SoC {reported!r}" if reported else "no board detected"
        raise UnsupportedPlatformError(
            f"Cannot select a platform automatically ({detail}). Pass "
            f"--platform explicitly. Supported platforms: "
            f"{', '.join(available_platforms())}.")
    return PLATFORMS[detected]


def model_directory(model_root: str, profile: PlatformProfile) -> str:
    """Return the directory that holds a platform's model artifacts.

    Args:
        model_root: The sample `model/` directory.
        profile: Platform whose artifacts are requested.

    Returns:
        The artifact directory. Platforms that store artifacts flat return
        `model_root` unchanged.
    """
    if not profile.model_subdir:
        return model_root
    return os.path.join(model_root, profile.model_subdir)

# ====================================================================
# Published asset registry: families, tasks, sizes, filenames and URLs.
# ====================================================================

TASK_DETECT = "detect"


TASK_SEG = "seg"


TASK_POSE = "pose"


TASK_CLS = "cls"


SUPPORTED_TASKS: Tuple[str, ...] = (TASK_DETECT, TASK_SEG, TASK_POSE, TASK_CLS, "obb")


DEFAULT_FAMILY = "yolo11"


DEFAULT_TASK = TASK_DETECT


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


CLASSIFICATION_LOW_RESOLUTION_FAMILIES = frozenset({"yolo26"})


CLASSIFICATION_LOW_RESOLUTION_TARGETS = frozenset({"s600"})


DEFAULT_RESOLUTION = "640x640"


def family_registry(profile: PlatformProfile) -> Dict[str, FamilySpec]:
    """Return the family registry of a platform.

    Args:
        profile: Platform whose published families are requested.

    Returns:
        A mapping of canonical family name to `FamilySpec`.
    """
    return X5_FAMILIES if profile.family == "x5" else S_FAMILIES


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
    from utils.py_utils.assets import resolve_asset
    filename = model_filename(profile, family, task, size)
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
    import os

    filename = model_filename(profile, family, task, size)
    return os.path.join(model_directory(model_root, profile), filename)


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

# ====================================================================
# Task dispatch: one (Model, Config) pair per family/task, lazily.
# ====================================================================

#: Alias for :func:`model_path` used by plan resolution.
resolve_model_path = model_path


def get_task_types(profile, family, task):
    """Return the task model class and its config class for one family/task.

    Args:
        profile: Platform profile naming the published families.
        family: Model family, for example ``yolo11``, ``yolov10`` or ``yolo26``.
        task: Task name: ``detect``, ``cls``, ``seg``, ``pose`` or ``obb``.

    Returns:
        Tuple of the task model class and its configuration dataclass.

    Raises:
        UnsupportedAssetError: If the family is unknown or does not publish
            the requested task.
    """
    spec = family_registry(profile).get(family)
    if spec is None or task not in spec.tasks:
        raise UnsupportedAssetError(f'{profile.key}/{family} does not support task {task}.')
    if task == 'detect':
        if is_nms_free(profile, family):
            from samples.vision.ultralytics_yolo.runtime.python.detect import YoloV10Detect, YoloV10DetectConfig
            return YoloV10Detect, YoloV10DetectConfig
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.detect import YOLO26Detect, YOLO26DetectConfig
            return YOLO26Detect, YOLO26DetectConfig
        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect, YoloDetectConfig
        return YoloDetect, YoloDetectConfig
    if task == 'cls':
        from samples.vision.ultralytics_yolo.runtime.python.classify import YoloCls, YoloClsConfig
        return YoloCls, YoloClsConfig
    if task == 'seg':
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.segment import YOLO26Seg, YOLO26SegConfig
            return YOLO26Seg, YOLO26SegConfig
        from samples.vision.ultralytics_yolo.runtime.python.segment import YoloSeg, YoloSegConfig
        return YoloSeg, YoloSegConfig
    if task == 'pose':
        if family == 'yolo26':
            from samples.vision.ultralytics_yolo.runtime.python.pose import YOLO26Pose, YOLO26PoseConfig
            return YOLO26Pose, YOLO26PoseConfig
        from samples.vision.ultralytics_yolo.runtime.python.pose import YoloPose, YoloPoseConfig
        return YoloPose, YoloPoseConfig
    from samples.vision.ultralytics_yolo.runtime.python.obb import YOLO26OBB, YOLO26OBBConfig
    return YOLO26OBB, YOLO26OBBConfig


def runtime_resize(profile, family, task):
    """Return the documented default resize policy for one family/task."""
    return 0 if family == 'yolo26' and task == 'cls' else default_resize_type(profile, task)


def prepare_runtime_model(profile, args):
    """Resolve the model class and build its configuration without loading SDKs.

    Args:
        profile: Resolved platform profile.
        args: Parsed command-line arguments from ``yolo_cli.build_parser``.

    Returns:
        Tuple of the model class and its configuration instance. The caller
        constructs the model with ``Model(config)``.

    Raises:
        UnsupportedAssetError: If the family/task combination is not published.
        ValueError: If DFL-only overrides are requested for a YOLO26 model.
    """
    Model, Config = get_task_types(profile, args.family, args.task)
    kw = dict(model_path=args.model_path, platform=profile, input_shape=args.input_shape,
              resize_type=args.resize_type if args.resize_type is not None else runtime_resize(profile, args.family, args.task))
    if args.task == 'cls':
        kw['topk'] = args.topk
    else:
        kw.update(score_thres=args.score_thres, strides=args.strides)
        if not is_nms_free(profile, args.family):
            kw['nms_thres'] = args.nms_thres if args.nms_thres is not None else default_nms_thres(profile, args.task)
        if args.classes_num is not None and args.task in ('detect', 'seg', 'obb'):
            kw['classes_num'] = args.classes_num
        if args.family != 'yolo26':
            kw['reg'] = args.reg
            if args.task == 'pose':
                kw['nkpt'] = args.nkpt
            if args.task == 'seg':
                kw['mces_num'] = args.mc
        elif args.reg != 16 or args.nkpt != 17 or args.mc != 32:
            raise ValueError('YOLO26 uses direct LTRB, 17 pose points and 32 mask coefficients; DFL overrides do not apply.')
        if args.task == 'obb':
            kw.update(angle_sign=args.angle_sign, angle_offset=args.angle_offset, regularize=bool(args.regularize))
    return Model, Config(**kw)

# ====================================================================
# Documented per-platform defaults.
# ====================================================================

def default_resize_type(profile: PlatformProfile, task: str) -> int:
    """Return the platform's documented default resize policy for a task.

    The two published trees resized classification inputs differently: the X5
    tree ran classification with letterbox resizing, the S tree with a direct
    stretch. Detection, segmentation and pose used letterbox resizing on both.

    Args:
        profile: Platform supplying the default.
        task: Task name.

    Returns:
        `0` for stretch resize, `1` for letterbox resize.
    """
    return profile.cls_resize_type if task == TASK_CLS else 1


def default_nms_thres(profile: PlatformProfile, task: str) -> Optional[float]:
    """Return the platform's documented default NMS threshold for a task.

    Args:
        profile: Platform supplying the default.
        task: Task name.

    Returns:
        The IoU threshold, or `None` for tasks that do not run NMS.
    """
    if task == TASK_CLS:
        return None
    return profile.nms_thres

# ====================================================================
# Parser, listing/dry-run, model preparation, labels and rendering.
# ====================================================================

_SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_MODEL_DIR = os.path.join(_SAMPLE_DIR, "model")


_TEST_DATA_DIR = os.path.join(_SAMPLE_DIR, "test_data")


_DEFAULT_IMAGE = os.path.join(_TEST_DATA_DIR, "bus.jpg")


_COCO_LABELS = os.path.join(_TEST_DATA_DIR, "coco_classes.names")


_IMAGENET_LABELS = os.path.join(_TEST_DATA_DIR, "imagenet_classes.names")


_OBB_LABELS = os.path.join(_TEST_DATA_DIR, "ultralytics_dota_classes.names")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO unified inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--platform', '--target', type=str, default=None,
                        choices=['auto', *available_platforms()],
                        help='Target platform. When omitted, the board is '
                             'detected from /sys/class/boardinfo. Preparation '
                             'may select any target; inference requires matching '
                             'local board identity.')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                        choices=list(SUPPORTED_TASKS),
                        help='Task type.')
    parser.add_argument('--family', type=str, default=None,
                        help='Model family, for example yolo11, yolov8, '
                             'yolov10 or yolov5u.')
    parser.add_argument('--model-size', type=str, default=None,
                        help='Model scale, for example n, s, m, l, x. '
                             'Defaults to the family default of the platform.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a compiled model. Overrides the resolved '
                             'default path and is never downloaded.')
    parser.add_argument('--asset-id', default=None,
                        help='Exact existing manifest reference group:sample:filename; '
                             'use --list-models to inspect references.')
    parser.add_argument('--input-shape', type=_input_shape, default=None,
                        help='Explicit model input geometry as HxW, used only '
                             'when the runtime does not report it.')
    parser.add_argument('--test-img', type=str, default=_DEFAULT_IMAGE,
                        help='Path to the test image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to a label file. Official default models '
                             'use the sample COCO labels (detect/seg), a '
                             'single person label (pose), ImageNet (cls) or '
                             'DOTA (obb); an explicit --model-path is treated '
                             'as a custom model that shows class IDs unless '
                             'this option is given.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save the rendered result image.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold.')
    parser.add_argument('--nms-thres', type=float, default=None,
                        help='IoU threshold for NMS. Defaults to the value the '
                             'selected platform documents.')
    parser.add_argument('--resize-type', type=int, default=None, choices=[0, 1],
                        help='Resize policy: 0 stretch, 1 letterbox. Defaults '
                             'to the value the selected platform documents.')
    parser.add_argument('--classes-num', type=int, default=None)
    parser.add_argument('--strides', type=lambda v: [int(x) for x in v.split(',')], default=[8,16,32])
    parser.add_argument('--mc', type=int, default=32)
    parser.add_argument('--angle-sign', type=float, default=1.0)
    parser.add_argument('--angle-offset', type=float, default=0.0, help='OBB offset in degrees')
    parser.add_argument('--regularize', type=int, choices=[0,1], default=1)
    parser.add_argument('--reg', type=int, default=16,
                        help='Number of DFL regression bins.')
    parser.add_argument('--nkpt', type=int, default=17,
                        help='[Pose] Number of keypoints.')
    parser.add_argument('--topk', type=int, default=5,
                        help='[Cls] Top-K classification results.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.50,
                        help='[Pose] Keypoint visibility threshold.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indexes to run inference on.')
    parser.add_argument('--list-models', action='store_true',
                        help='List the model assets the selected platform '
                             'publishes, then exit. Needs no board runtime.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the resolved model path and download URL, '
                             'then exit without downloading or running. Needs '
                             'no board runtime.')
    parser.add_argument('--download', action='store_true',
                        help='Download the resolved model if it is missing, '
                             'then exit without running inference.')
    return parser


def _input_shape(value):
    """Parse an `HxW` input geometry override.

    Args:
        value: The raw `--input-shape` value, or `None`.

    Returns:
        A `(height, width)` tuple, or `None` when no override was given.

    Raises:
        argparse.ArgumentTypeError: If the value is not `HxW` with positive
            integers.
    """
    if value is None:
        return None
    parts = str(value).lower().replace('*', 'x').split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be HxW, got {value!r}.")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be HxW with integers, got {value!r}.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be positive, got {value!r}.")
    return height, width


def print_model_listing(profile: PlatformProfile) -> None:
    """Print the model assets a platform publishes.

    Args:
        profile: Platform to describe.

    Returns:
        None.
    """
    print(f"Platform: {profile.key}")
    print(f"  artifact format : {profile.model_format}")
    print(f"  compiler march  : {profile.march}")
    print(f"  filename suffix : {profile.model_suffix}")
    print(f"  artifact dir    : "
          f"{os.path.join('model', profile.model_subdir) if profile.model_subdir else 'model'}")
    print(f"  NV12 input      : {profile.input_protocol}")
    print(f"  default NMS IoU : {profile.nms_thres}")
    print("  C++ sources     : detect/classify/pose/segment; board verification separate")
    print("  published assets:")
    for entry in family_listing(profile):
        tasks = ", ".join(
            f"{task}({'/'.join(sizes)})"
            for task, sizes in entry["tasks"].items())
        note = f"  # {entry['notes']}" if entry["notes"] else ""
        print(f"    {entry['family']:<9} default size {entry['default_size']}"
              f"  ->  {tasks}{note}")
        for task, sizes in entry['tasks'].items():
            for size in sizes:
                record = manifest_asset(profile, entry['family'], task, size)
                print(f'      {task}: {record.reference} (asset available; validation is separate)')


def select_manifest_reference(profile: PlatformProfile, args) -> None:
    """Resolve an official reference using the sample's finite task/size policy."""
    from utils.py_utils.assets import resolve_asset
    try:
        record = resolve_asset(args.asset_id)
    except ValueError as exc:
        raise UnsupportedAssetError(str(exc)) from exc
    if record.group != profile.family or record.sample_id not in ('ultralytics_yolo', 'ultralytics_yolo26'):
        raise UnsupportedAssetError('Asset reference does not belong to this target and Sample.')
    candidates = []
    for entry in family_listing(profile):
        for size in entry['tasks'].get(args.task, ()):
            candidate = manifest_asset(profile, entry['family'], args.task, size)
            if candidate.reference == record.reference:
                candidates.append((entry['family'], size))
    if len(candidates) != 1:
        raise UnsupportedAssetError('Asset reference is not registered for the selected target/task.')
    family, size = candidates[0]
    if (args.family and args.family != family) or (args.model_size and args.model_size != size):
        raise UnsupportedAssetError('Asset reference conflicts with family or model size.')
    args.family, args.model_size = family, size


def describe_plan(profile: PlatformProfile, args) -> dict:
    """Resolve the model asset a set of arguments selects.

    Args:
        profile: Selected platform.
        args: Parsed command-line arguments.

    Returns:
        A dictionary with the resolved filename, local path, download URL and
        whether the artifact is already present. When `--model-path` was given
        the URL is `None`, because an explicit path is never downloaded.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    if args.asset_id:
        select_manifest_reference(profile, args)
    inferred = family_from_filename(profile, args.model_path) if args.model_path else None
    if args.family and inferred and args.family != inferred:
        raise UnsupportedAssetError("--family conflicts with --model-path filename.")
    args.family = args.family or inferred or DEFAULT_FAMILY
    get_task_types(profile, args.family, args.task)
    if args.model_path:
        return {
            "asset_reference": args.asset_id,
            "filename": os.path.basename(args.model_path),
            "path": args.model_path,
            "url": None,
            "present": os.path.exists(args.model_path),
            "explicit": True,
        }
    filename = model_filename(
        profile, args.family, args.task, args.model_size)
    path = resolve_model_path(
        _MODEL_DIR, profile, args.family, args.task, args.model_size)
    return {
        "asset_reference": manifest_asset(profile, args.family, args.task, args.model_size).reference,
        "filename": filename,
        "path": path,
        "url": model_url(profile, args.family, args.task, args.model_size),
        "present": os.path.exists(path),
        "explicit": False,
    }


def print_dry_run(profile: PlatformProfile, args, plan: dict) -> None:
    """Print the resolved plan without touching the network or the board.

    Args:
        profile: Selected platform.
        args: Parsed command-line arguments.
        plan: The dictionary returned by `describe_plan`.

    Returns:
        None.
    """
    print("[dry-run] No model is downloaded and no inference is run.")
    print(f"  platform        : {profile.key} ({profile.march})")
    print(f"  task            : {args.task}")
    print(f"  family          : {args.family}")
    print(f"  model file      : {plan['filename']}")
    if plan.get('asset_reference'):
        print(f"  asset reference : {plan['asset_reference']}")
    print(f"  resolved path   : {plan['path']}")
    print(f"  already present : {'yes' if plan['present'] else 'no'}")
    if plan['explicit']:
        print("  download url    : (not applicable, --model-path was given)")
    else:
        print(f"  download url    : {plan['url']}")
    print(f"  NV12 protocol   : {profile.input_protocol}")
    print(f"  resize policy   : "
          f"{args.resize_type if args.resize_type is not None else runtime_resize(profile, args.family, args.task)}")
    if args.task != 'cls':
        print(f"  NMS IoU         : "
              f"{args.nms_thres if args.nms_thres is not None else profile.nms_thres}")
    if profile.model_subdir:
        print(f"  artifact dir    : model/{profile.model_subdir}/")


def ensure_model(plan: dict) -> None:
    """Download the resolved model when it is missing.

    Args:
        plan: The dictionary returned by `describe_plan`.

    Returns:
        None.

    Raises:
        FileNotFoundError: If an explicit model path does not exist, or if no
            download URL is available for a missing default asset.
    """
    from pathlib import Path

    from utils.py_utils.assets import resolve_asset, verify_asset_file, download_asset
    asset = resolve_asset(plan['asset_reference']) if plan.get('asset_reference') else None
    if plan['present']:
        if asset is not None:
            verify_asset_file(asset, Path(plan['path']))
        return
    if plan['explicit'] or not plan['url']:
        raise FileNotFoundError(f"Model file not found: {plan['path']}")
    print(f"[Download] {plan['url']}")
    if asset is None:
        raise ValueError('Default model download requires a manifest asset reference.')
    download_asset(asset, Path(plan['path']))


def load_labels(args, task: str, *, custom_model: bool = False) -> list:
    """Load the label names a task renders with.

    Args:
        args: Parsed command-line arguments.
        task: Task name.
        custom_model: True when the run uses an explicit local model path.
            Custom models never inherit the official COCO/ImageNet/DOTA
            label sets: without ``--label-file`` the caller renders class
            IDs (spec §6 — no default label guessing for self-trained
            models).

    Returns:
        A list of label names, empty when no label file is available.

    Raises:
        FileNotFoundError: If an explicit `--label-file` does not exist.
    """
    from utils.py_utils import file_io  # noqa: PLC0415 - keeps imports lazy

    if args.label_file:
        if not os.path.exists(args.label_file):
            raise FileNotFoundError(f"Label file not found: {args.label_file}")
        return _load_explicit_label_file(args.label_file)
    if custom_model:
        return []
    if task == 'pose':
        # Published pose models are single-class person. The old default
        # applied the 80-class COCO file (accidentally correct only because
        # COCO id 0 is "person"); the exact single label is returned now.
        return ['person']
    if task == 'obb':
        return file_io.load_class_names(_OBB_LABELS)
    default = _IMAGENET_LABELS if task == 'cls' else _COCO_LABELS
    if os.path.exists(default):
        return file_io.load_class_names(default)
    return []


class _IdFallbackLabels:
    """Label sequence rendering any class index as its ID string.

    Custom models without a label file must show class IDs on every
    presentation path; ``visualize.draw_boxes`` indexes
    ``class_names[cls_id]`` directly, so an empty list would crash. A zero
    length keeps guard-style consumers (``print_detections``) on their own
    ID fallback.
    """

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return str(int(index))


def _render_labels(labels):
    """Return a label sequence that never IndexErrors on unknown IDs."""
    return labels if labels else _IdFallbackLabels()


def _load_explicit_label_file(path: str) -> list:
    """Load one explicit ``--label-file`` preserving the legacy formats.

    The historical presentation path read cls label files through
    ``file_io.load_labels``, which accepts json dicts, json lists and
    line-per-name files; keep those working. The result must be a
    contiguous 0..N-1 mapping so the count validation and the render
    sequences stay meaningful; sparse mappings fail with a clear error
    instead of silently renumbering classes.
    """
    from utils.py_utils import file_io  # noqa: PLC0415 - keeps imports lazy

    mapping = file_io.load_labels(path)
    if not mapping:
        raise ValueError(
            f"Label file is empty or unreadable: {path}")
    if set(mapping) != set(range(len(mapping))):
        raise ValueError(
            f"Label file must map contiguous class IDs 0..{len(mapping) - 1} "
            f"found sparse keys {sorted(mapping)}: {path}")
    return [mapping[index] for index in range(len(mapping))]


def validate_label_count(labels, model) -> None:
    """Reject explicit labels whose count disagrees with the bound model.

    The class count comes from the model's bound contract (``contract.classes``
    exists for every dispatched task, including the 1000-class classification
    contract and the single-class pose contracts); ``--classes-num`` alone is
    not trusted. A model that does not expose an integer class count is an
    error, not a silent skip: real task models always expose one, and injected
    host doubles must declare their actual class count instead of leaving the
    count to be guessed from output protocols.
    """

    if not labels:
        return
    count = getattr(getattr(model, "contract", None), "classes", None)
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError(
            "Cannot validate labels: the constructed model does not expose "
            "an integer contract.classes. Real task models always do; "
            "injected test doubles must declare their actual class count.")
    if len(labels) != count:
        raise ValueError(
            f"{len(labels)} labels do not match the bound model's "
            f"{count} classes; check --label-file / --classes-num against "
            "the compiled model.")


def present_result(args, image, result, labels: list) -> None:
    """Render and save one finished prediction (presentation only).

    Args:
        args: Parsed command-line arguments.
        image: The BGR input image the prediction ran on.
        result: The task result returned by ``model.predict``.
        labels: Label names for rendering.

    Returns:
        None.
    """
    import cv2

    from utils.py_utils import file_io, visualize

    result_img = None
    render_labels = _render_labels(labels)
    if args.task == 'detect':
        boxes, scores, ids = result
        visualize.print_detections(boxes, scores, ids, render_labels)
        result_img = visualize.draw_boxes(image, boxes, ids, scores, render_labels, visualize.rdk_colors)
    elif args.task == 'seg':
        boxes, scores, ids, masks = result
        visualize.draw_masks(image, boxes, masks, ids, visualize.rdk_colors)
        result_img = visualize.draw_boxes(image, boxes, ids, scores, render_labels, visualize.rdk_colors)
    elif args.task == 'pose':
        boxes, scores, ids, xy, confidence = result
        kpts = np.concatenate([xy, confidence], axis=-1)
        result_img = visualize.draw_pose(image, boxes, kpts, kpt_conf_thres=args.kpt_conf_thres, scores=scores, class_ids=ids, colors=visualize.rdk_colors)
    elif args.task == 'obb':
        result_img = image.copy()
        for item in result:
            cx, cy, w, h, angle = item['rrect']
            points = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(angle)))).astype(np.int32)
            color = tuple(int(v) for v in visualize.rdk_colors[int(item['id']) % len(visualize.rdk_colors)])
            cv2.polylines(result_img, [points], True, color, 2)
            label = labels[item['id']] if 0 <= item['id'] < len(labels) else str(item['id'])
            cv2.putText(result_img, f"{label} {item['score']:.2f}", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    else:
        # Use the caller's validated labels; without labels render class
        # IDs. The default ImageNet file is never re-read here (custom
        # models must not fall back to official label sets).
        idx2label = {index: str(name) for index, name in enumerate(labels)} if labels \
            else {int(class_id): str(class_id) for class_id, _ in result}
        visualize.print_classification_results(result, idx2label)
    if result_img is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.img_save_path)), exist_ok=True)
        if not cv2.imwrite(args.img_save_path, result_img):
            raise OSError(f'Could not write image: {args.img_save_path}')
        print(f'[Saved] Result saved to: {args.img_save_path}')

__all__ = [
    "DEFAULT_FAMILY",
    "DEFAULT_TASK",
    "FamilySpec",
    "PlatformProfile",
    "SUPPORTED_TASKS",
    "UnsupportedAssetError",
    "UnsupportedPlatformError",
    "available_families",
    "available_platforms",
    "build_parser",
    "default_nms_thres",
    "default_resize_type",
    "describe_plan",
    "ensure_model",
    "family_from_filename",
    "family_listing",
    "family_registry",
    "get_task_types",
    "is_nms_free",
    "load_labels",
    "manifest_asset",
    "model_filename",
    "model_path",
    "model_url",
    "prepare_runtime_model",
    "present_result",
    "print_dry_run",
    "print_model_listing",
    "resolve_model_path",
    "resolve_platform",
    "select_manifest_reference",
    "validate_label_count",
]
