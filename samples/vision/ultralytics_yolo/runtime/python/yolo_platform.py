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

"""Declare the deployment platforms this sample supports.

RDK X5 and the RDK S series run the same Ultralytics YOLO post-processing, but
they differ in the compiler toolchain, in the artifact format, and in the NV12
input tensor protocol. This module is the single place where those differences
are written down, so the runtime, the downloader and the evaluators can select
a platform explicitly instead of guessing.

An explicit platform argument always wins over host detection. An unrecognised
platform is rejected rather than silently mapped onto a supported one.

Typical Usage:
    >>> from yolo_platform import resolve_platform
    >>> profile = resolve_platform("x5")
    >>> profile.model_suffix
    'bayese'
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: Path that RDK boards use to publish the SoC name.
SOC_NAME_PATH = "/sys/class/boardinfo/soc_name"

#: Path that RDK S boards use to publish the board variant.
BOARD_TYPE_PATH = "/sys/class/boardinfo/board_type"

#: NV12 input is delivered as one packed tensor.
INPUT_PROTOCOL_PACKED = "packed"

#: NV12 input is delivered as two tensors, luma (Y) and chroma (UV).
INPUT_PROTOCOL_SPLIT = "split"

#: Artifact format produced by the RDK X5 toolchain.
MODEL_FORMAT_BIN = ".bin"

#: Artifact format produced by the RDK S toolchain.
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
    from samples._shared.platforms import detect_target
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
    from samples._shared.platforms import match_target
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
