"""Platform deployment profiles shared by the unified samples (Phase 1.5 H5).

A :class:`PlatformProfile` records the facts that are invariant for one
deployment platform across the samples that support it: the compiler march,
the published artifact format, the NV12 input protocol and the URL layout of
the archive.  Facts that vary per sample or per model variant (resize
defaults, NMS thresholds, input geometry) deliberately stay in each sample's
binding contract; the profile never becomes a second authority for them.

The reference field shape is the pilot ``ultralytics_yolo`` module
(``yolo_platform.py``), which keeps its own local copy until the B9 lift.
Board identity itself is owned by :mod:`samples._shared.platforms`; this
module only intersects that identity with the platforms a sample declares.

Typical Usage:
    >>> from samples._shared.platform_profile import (
    ...     classification_profiles, resolve_profile)
    >>> PLATFORMS = classification_profiles(url_prefix_s="rdk_s100/MobileNet")
    >>> profile = resolve_profile(PLATFORMS, "s100")
    >>> profile.model_format
    '.hbm'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

#: Root of the published artifact archive.
ARCHIVE_ROOT = "https://archive.d-robotics.cc/downloads/rdk_model_zoo"

#: NV12 input is delivered as one packed tensor (flat 1-D buffer, H2).
INPUT_PROTOCOL_PACKED = "packed"

#: NV12 input is delivered as two tensors, luma (Y) and chroma (UV).
INPUT_PROTOCOL_SPLIT = "split"

#: Artifact format produced by the RDK X5 toolchain.
MODEL_FORMAT_BIN = ".bin"

#: Artifact format produced by the RDK S toolchain.
MODEL_FORMAT_HBM = ".hbm"


class UnsupportedProfileError(ValueError):
    """Raised when a requested platform is not declared by this sample."""


@dataclass(frozen=True)
class PlatformProfile:
    """Deployment facts that are constant for one platform across samples.

    Attributes:
        key: Canonical platform name accepted on the command line.
        family: Broad platform family, ``"x5"`` or ``"s"``.
        soc_names: SoC names that map onto this platform during detection.
        march: BPU architecture name used by the compiler toolchain.
        model_suffix: Suffix embedded in published model filenames.
        model_format: Published artifact file extension.
        model_subdir: Sub-directory that holds this platform's artifacts,
            relative to the sample ``model/`` directory.  Empty for flat
            layouts.
        input_protocol: NV12 tensor protocol, ``"packed"`` or ``"split"``.
        url_prefix: Directory segment appended to :data:`ARCHIVE_ROOT` for
            this platform's published artifacts, or ``None`` when no artifact
            is published (the profile then exists only so an explicit
            selection fails with an honest "no published asset" error).
        cls_interpolation: Platform default resize interpolation for the
            standardized classification samples (X5 sources pass INTER_LINEAR
            explicitly; S sources use their helper's INTER_NEAREST default).
        supports_cpp: Whether a C++ runtime is delivered for this platform.
            Per-sample; the shared profile carries the common classification
            delivery shape.
    """

    key: str
    family: str
    soc_names: tuple[str, ...]
    march: str
    model_suffix: str
    model_format: str
    model_subdir: str
    input_protocol: str
    url_prefix: Optional[str]
    cls_interpolation: str
    supports_cpp: bool

    @property
    def is_packed_input(self) -> bool:
        """Return True when NV12 is bound as a single packed tensor."""

        return self.input_protocol == INPUT_PROTOCOL_PACKED

    def model_base_url(self) -> str:
        """Return the archive directory that published filenames append to.

        Returns:
            The URL prefix ``ARCHIVE_ROOT/<url_prefix>``.

        Raises:
            UnsupportedProfileError: If no artifact is published for this
                platform (``url_prefix`` is ``None``).
        """

        if not self.url_prefix:
            raise UnsupportedProfileError(
                f"No artifact is published for platform {self.key!r}; the "
                "profile exists so explicit selection fails honestly."
            )
        return f"{ARCHIVE_ROOT}/{self.url_prefix}"


def resolve_profile(
    platforms: Mapping[str, PlatformProfile],
    platform: Optional[str] = None,
    *,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> PlatformProfile:
    """Select one declared profile, preferring an explicit request.

    Args:
        platforms: The profiles a sample declares, keyed by platform name.
        platform: Explicit platform name; validated against ``platforms``
            without consulting the host.
        soc_name: SoC name to use instead of reading the board.  Only used
            when ``platform`` is not given.
        board_type: Board variant refining ``soc_name``.

    Returns:
        The selected :class:`PlatformProfile`.

    Raises:
        UnsupportedProfileError: If the explicit platform is not declared, or
            no explicit platform was given and the host either cannot be
            identified or is not declared by this sample.  Detection never
            falls back to a different platform.
    """

    from samples._shared.platforms import detect_target, match_target

    supported = ", ".join(platforms) or "none"
    if platform is not None:
        key = str(platform).strip().lower()
        profile = platforms.get(key)
        if profile is None:
            raise UnsupportedProfileError(
                f"Unsupported platform {platform!r}. Declared platforms: "
                f"{supported}."
            )
        return profile

    detected = (
        match_target(soc_name, board_type) if soc_name is not None
        else detect_target()
    )
    if detected is None:
        raise UnsupportedProfileError(
            "Cannot select a platform automatically (no declared board "
            f"detected). Pass --target explicitly. Declared platforms: "
            f"{supported}."
        )
    profile = platforms.get(detected)
    if profile is None:
        raise UnsupportedProfileError(
            f"Detected platform {detected!r} is not declared by this sample. "
            f"Declared platforms: {supported}."
        )
    return profile


def classification_profiles(
    *,
    url_prefix_s: str,
    url_prefix_x5: str = "rdk_x5",
    supports_cpp_x5: bool = False,
    supports_cpp_s: bool = False,
) -> dict[str, PlatformProfile]:
    """Build the standard four-profile table for a classification sample.

    The standardized classification samples share one delivery shape: X5
    publishes flat ``.bin`` artifacts, S100/S600 publish ``.hbm`` artifacts
    under an ``s100``/``s600`` directory, and S100P publishes nothing (the
    legacy ``download_model.sh`` silently fell back to the S100 build, which
    the unified samples reject).  Sample-specific values (the archive
    directory, C++ delivery) are parameters; everything else is platform
    fact.

    Args:
        url_prefix_s: Archive directory for the S100/S600 artifacts, e.g.
            ``"rdk_s100/MobileNet"``.  The same second segment is used for
            S600 (``rdk_s600/...``).
        url_prefix_x5: Archive directory for the X5 artifacts.
        supports_cpp_x5: Whether this sample ships an X5 C++ runtime.
        supports_cpp_s: Whether this sample ships an S-series C++ runtime.

    Returns:
        A mapping of platform key to profile, covering ``x5``, ``s100``,
        ``s100p`` and ``s600``.
    """

    s_base, _, s_leaf = url_prefix_s.partition("/")
    if not s_leaf:
        raise ValueError(
            f"url_prefix_s must be '<soc_dir>/<sample_dir>', got {url_prefix_s!r}."
        )
    return {
        "x5": PlatformProfile(
            key="x5",
            family="x5",
            soc_names=("x5",),
            march="bayes-e",
            model_suffix="bayese",
            model_format=MODEL_FORMAT_BIN,
            model_subdir="",
            input_protocol=INPUT_PROTOCOL_PACKED,
            url_prefix=url_prefix_x5,
            cls_interpolation="linear",
            supports_cpp=supports_cpp_x5,
        ),
        "s100": PlatformProfile(
            key="s100",
            family="s",
            soc_names=("s100",),
            march="nash-e",
            model_suffix="nashe",
            model_format=MODEL_FORMAT_HBM,
            model_subdir="s100",
            input_protocol=INPUT_PROTOCOL_SPLIT,
            url_prefix=url_prefix_s,
            cls_interpolation="nearest",
            supports_cpp=supports_cpp_s,
        ),
        "s100p": PlatformProfile(
            key="s100p",
            family="s",
            soc_names=("s100p",),
            march="nash-m",
            model_suffix="nashm",
            model_format=MODEL_FORMAT_HBM,
            model_subdir="s100p",
            input_protocol=INPUT_PROTOCOL_SPLIT,
            # No classification artifact is published for S100P; selecting it
            # explicitly must fail with "no published asset", not fall back.
            url_prefix=None,
            cls_interpolation="nearest",
            supports_cpp=supports_cpp_s,
        ),
        "s600": PlatformProfile(
            key="s600",
            family="s",
            soc_names=("s600",),
            march="nash-p",
            model_suffix="nashp",
            model_format=MODEL_FORMAT_HBM,
            model_subdir="s600",
            input_protocol=INPUT_PROTOCOL_SPLIT,
            url_prefix=url_prefix_s.replace(s_base, "rdk_s600", 1),
            cls_interpolation="nearest",
            supports_cpp=supports_cpp_s,
        ),
    }


__all__ = [
    "ARCHIVE_ROOT",
    "INPUT_PROTOCOL_PACKED",
    "INPUT_PROTOCOL_SPLIT",
    "MODEL_FORMAT_BIN",
    "MODEL_FORMAT_HBM",
    "PlatformProfile",
    "UnsupportedProfileError",
    "classification_profiles",
    "resolve_profile",
]
