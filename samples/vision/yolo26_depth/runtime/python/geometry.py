# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Immutable per-call depth geometry shared by preparation and restoration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageContext:
    original_height: int
    original_width: int
    profile: str
    variant: str
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    size: int = 768


def make_context(height, width, profile, variant, size=768):
    if any(type(v) is not int or v <= 0 for v in (height, width, size)):
        raise ValueError("Image dimensions must be positive integers")
    if profile == "lite":
        return ImageContext(height, width, profile, variant, size=size)
    if profile != "nv12":
        raise ValueError(f"Unknown depth profile {profile!r}")
    ratio = min(size / height, size / width)
    h, w = round(height * ratio), round(width * ratio)
    if h <= 0 or w <= 0:
        raise ValueError("Aspect ratio collapses a letterbox dimension to zero")
    ph, pw = size - h, size - w
    return ImageContext(
        height,
        width,
        profile,
        variant,
        round(ph / 2 - 0.1),
        round(ph / 2 + 0.1),
        round(pw / 2 - 0.1),
        round(pw / 2 + 0.1),
        size,
    )


def validate_context(context, selection):
    if (
        not isinstance(context, ImageContext)
        or context.size != 768
        or context.profile != selection.profile
        or context.variant != selection.variant
    ):
        raise ValueError("Depth context must match the bound variant/profile/size")
    expected = make_context(
        context.original_height,
        context.original_width,
        context.profile,
        context.variant,
        context.size,
    )
    if context != expected:
        raise ValueError("Depth context padding does not match its source geometry")
