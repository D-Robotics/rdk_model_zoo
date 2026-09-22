"""Image preprocessing and NV12 tensor packing for the EfficientFormerV2 sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.tensor_io`); this module is the EfficientFormerV2 import path
for that surface.  It is independent of the board runtime: it translates an
OpenCV BGR image into the physical input arrays required by a validated
``ModelBinding`` and records the actual geometry used for the transformation.
"""

from __future__ import annotations

from samples._shared.tensor_io import (  # noqa: F401 - re-exported surface
    ImageTransform,
    PreparedInput,
    as_packed,
    as_split,
    bgr_to_nv12_planes,
    pack_nv12,
    pack_nv12_planes,
    pack_nv12_single,
    prepare_nv12,
    resize_bgr,
    validate_input_tensors,
)

__all__ = [
    "ImageTransform",
    "PreparedInput",
    "as_packed",
    "as_split",
    "bgr_to_nv12_planes",
    "pack_nv12",
    "pack_nv12_planes",
    "pack_nv12_single",
    "prepare_nv12",
    "resize_bgr",
    "validate_input_tensors",
]
