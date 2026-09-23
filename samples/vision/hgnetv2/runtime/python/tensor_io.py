# Copyright (c) 2026 D-Robotics Corporation
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

"""Image preprocessing and NV12 tensor packing for the HGNetV2 sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.tensor_io`); this module is the HGNetV2 import path
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
