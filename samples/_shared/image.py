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

"""Image-byte conversion shared by YOLO, ResNet and PaddleOCR.

Only color conversion and plane interleaving live here. Resizing, validation,
physical tensor naming and packed/split transport remain with each sample.
Importing this module does not import NumPy, OpenCV or a board SDK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def bgr_to_nv12_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert even HxWx3 uint8 BGR pixels to uint8 Y and interleaved UV.

    Returns contiguous arrays shaped (1,H,W,1) and (1,H/2,W/2,2).
    Uses OpenCV I420 conversion exactly as the original three consumers.
    The caller owns image validation; OpenCV errors are not intercepted.
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    area = height * width
    planar = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(area * 3 // 2)
    y = planar[:area].reshape(1, height, width, 1)
    u = planar[area:area + area // 4].reshape(height // 2, width // 2)
    v = planar[area + area // 4:].reshape(height // 2, width // 2)
    uv = np.stack((u, v), axis=-1)[None]
    return np.ascontiguousarray(y), np.ascontiguousarray(uv)
