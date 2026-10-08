"""Read BGR images and convert pixels to NV12 planes for model samples.

NumPy and OpenCV are imported only when a helper is called. Resizing and
physical tensor naming remain with the caller.
"""

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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def bgr_to_nv12_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert BGR pixels to contiguous NV12 Y and interleaved UV planes.

    Args:
        image: uint8 BGR array shaped (H, W, 3), with positive even H/W and
            values in [0, 255]. The caller validates these requirements.

    Returns:
        tuple[np.ndarray, np.ndarray]: Contiguous uint8 Y (1, H, W, 1) and
        interleaved UV (1, H//2, W//2, 2), with byte values in [0, 255].

    Raises:
        cv2.error: OpenCV rejects the image format or dimensions.
        ValueError: The converted buffer cannot be reshaped to NV12 planes.
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


def read_bgr_image(path: str | Path) -> np.ndarray:
    """Read a local image as three-channel BGR pixels.

    Args:
        path: Image file path; leading ~ is expanded before reading.

    Returns:
        np.ndarray: uint8 BGR array shaped (H, W, 3), with values in [0, 255].
        OpenCV decodes the file in color mode, without preserving an alpha channel.

    Raises:
        FileNotFoundError: The path is missing or OpenCV cannot decode the image.
    """
    import cv2

    resolved = Path(path).expanduser()
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"image not found or unreadable: {resolved}")
    return image
