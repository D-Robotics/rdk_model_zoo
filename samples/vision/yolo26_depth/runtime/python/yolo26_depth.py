"""Provide reusable RDK X5 inference for YOLO26 monocular depth models.

The module keeps model-specific letterbox geometry and depth restoration local
while reusing the repository NV12 conversion helpers.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from hbm_runtime import HB_HBMRuntime

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from utils.py_utils import preprocess as pre_utils


@dataclass(frozen=True)
class LetterboxGeometry:
    """Describe padding required to restore depth to the source image.

    Attributes:
        original_height: Source image height in pixels.
        original_width: Source image width in pixels.
        top: Padding inserted above the resized image.
        bottom: Padding inserted below the resized image.
        left: Padding inserted to the left of the resized image.
        right: Padding inserted to the right of the resized image.
    """

    original_height: int
    original_width: int
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class DepthResult:
    """Store one depth inference result.

    Attributes:
        log_depth: Raw calibrated log-depth output from the BPU model.
        depth_native: Relative depth restored to the source resolution.
        latency_ms: Time spent in the measured BPU inference call.
        geometry: Letterbox metadata used during restoration.
    """

    log_depth: np.ndarray
    depth_native: np.ndarray
    latency_ms: float
    geometry: LetterboxGeometry


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxGeometry]:
    """Resize an image with the calibration-time 114-padding policy.

    Args:
        image: Source BGR image with shape ``(height, width, 3)``.
        size: Square model input size in pixels.

    Returns:
        A tuple containing the padded BGR image and restoration geometry.

    Raises:
        ValueError: If the input image is empty or the target size is invalid.
    """
    if image.size == 0:
        raise ValueError("image must not be empty")
    if size <= 0:
        raise ValueError("size must be positive")

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = size - resized_width
    pad_height = size - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    geometry = LetterboxGeometry(height, width, top, bottom, left, right)
    return padded, geometry


def bgr_to_nv12(image: np.ndarray) -> np.ndarray:
    """Pack a BGR image into the contiguous NV12 layout used by X5.

    Args:
        image: Even-sized BGR image with shape ``(height, width, 3)``.

    Returns:
        One-dimensional uint8 NV12 tensor accepted by ``hbm_runtime``.
    """
    y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(image)
    return np.concatenate((y_plane.reshape(-1), uv_plane.reshape(-1))).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert relative depth into a robust Turbo visualization.

    Args:
        depth: Float32 relative-depth map.

    Returns:
        BGR uint8 visualization with the same spatial resolution.
    """
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(depth[finite], [2.0, 98.0])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    gray = np.asarray(normalized * 255.0, dtype=np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)


class Yolo26Depth:
    """Run one YOLO26 Depth BIN on RDK X5.

    Args:
        model_path: Path to a YOLO26 Depth ``.bin`` model.

    Attributes:
        model_name: Packed model name reported by ``hbm_runtime``.
        input_name: Model input tensor name.
        output_name: Model output tensor name.
        input_size: Square model input size in pixels.

    Notes:
        Instances are intended for serial inference from one process.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.runtime = HB_HBMRuntime(str(model_path))
        self.model_name = self.runtime.model_names[0]
        self.input_name = self.runtime.input_names[self.model_name][0]
        self.output_name = self.runtime.output_names[self.model_name][0]
        input_shape = self.runtime.input_shapes[self.model_name][self.input_name]
        self.input_size = int(input_shape[2])

    def infer(self, image: np.ndarray, warmup: int = 3) -> DepthResult:
        """Run BPU inference and restore relative depth to source resolution.

        Args:
            image: Source BGR image with shape ``(height, width, 3)``.
            warmup: Number of unmeasured inference calls before timing.

        Returns:
            Raw log-depth, restored depth, latency, and letterbox geometry.

        Raises:
            ValueError: If ``warmup`` is negative.
        """
        if warmup < 0:
            raise ValueError("warmup must be non-negative")

        # Preserve calibration geometry before packing the model input.
        padded, geometry = letterbox(image, self.input_size)
        nv12 = bgr_to_nv12(padded)
        inputs = {self.model_name: {self.input_name: nv12}}
        for _ in range(warmup):
            self.runtime.run(inputs)

        # Measure only the BPU runtime call, excluding image processing.
        started = time.perf_counter()
        outputs = self.runtime.run(inputs)[self.model_name]
        latency_ms = (time.perf_counter() - started) * 1000.0
        log_depth = np.asarray(outputs[self.output_name], dtype=np.float32).squeeze()

        # Decode log-depth and remove letterbox padding before final resize.
        depth_square = cv2.resize(
            np.exp(log_depth),
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        cropped = depth_square[
            geometry.top : self.input_size - geometry.bottom,
            geometry.left : self.input_size - geometry.right,
        ]
        depth_native = cv2.resize(
            cropped,
            (geometry.original_width, geometry.original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return DepthResult(log_depth, depth_native, latency_ms, geometry)
