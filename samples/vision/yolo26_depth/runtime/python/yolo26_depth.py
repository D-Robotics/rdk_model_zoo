"""Mixed-profile inference for the YOLO26 Depth RDK-S models.

Two model families share this module:

- ``NV12`` profile (variants ``n`` / ``s`` / ``m``): the ONNX graph ends with
  the calibrated ``clip -> scale/bias -> exp -> resize4x`` postprocess, the
  runtime feeds letterboxed NV12, and the model outputs 192x192 calibrated
  log-depth directly.
- ``lite`` profile (variants ``l`` / ``x``): the ONNX boundary is the raw
  192x192 depth logit; the runtime feeds a scale-filled float32 featuremap and
  applies ``clip -> scale/bias -> exp`` plus the final resize on the CPU.

Both profiles expose the same public ``DepthResult`` so callers do not need to
know which family a variant belongs to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from hbm_runtime import HB_HBMRuntime

# Variants whose HBM was compiled from the raw-logit lite ONNX boundary.
LITE_VARIANTS = ("l", "x")

# Per-variant scale/bias applied to the clipped raw logit (lite profile only).
LITE_CALIBRATION = {
    "l": (1.0, -0.2498779296875),
    "x": (1.0, -0.316650390625),
}

LOGIT_CLIP = (-4.0, 5.0)


@dataclass(frozen=True)
class DepthResult:
    """Store one depth inference result.

    Attributes:
        log_depth: Calibrated log-depth (both profiles, after clip/scale/bias).
        raw_logit: Uncalibrated raw logit (lite profile; ``None`` for NV12).
        depth_native: Relative depth restored to the source resolution.
        latency_ms: Time spent in the measured BPU inference call.
        profile: ``"nv12"`` or ``"lite"``.
    """

    log_depth: np.ndarray
    raw_logit: np.ndarray | None
    depth_native: np.ndarray
    latency_ms: float
    profile: str


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple]:
    """Resize with 114-value padding and return the restoration geometry.

    Args:
        image: Source BGR image with shape ``(height, width, 3)``.
        size: Square model input size in pixels.

    Returns:
        The padded BGR image and ``(top, bottom, left, right)`` padding.
    """
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
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, (top, bottom, left, right)


def bgr_to_nv12(image: np.ndarray) -> np.ndarray:
    """Pack an even-sized BGR image into the contiguous NV12 layout.

    Args:
        image: Even-sized BGR image with shape ``(height, width, 3)``.

    Returns:
        One-dimensional uint8 NV12 tensor accepted by ``hbm_runtime``.
    """
    import sys

    repository_root = Path(__file__).resolve().parents[5]
    if str(repository_root) not in sys.path:
        sys.path.append(str(repository_root))
    from utils.py_utils import preprocess as pre_utils

    y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(image)
    return np.concatenate((y_plane.reshape(-1), uv_plane.reshape(-1))).astype(np.uint8)


def featuremap(image: np.ndarray, size: int) -> np.ndarray:
    """Scale-fill BGR input, convert to RGB /255 NCHW float32.

    Args:
        image: Source BGR image with shape ``(height, width, 3)``.
        size: Square model input size in pixels.

    Returns:
        A ``(1, 3, size, size)`` float32 tensor in ``[0, 1]``.
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Render relative depth using a robust Turbo colormap.

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
    """Run one YOLO26 Depth HBM (NV12 or lite profile) on an RDK-S board.

    Args:
        model_path: Path to a YOLO26 Depth ``.hbm`` model.
        variant: One of ``n`` / ``s`` / ``m`` / ``l`` / ``x``.

    Attributes:
        model_name: Packed model name reported by ``hbm_runtime``.
        input_name: Model input tensor name.
        output_name: Model output tensor name.
        input_size: Square model input size in pixels.
        profile: ``"nv12"`` for n/s/m models, ``"lite"`` for l/x models.
    """

    def __init__(self, model_path: Path, variant: str) -> None:
        if variant not in ("n", "s", "m", "l", "x"):
            raise ValueError(f"unsupported variant: {variant}")
        self.model_path = model_path
        self.variant = variant
        self.profile = "lite" if variant in LITE_VARIANTS else "nv12"
        if self.profile == "lite":
            self.cal_a, self.cal_b = LITE_CALIBRATION[variant]
        self.runtime = HB_HBMRuntime(str(model_path))
        self.model_name = self.runtime.model_names[0]
        self.input_name = self.runtime.input_names[self.model_name][0]
        self.output_name = self.runtime.output_names[self.model_name][0]
        input_shape = self.runtime.input_shapes[self.model_name][self.input_name]
        self.input_size = int(input_shape[2])

    def _infer_nv12(self, image: np.ndarray, warmup: int) -> DepthResult:
        """Run the in-graph postprocess profile (n/s/m)."""
        padded, (top, bottom, left, right) = letterbox(image, self.input_size)
        nv12 = bgr_to_nv12(padded)
        inputs = {self.model_name: {self.input_name: nv12}}
        for _ in range(warmup):
            self.runtime.run(inputs)
        started = time.perf_counter()
        outputs = self.runtime.run(inputs)[self.model_name]
        latency_ms = (time.perf_counter() - started) * 1000.0
        log_depth = np.asarray(outputs[self.output_name], dtype=np.float32).squeeze()

        depth_square = cv2.resize(
            np.exp(log_depth), (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        height, width = image.shape[:2]
        cropped = depth_square[
            top : self.input_size - bottom, left : self.input_size - right
        ]
        depth_native = cv2.resize(
            cropped, (width, height), interpolation=cv2.INTER_LINEAR
        )
        return DepthResult(log_depth, None, depth_native, latency_ms, "nv12")

    def _infer_lite(self, image: np.ndarray, warmup: int) -> DepthResult:
        """Run the external postprocess profile (l/x)."""
        tensor = featuremap(image, self.input_size)
        inputs = {self.model_name: {self.input_name: tensor}}
        for _ in range(warmup):
            self.runtime.run(inputs)
        started = time.perf_counter()
        outputs = self.runtime.run(inputs)[self.model_name]
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_logit = np.asarray(outputs[self.output_name], dtype=np.float32).squeeze()
        log_depth = np.clip(raw_logit, *LOGIT_CLIP) * self.cal_a + self.cal_b
        height, width = image.shape[:2]
        depth_native = cv2.resize(
            np.exp(log_depth), (width, height), interpolation=cv2.INTER_LINEAR
        )
        return DepthResult(log_depth, raw_logit, depth_native, latency_ms, "lite")

    def infer(self, image: np.ndarray, warmup: int = 3) -> DepthResult:
        """Run BPU inference and restore relative depth to source resolution.

        Args:
            image: Source BGR image with shape ``(height, width, 3)``.
            warmup: Number of unmeasured inference calls before timing.

        Returns:
            Calibrated log-depth, raw logit (lite only), restored depth,
            latency, and the profile identifier.

        Raises:
            ValueError: If ``warmup`` is negative.
        """
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.profile == "lite":
            return self._infer_lite(image, warmup)
        return self._infer_nv12(image, warmup)

    def set_scheduling_params(self, priority: int, bpu_cores: list[int]) -> None:
        """Apply S-series scheduling parameters using the packed model name."""
        self.runtime.set_scheduling_params(
            priority={self.model_name: priority},
            bpu_cores={self.model_name: bpu_cores},
        )
