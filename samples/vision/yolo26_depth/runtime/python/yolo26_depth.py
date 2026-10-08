# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three-stage depth inference with explicit per-call restoration geometry.

``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
established ``pre_process``/``forward``/``post_process`` names stay thin
aliases of those implementations.
"""

from dataclasses import dataclass
import time
from typing import Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from samples.vision.yolo26_depth.runtime.python.tensor_io import restore_log_depth
from samples.vision.yolo26_depth.runtime.python.geometry import (
    ImageContext,
    make_context,
    validate_context,
)
from samples.vision.yolo26_depth.runtime.python.model_binding import LITE_CALIBRATION


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: ImageContext


@dataclass(frozen=True)
class DepthResult:
    log_depth: np.ndarray
    depth_native: np.ndarray
    raw_logit: np.ndarray | None
    context: ImageContext


@dataclass(frozen=True)
class DepthPredictionDetails:
    """One predict call's owned result plus warmup count and forward latency.

    ``latency_ms`` covers exactly one forward call (transport validation and
    the owned output copy included, preprocessing/postprocessing excluded) and
    ``warmup`` records how many unmeasured forwards preceded it.  Callers
    request this record with ``return_details=True``; the task never retains a
    last image, output or timing.
    """

    result: DepthResult
    prepared: PreparedInput
    raw: np.ndarray
    warmup: int
    latency_ms: float


class Yolo26DepthTask:
    """BGR → profile-specific input → raw F32 → original-size relative depth.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.
    """

    def __init__(self, selection=None, *, runner=None, binding=None, runtime_factory=None):
        """Load the selected artifact and bind it, or accept an injected runner.

        Args:
            selection: ModelSelection from ``model_binding.resolve_selection``;
                construction loads the artifact through the shared SDK adapter
                (board identity and artifact bytes verified before the SDK is
                imported) and binds its tensor contract.
            runner: Already-constructed runner (host-test seam).
            binding: Explicit binding for a callable-only runner; defaults to
                the runner's own loaded binding.
            runtime_factory: Optional SDK factory injection for host fixtures.

        Raises:
            TypeError: When neither a ModelSelection nor a runner is given.
            ValueError: When an injected runner carries no binding.
        """
        if runner is None and binding is None:
            from samples.vision.yolo26_depth.runtime.python.model_binding import ModelSelection
            from samples.vision.yolo26_depth.runtime.python.model_runner import RuntimeModelRunner
            if not isinstance(selection, ModelSelection):
                raise TypeError("Pass a ModelSelection from resolve_selection, or inject runner=/binding=.")
            if runtime_factory is None:
                runner = RuntimeModelRunner(selection)
            else:
                runner = RuntimeModelRunner(selection, runtime_factory=runtime_factory)
            binding = runner.load()
        elif runner is not None and binding is None:
            binding = getattr(runner, "binding", None)
            if binding is None:
                raise ValueError("An injected runner must supply its binding.")
        self.runner = runner
        self.binding = binding

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply explicit scheduling values to the loaded board runtime."""
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> PreparedInput:
        """Nonempty BGR uint8 HWC → owned flat NV12 or RGB float32 NCHW.

        NV12 uses INTER_LINEAR letterbox with padding 114; lite uses scale-fill
        and /255. Geometry is returned with the tensor, never stored on the task.
        Invalid images or a collapsed letterbox dimension raise ValueError.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected a nonempty BGR uint8 HWC image")
        s = self.binding.selection
        ctx = make_context(*image.shape[:2], s.profile, s.variant)
        if s.profile == "lite":
            resized = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            value = (
                np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
                / 255.0
            )
        else:
            h, w = 768 - ctx.top - ctx.bottom, 768 - ctx.left - ctx.right
            resized = (
                image
                if image.shape[:2] == (h, w)
                else cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            )
            padded = cv2.copyMakeBorder(
                resized,
                ctx.top,
                ctx.bottom,
                ctx.left,
                ctx.right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            y, uv = bgr_to_nv12_planes(padded)
            value = np.concatenate((y.reshape(-1), uv.reshape(-1)))
        return PreparedInput({self.binding.input_name: value}, ctx)

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return the runner's raw float32 single-channel tensor without decoding."""
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray, context: ImageContext) -> DepthResult:
        """Raw F32 192-square output → calibrated log map and relative depth.

        NV12 is already calibrated: exp, resize to 768, crop padding, restore.
        Lite alone clips [-4,5] and applies source calibration before exp and
        direct restoration. Wrong tensors/context, NaN/Inf and exp overflow
        raise ValueError; arrays are owned and no metric-depth claim is made.
        """
        validate_context(context, self.binding.selection)
        shape = self.binding.metadata.output_shapes[self.binding.output_name]
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != shape
            or raw.dtype != np.float32
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Expected finite float32 output matching the bound shape")
        plane = raw.reshape(192, 192).copy()
        raw_logit = None
        if context.profile == "lite":
            raw_logit = plane.copy()
            a, b = LITE_CALIBRATION[context.variant]
            log_depth = np.clip(plane, -4.0, 5.0) * a + b
        else:
            log_depth = plane
        restored = restore_log_depth(log_depth, context)
        return DepthResult(log_depth, restored, raw_logit, context)

    def predict(self, image: np.ndarray, *, warmup: int = 0,
                return_details: bool = False):
        """Run the same three stages once, with no file IO or rendering.

        ``warmup`` (validated before any execution) runs that many unmeasured
        forwards on the prepared tensors first.  ``return_details=True`` wraps
        the usual :class:`DepthResult` with this call's prepared input, raw
        output, the applied warmup count and the one-forward latency; the
        default return stays the plain :class:`DepthResult` and the timing
        never includes preprocessing or postprocessing.
        """
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        prepared = self.preprocess(image)
        for _ in range(warmup):
            self.infer(prepared.tensors)
        started = time.perf_counter()
        raw = self.infer(prepared.tensors)
        latency_ms = (time.perf_counter() - started) * 1000
        result = self.postprocess(raw, prepared.context)
        if return_details:
            return DepthPredictionDetails(
                result, prepared, raw, warmup, latency_ms)
        return result

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray, context: ImageContext) -> DepthResult:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw, context)
