# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three-stage depth inference with explicit per-call restoration geometry.

``Yolo26DepthTask`` owns the model end to end: the immutable per-call depth
geometry (round/114 letterbox for the NV12 profiles), the published depth
tensor binding, the lazy SDK transport and the ``predict`` chain
(``preprocess`` → ``infer`` → ``postprocess``); the established
``pre_process``/``forward``/``post_process`` names stay thin aliases of those
implementations.  Published asset selection/listing and evidence rendering
live in ``cli.py``.
"""

from dataclasses import dataclass
import time
from typing import Mapping

import cv2
import numpy as np

from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import (
    RuntimeUnavailableError,
    SingleArrayRunner,
)
from samples.vision.yolo26_depth.runtime.python.cli import (
    LITE_CALIBRATION,
    ModelSelection,
    resolve_selection,
)


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

def resize_opencv(depth, height, width):
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def restore_log_depth(log_depth, context, *, resize=resize_opencv):
    """Finite calibrated F32 H×W log-depth → original-size relative depth.

    Caller validates profile/geometry. Optional resize supplies the source
    evaluator's Torch backend without changing the runtime's OpenCV default.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        depth = np.exp(log_depth)
    if not np.isfinite(depth).all():
        raise ValueError("Depth exponential overflow; verify model output boundary")
    if context.profile == "nv12":
        square = resize(depth, context.size, context.size)
        depth = square[
            context.top : context.size - context.bottom,
            context.left : context.size - context.right,
        ]
    return resize(depth, context.original_height, context.original_width)

@dataclass(frozen=True)
class ModelBinding:
    """Validated depth tensors and physical input roles.

    Attributes:
        input_name: Packed NV12, RGB lite, or split luma tensor name.
        uv_name: Split chroma tensor name; None for single-input profiles.
        output_name: The single float32 192-square depth tensor name.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str
    input_size: int = 768
    uv_name: str | None = None

    @property
    def model_name(self):
        return self.metadata.model_name


def bind_model(selection, metadata):
    resolved = resolve_selection(
        selection.target,
        variant=selection.variant,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
        converted_model=selection.converted_model,
    )
    if selection != resolved:
        raise ValueError(
            "ModelSelection differs from the manifest identity/profile/path"
        )
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if (
        meta.model_names != (meta.model_name,)
        or len(meta.input_names) not in (1, 2)
        or len(meta.output_names) != 1
    ):
        raise MetadataMismatchError(
            "Depth requires one model, one packed or two split inputs and one output"
        )
    inp, out = meta.input_names[0], meta.output_names[0]
    uv_name = None
    if len(meta.input_names) == 2:
        if selection.profile != "nv12" or selection.target == "x5":
            raise MetadataMismatchError("Split NV12 is supported by the S full depth profile")
        y_names = [n for n in meta.input_names if meta.input_shapes.get(n) == (1, 768, 768, 1)]
        uv_names = [n for n in meta.input_names if meta.input_shapes.get(n) == (1, 384, 384, 2)]
        if len(y_names) != 1 or len(uv_names) != 1 or any(
            meta.input_dtypes.get(n) != "uint8" for n in meta.input_names
        ):
            raise MetadataMismatchError("Split depth NV12 requires uint8 Y[1,768,768,1] and UV[1,384,384,2]")
        inp, uv_name = y_names[0], uv_names[0]
    if selection.profile == "lite":
        shapes = ((1, 3, 768, 768),)
        dtype = "float32"
        semantics = ("raw_logit",)
    else:
        shapes = ((1, 3, 768, 768), (1, 768, 768, 3), (1, 1152, 768, 1))
        dtype = "nv12"
        semantics = ("log_depth", "calibrated_log_depth")
    if uv_name is None and (
        meta.input_shapes.get(inp) not in shapes or meta.input_dtypes.get(inp) != dtype
    ):
        raise MetadataMismatchError(
            f"{selection.profile} requires 768-square {dtype} input metadata"
        )
    if (
        meta.output_shapes.get(out) not in ((1, 192, 192, 1), (1, 1, 192, 192))
        or meta.output_dtypes.get(out) != "float32"
    ):
        raise MetadataMismatchError(
            "Expected one float32 NHWC/NCHW 192-square depth channel"
        )
    declared = meta.output_semantics
    if isinstance(declared, dict):
        declared = declared.get(out)
    if declared is not None and declared not in semantics:
        raise MetadataMismatchError(
            f"Output semantic {declared!r} conflicts with {selection.profile}"
        )
    return ModelBinding(selection, meta, inp, out, uv_name=uv_name)

class RuntimeModelRunner(SingleArrayRunner):
    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        if selection.converted_model and runtime_factory is None and runtime is None:
            runtime_factory = _converted_factory(selection)
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_inputs=lambda b: (
                {n: (b.metadata.input_shapes[n], "uint8") for n in (b.input_name, b.uv_name)}
                if b.uv_name is not None else
                {b.input_name: (((1, 3, 768, 768), "float32")
                    if b.selection.profile == "lite" else ((768 * 768 * 3 // 2,), "uint8"))}
            ),
            task_name="YOLO26 Depth",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )


def _converted_factory(selection):
    """Explicit custom-artifact loader; target gating is never optional here.

    The referenced published asset selects only the declared tensor contract.
    Its publisher digest cannot certify user-generated bytes.
    """

    def create(path):
        require_execution_target(selection.target)
        if (
            not selection.model_path.is_file()
            or selection.model_path.stat().st_size == 0
        ):
            raise ValueError(
                f"Missing or empty converted model: {selection.model_path}"
            )
        try:
            from hbm_runtime import HB_HBMRuntime
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "hbm_runtime is required on the selected board"
            ) from exc
        return HB_HBMRuntime(path)

    return create

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
        """Nonempty BGR uint8 HWC → owned packed/split NV12 or RGB NCHW.

        Split S inputs keep the bound batch-one Y/UV NHWC plane shapes.
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
            if self.binding.uv_name is not None:
                tensors = {
                    self.binding.input_name: np.ascontiguousarray(y.reshape(1, 768, 768, 1)),
                    self.binding.uv_name: np.ascontiguousarray(uv.reshape(1, 384, 384, 2)),
                }
                return PreparedInput(tensors, ctx)
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
