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
"""YOLO detection: DFL, NMS-free S YOLOv10 and direct-LTRB YOLO26 in one file.

This task file owns the complete detection pipeline for every published
family.  :class:`YoloDetect` is the readable DFL detector; the S-series
NMS-free YOLOv10 head is :class:`YoloV10Detect`, a contract adapter on the
same three stages; YOLO26's direct four-channel LTRB distances are
:class:`YOLO26Detect`.  Below the classes this file also carries the numeric
decode implementation (DFL soft-max integration, LTRB offsets,
classwise/agnostic NMS), the letterbox geometry and the shared NV12
prepare/forward transport that ``segment.py``/``pose.py``/``obb.py``/
``classify.py`` and the YOLOE sample reuse for the same algorithms.  Tensor
contracts, the runner and metadata binding live in ``backend.py``; selection
and rendering in ``cli.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.py_utils import (
    preprocess as pre_utils,
)
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    BindingError,
    DFLDetectionContract,
    LTRBDetectionContract,
    ModelRunner,
    ModelSelection,
    RawOutputs,
    build_runner,
    default_dfl_contract,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import (
    PlatformProfile,
    resolve_platform,
)

# ====================================================================
# The DFL detector.
# ====================================================================

@dataclass
class YoloDetectConfig:
    """Configuration retained for the established detector entrypoints."""

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = None
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    contract: Optional[DFLDetectionContract] = None


class YoloDetect:
    """DFL detector with a replaceable model execution boundary.

    The object loads the model once and reuses it across ``predict``
    calls.  Each call carries its own resize/letterbox context on the
    prepared input, so consecutive images of different sizes never reuse
    a stale transform.  Prediction itself prints nothing, draws nothing
    and writes no files; presentation belongs to the caller.
    """

    def __init__(self,
                 config: YoloDetectConfig,
                 runner: Any = None,
                 model_runner: Any = None) -> None:
        if runner is not None and model_runner is not None:
            raise ValueError("Pass only one of runner or model_runner.")
        if model_runner is not None:
            runner = model_runner
        self.cfg = config
        self.runner = build_runner(config) if runner is None else runner
        self.binding = getattr(self.runner, "binding", None)
        self.model = getattr(self.runner, "model", self.runner)
        self.input_adapter = getattr(self.runner, "input_adapter", None)
        if self.input_adapter is None and self.binding is not None:
            self.input_adapter = getattr(self.binding, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(self.runner, config)
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError("Model input dimensions must be positive.")
        self.input_size = (self.input_h, self.input_w)
        self.model_name = getattr(self.runner, "model_name", None)
        if self.model_name is None and self.binding is not None:
            self.model_name = getattr(self.binding, "model_name", None)
        self.input_names = tuple(getattr(self.runner, "input_names", ()) or ())
        if not self.input_names and self.input_adapter is not None:
            self.input_names = tuple(getattr(self.input_adapter, "input_names", ()) or ())
        self.output_names = tuple(getattr(self.runner, "output_names", ()) or ())
        self.input_shapes = dict(getattr(self.runner, "input_shapes", {}) or {})
        self.output_shapes = dict(getattr(self.runner, "output_shapes", {}) or {})

        if self.binding is not None:
            self.contract = getattr(self.binding, "contract", None)
        else:
            self.contract = config.contract
        if self.contract is None:
            self.contract = default_dfl_contract(
                classes=config.classes_num,
                reg_bins=config.reg,
                strides=config.strides,
            )
        self.anchor_sizes = _normalise_grids(
            config.anchor_sizes, self.input_size, self.contract.strides)
        self.grid_shapes = list(self.anchor_sizes)
        self.weights_static = np.arange(
            int(self.contract.reg_bins), dtype=np.float32)[None, None, :]
        if self.cfg.nms_thres is None:
            profile = config.platform
            value = getattr(profile, "nms_thres", None) if profile is not None else None
            if value is not None:
                self.cfg.nms_thres = float(value)

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Forward explicit runtime scheduling parameters to the runner."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self,
                   source: "str | Path | np.ndarray",
                   image_format: str = "BGR") -> PreparedDetection:
        """Read one image (path or BGR array) and pack the bound NV12 tensors.

        The input array is never modified in place; the returned
        :class:`PreparedDetection` carries this call's resize/letterbox
        geometry on ``.transform`` for ``postprocess``.
        """

        if str(image_format).upper() != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")
        img = _read_source_image(source)
        if (
            not isinstance(img, np.ndarray)
            or img.ndim != 3
            or img.shape[2] != 3
            or img.dtype != np.uint8
            or min(img.shape[:2]) <= 0
        ):
            raise ValueError("img must be a nonempty BGR uint8 HxWx3 NumPy array.")
        resized, transform = resize_with_transform(
            img, self.input_size, resize_type=self.cfg.resize_type)
        y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(resized)
        tensors = _build_input(self.runner, self.input_adapter, y_plane, uv_plane)
        return PreparedDetection(tensors, transform)

    def infer(self, input_tensor: "PreparedDetection | Mapping[str, Any]"):
        """Execute exactly one model call on the prepared NV12 tensors."""

        tensors = (input_tensor.tensors
                   if isinstance(input_tensor, PreparedDetection) else input_tensor)
        runner = self.runner
        if callable(runner):
            return runner(tensors)
        method = getattr(runner, "forward", None) or getattr(runner, "run", None)
        if method is None:
            raise BindingError(
                "The injected runner is not callable and has no forward/run method."
            )
        return method(tensors)

    def postprocess(self,
                    outputs: Any,
                    ori_img_w: Optional[int] = None,
                    ori_img_h: Optional[int] = None,
                    score_thres: Optional[float] = None,
                    nms_thres: Optional[float] = None,
                    transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Decode DFL outputs into owned DetectionResult arrays.

        Supply the matching PreparedDetection.transform, or explicit original
        dimensions for the legacy stateless path. RawOutputs uses its validated
        binding for postprocess layout adaptation; injected semantic mappings must hold
        floating values. Wrong binding/geometry/quantization raises ValueError
        (including BindingError); score/NMS overrides follow the decoder contract.
        """

        semantic = _semantic_outputs(self.binding, self.contract, outputs, "DFL")
        score = self.cfg.score_thres if score_thres is None else float(score_thres)
        nms = self.cfg.nms_thres if nms_thres is None else float(nms_thres)
        if self.contract.nms == "none":
            nms = None
        boxes, scores, class_ids = decode_dfl(
            semantic,
            self.contract,
            input_size=self.input_size,
            score_thres=score,
            nms_thres=nms,
        )
        concrete = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h, self.input_size, self.cfg.resize_type)
        boxes = inverse_boxes(boxes, concrete)
        return DetectionResult(boxes, scores, class_ids)

    def predict(self,
                source: "str | Path | np.ndarray",
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None) -> DetectionResult:
        """Run the full pipeline for one image path or BGR array."""

        prepared = self.preprocess(source, image_format)
        outputs = self.infer(prepared)
        return self.postprocess(
            outputs,
            score_thres=score_thres,
            nms_thres=nms_thres,
            transform=prepared.transform,
        )

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None) -> DetectionResult:
        """Tuple-compatible alias for :meth:`predict`."""
        return self.predict(img, image_format, score_thres, nms_thres)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self,
                    img: "str | Path | np.ndarray",
                    image_format: str = "BGR") -> PreparedDetection:
        """Compatibility alias for :meth:`preprocess`."""

        return self.preprocess(img, image_format)

    def forward(self, input_tensor: "PreparedDetection | Mapping[str, Any]"):
        """Compatibility alias for :meth:`infer`."""

        return self.infer(input_tensor)

    def post_process(self,
                     outputs: Any,
                     ori_img_w: Optional[int] = None,
                     ori_img_h: Optional[int] = None,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Compatibility alias for :meth:`postprocess`."""

        return self.postprocess(
            outputs, ori_img_w, ori_img_h, score_thres, nms_thres, transform)

# ====================================================================
# S-series YOLOv10: the shared DFL detector with an enforced NMS-free contract.
# ====================================================================

@dataclass
class YoloV10DetectConfig:
    """Published S v10 DFL logits: 16 bins, strides 8/16/32 and no NMS.

    Geometry comes from metadata. resize_type is 0=stretch or 1=letterbox;
    classes_num defaults to COCO-80 and score_thres to 0.25.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    contract: Optional[DFLDetectionContract] = None
    nms_thres: Optional[float] = field(default=None, init=False)


class YoloV10Detect(YoloDetect):
    """Reuse all three DFL stages, preserving every score-qualified anchor.

    Results follow stride/grid traversal rather than score sorting or Top-K.
    PreparedDetection carries per-image geometry; results own their arrays.
    """

    def __init__(self, config: YoloV10DetectConfig, runner=None):
        requested = config.contract or DFLDetectionContract(
            classes=config.classes_num,
            reg_bins=config.reg,
            strides=config.strides,
            nms="none",
        )
        effective = getattr(getattr(runner, "binding", None), "contract", requested)
        for contract in (requested, effective):
            if (
                contract.task != "detect"
                or contract.nms != "none"
                or getattr(contract, "reg_bins", None) != 16
                or tuple(contract.strides) != (8, 16, 32)
                or contract.classification != "logits"
                or contract.box_distribution != "logits"
            ):
                raise ValueError(
                    "S YOLOv10 requires DFL logits, 16 bins, strides 8/16/32 and no NMS."
                )
        super().__init__(replace(config, contract=requested), runner=runner)
        if self.input_h != self.input_w:
            raise ValueError("Published S YOLOv10 requires square model input.")

# ====================================================================
# YOLO26 direct-offset detection.
# ====================================================================

@dataclass
class YOLO26DetectConfig:
    """Configuration retained for the established YOLO26 detector entrypoint."""

    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = 0.45
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    contract: Optional[LTRBDetectionContract] = None


def _ltrb_selection_from_config(config: YOLO26DetectConfig,
                                 contract: LTRBDetectionContract) -> ModelSelection:
    profile = getattr(config, "platform", None)
    target = getattr(profile, "key", profile if isinstance(profile, str) else None)
    return ModelSelection(
        model_path=str(config.model_path),
        target=target,
        platform=profile,
        task="detect",
        contract=contract,
        input_shape=getattr(config, "input_shape", None),
        family="yolo26",
    )


def _profile_from_config(config: YOLO26DetectConfig) -> Optional[PlatformProfile]:
    profile = getattr(config, "platform", None)
    if isinstance(profile, str):
        return resolve_platform(profile)
    return profile


def _ltrb_grid(input_size: Tuple[int, int], stride: int) -> np.ndarray:
    height, width = input_size
    if height % stride or width % stride:
        raise BindingError(
            f"Stride {stride} does not divide model input {height}x{width}.")
    grid_height, grid_width = height // stride, width // stride
    grid_y, grid_x = np.indices((grid_height, grid_width), dtype=np.float32)
    return np.stack((grid_x.reshape(-1) + 0.5,
                     grid_y.reshape(-1) + 0.5), axis=-1)


class YOLO26Detect:
    """YOLO26 direct-LTRB detector with an injectable model runner."""

    task = "detect"

    def __init__(self,
                 config: YOLO26DetectConfig,
                 runner: Any = None,
                 model_runner: Any = None,
                 runtime_loader: Any = None) -> None:
        if runner is not None and model_runner is not None:
            raise ValueError("Pass only one of runner or model_runner.")
        if runner is None:
            runner = model_runner

        self.cfg = config
        contract = getattr(config, "contract", None)
        if contract is None:
            contract = LTRBDetectionContract(
                classes=int(config.classes_num),
                strides=tuple(config.strides),
            )
        if getattr(contract, "protocol", None) != "LTRB":
            raise BindingError(
                "YOLO26 detection requires the direct LTRB detection contract.")
        self.contract = contract

        if runner is None:
            if getattr(config, "platform", None) is None:
                # Preserve the legacy constructor's automatic board-profile
                # selection.  Injected host runners remain platform-free.
                config.platform = resolve_platform()
            selection = _ltrb_selection_from_config(config, contract)
            runner = ModelRunner.from_selection(selection, runtime_loader)
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        if self.binding is not None:
            bound_contract = getattr(self.binding, "contract", None)
            if getattr(bound_contract, "protocol", None) != "LTRB":
                raise BindingError(
                    "YOLO26 detection runner is bound to a non-LTRB contract.")
            self.contract = bound_contract
        self.model = getattr(runner, "model", runner)
        self.input_adapter = getattr(runner, "input_adapter", None)
        if self.input_adapter is None and self.binding is not None:
            self.input_adapter = getattr(self.binding, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(self.runner, config)
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError("Model input dimensions must be positive.")
        self.input_size = (self.input_h, self.input_w)
        self.model_name = getattr(self.runner, "model_name", None)
        if self.model_name is None and self.binding is not None:
            self.model_name = getattr(self.binding, "model_name", None)
        self.input_names = tuple(getattr(self.runner, "input_names", ()) or ())
        if not self.input_names and self.input_adapter is not None:
            self.input_names = tuple(getattr(self.input_adapter, "input_names", ()) or ())
        self.output_names = tuple(getattr(self.runner, "output_names", ()) or ())
        self.input_shapes = dict(getattr(self.runner, "input_shapes", {}) or {})
        self.output_shapes = dict(getattr(self.runner, "output_shapes", {}) or {})

        self.anchor_sizes = _normalise_grids(
            getattr(config, "anchor_sizes", None), self.input_size,
            self.contract.strides)
        self.grid_shapes = list(self.anchor_sizes)
        self.cfg.anchor_sizes = list(self.anchor_sizes)
        self.grids = {
            int(stride): _ltrb_grid(self.input_size, int(stride))
            for stride in self.contract.strides
        }
        self.map_idx = {
            int(stride): (index * 2, index * 2 + 1)
            for index, stride in enumerate(self.contract.strides)
        }
        self.conf_raw = self.logit_threshold(self.cfg.score_thres)
        profile = _profile_from_config(config)
        if self.cfg.nms_thres is None and profile is not None:
            self.cfg.nms_thres = float(profile.nms_thres)

    @staticmethod
    def logit_threshold(score: float) -> float:
        if not 0.0 < float(score) < 1.0:
            raise ValueError("score_thres must lie strictly between 0 and 1.")
        return float(-np.log(1.0 / float(score) - 1.0))

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Forward explicit scheduling parameters to the selected runner."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores)

    def preprocess(self, img: np.ndarray, image_format: str = "BGR") -> PreparedDetection:
        """Validate BGR uint8 HxWx3 and return NV12 tensors plus frozen geometry."""
        tensors, transform = _prepare_image(
            self.runner, self.input_adapter, self.input_size,
            self.cfg.resize_type, img, image_format)
        return PreparedDetection(tensors, transform)

    def infer(self, input_tensor: Mapping[str, Any]):
        """Call the injected or factory-created runner exactly once."""
        return _forward_runner(self.runner, input_tensor)

    def postprocess(self,
                    outputs: Any,
                    ori_img_w: Optional[int] = None,
                    ori_img_h: Optional[int] = None,
                    score_thres: Optional[float] = None,
                    nms_thres: Optional[float] = None,
                    transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Decode direct LTRB outputs into owned DetectionResult arrays.

        Supply the matching PreparedDetection.transform, or explicit original
        dimensions for the legacy stateless path. RawOutputs uses its validated
        binding for postprocess transforms; injected semantic mappings must hold
        floating values. Wrong binding/geometry/quantization raises ValueError
        (including BindingError); score/NMS overrides follow the decoder contract.
        """
        semantic = _semantic_outputs(self.binding, self.contract, outputs, "LTRB")
        score = self.cfg.score_thres if score_thres is None else float(score_thres)
        nms = self.cfg.nms_thres if nms_thres is None else float(nms_thres)
        try:
            boxes, scores, class_ids = decode_ltrb(
                semantic,
                self.contract,
                input_size=self.input_size,
                score_thres=score,
                nms_thres=nms,
            )
        except DecodeError as exc:
            raise BindingError(str(exc)) from exc
        concrete = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h,
            self.input_size, self.cfg.resize_type)
        boxes = inverse_boxes(boxes, concrete)
        profile = _profile_from_config(self.cfg)
        if profile is not None and profile.family == "x5" and len(class_ids):
            order = np.argsort(class_ids, kind="stable")
            boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]
        return DetectionResult(boxes, scores, class_ids)

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None) -> DetectionResult:
        """Run preprocessing, one model call and direct-offset postprocessing."""
        prepared = self.preprocess(img, image_format)
        outputs = self.infer(prepared)
        return self.postprocess(
            outputs,
            score_thres=score_thres,
            nms_thres=nms_thres,
            transform=prepared.transform,
        )

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None) -> DetectionResult:
        """Tuple-compatible alias for :meth:`predict`."""
        return self.predict(img, image_format, score_thres, nms_thres)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, img: np.ndarray, image_format: str = "BGR") -> PreparedDetection:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(img, image_format)

    def forward(self, input_tensor: Mapping[str, Any]):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(input_tensor)

    def post_process(self,
                     outputs: Any,
                     ori_img_w: Optional[int] = None,
                     ori_img_h: Optional[int] = None,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(
            outputs, ori_img_w, ori_img_h, score_thres, nms_thres, transform)

# ====================================================================
# Image geometry shared by preprocessing and postprocessing (and by the other YOLO tasks/YOLOE).
# ====================================================================

Size = Tuple[int, int]


Padding = Tuple[int, int, int, int]


def _size(value: Sequence[int], name: str) -> Size:
    """Return a validated ``(height, width)`` pair."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain (height, width).")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain positive dimensions.")
    return height, width


@dataclass(frozen=True)
class ImageTransform:
    """Describe the concrete transform used for one model input.

    Attributes:
        original_size: Source image dimensions as ``(height, width)``.
        model_size: Final model input dimensions as ``(height, width)``.
        resized_size: Actual resized image dimensions before padding.
        padding: Actual integer padding as ``(left, top, right, bottom)``.
        scale_x: Actual horizontal scale from source pixels to resized pixels.
        scale_y: Actual vertical scale from source pixels to resized pixels.
        crop_offset: Source coordinate offset when a crop was applied.  It is
            normally ``(0, 0)`` and is retained for task pipelines that crop
            before resizing.
        resize_type: ``0`` for stretch and ``1`` for letterbox.
    """

    original_size: Size
    model_size: Size
    resized_size: Size
    padding: Padding
    scale_x: float
    scale_y: float
    crop_offset: Tuple[float, float] = (0.0, 0.0)
    resize_type: int = 1

    def __post_init__(self) -> None:
        original = _size(self.original_size, "original_size")
        model = _size(self.model_size, "model_size")
        resized = _size(self.resized_size, "resized_size")
        if len(self.padding) != 4 or any(int(v) < 0 for v in self.padding):
            raise ValueError("padding must contain four non-negative integers.")
        padding = tuple(int(v) for v in self.padding)
        if resized[0] + padding[1] + padding[3] != model[0]:
            raise ValueError("vertical resize and padding do not fill model_size.")
        if resized[1] + padding[0] + padding[2] != model[1]:
            raise ValueError("horizontal resize and padding do not fill model_size.")
        if self.resize_type not in (0, 1):
            raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox).")
        if not np.isfinite(self.scale_x) or not np.isfinite(self.scale_y):
            raise ValueError("scale_x and scale_y must be finite.")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("scale_x and scale_y must be positive.")
        if len(self.crop_offset) != 2:
            raise ValueError("crop_offset must contain (x, y).")
        object.__setattr__(self, "original_size", original)
        object.__setattr__(self, "model_size", model)
        object.__setattr__(self, "resized_size", resized)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "crop_offset", (float(self.crop_offset[0]),
                                                   float(self.crop_offset[1])))
        object.__setattr__(self, "scale_x", float(self.scale_x))
        object.__setattr__(self, "scale_y", float(self.scale_y))

    @property
    def target_size(self) -> Size:
        """Alias for the final model input size."""
        return self.model_size

    @property
    def input_size(self) -> Size:
        """Alias for the final model input size."""
        return self.model_size

    @property
    def original_height(self) -> int:
        return self.original_size[0]

    @property
    def original_width(self) -> int:
        return self.original_size[1]

    @property
    def model_height(self) -> int:
        return self.model_size[0]

    @property
    def model_width(self) -> int:
        return self.model_size[1]

    @property
    def pad_left(self) -> int:
        return self.padding[0]

    @property
    def pad_top(self) -> int:
        return self.padding[1]

    @property
    def pad_right(self) -> int:
        return self.padding[2]

    @property
    def pad_bottom(self) -> int:
        return self.padding[3]

    @property
    def scale(self) -> Tuple[float, float]:
        """Return actual ``(scale_x, scale_y)`` values."""
        return self.scale_x, self.scale_y


def make_transform(original_size: Sequence[int],
                   model_size: Sequence[int],
                   resize_type: int = 1) -> ImageTransform:
    """Calculate a transform using the same integer rounding as preprocessing.

    Letterbox dimensions intentionally use truncation, matching the existing
    sample's ``int(original * scale)`` behavior.  The returned scales are
    calculated from those actual dimensions, rather than from the ideal scale.
    """
    original = _size(original_size, "original_size")
    model = _size(model_size, "model_size")
    if resize_type == 0:
        resized = model
        padding: Padding = (0, 0, 0, 0)
    elif resize_type == 1:
        source_h, source_w = original
        target_h, target_w = model
        ideal_scale = min(target_h / source_h, target_w / source_w)
        resized_h = max(1, min(target_h, int(source_h * ideal_scale)))
        resized_w = max(1, min(target_w, int(source_w * ideal_scale)))
        resized = (resized_h, resized_w)
        pad_w = target_w - resized_w
        pad_h = target_h - resized_h
        padding = (pad_w // 2, pad_h // 2,
                   pad_w - pad_w // 2, pad_h - pad_h // 2)
    else:
        raise ValueError("resize_type must be 0 (stretch) or 1 (letterbox).")

    return ImageTransform(
        original_size=original,
        model_size=model,
        resized_size=resized,
        padding=padding,
        scale_x=resized[1] / original[1],
        scale_y=resized[0] / original[0],
        resize_type=resize_type,
    )


def resize_with_transform(image: np.ndarray,
                          model_size: Sequence[int],
                          resize_type: int = 1,
                          interpolation: Optional[int] = None,
                          pad_value=(127, 127, 127)) -> Tuple[np.ndarray, ImageTransform]:
    """Resize an image and return both pixels and the concrete transform.

    ``model_size`` is ``(height, width)``.  The interpolation defaults retain
    the old sample behavior: nearest-neighbor for stretch and OpenCV's linear
    interpolation for the letterbox resize.
    """
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("image must be a NumPy array with at least two dimensions.")
    transform = make_transform(image.shape[:2], model_size, resize_type)
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST if resize_type == 0 else cv2.INTER_LINEAR
    resized_h, resized_w = transform.resized_size
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    left, top, right, bottom = transform.padding
    if any((left, top, right, bottom)):
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    if tuple(resized.shape[:2]) != transform.model_size:
        raise ValueError(
            f"preprocessing produced {resized.shape[:2]}, expected {transform.model_size}.")
    return resized, transform


def inverse_boxes(boxes: np.ndarray,
                  transform: ImageTransform,
                  clip: bool = True) -> np.ndarray:
    """Map ``xyxy`` boxes from model-input coordinates to source coordinates."""
    array = np.asarray(boxes)
    if array.size == 0:
        if array.ndim == 1:
            return np.empty((0, 4), dtype=np.float32)
        if array.shape[-1:] != (4,):
            raise ValueError("boxes must have a final dimension of four.")
        return np.empty(array.shape, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4).")
    result = np.asarray(array, dtype=np.float32).copy()
    left, top, _, _ = transform.padding
    crop_x, crop_y = transform.crop_offset
    result[:, [0, 2]] = (result[:, [0, 2]] - left) / transform.scale_x + crop_x
    result[:, [1, 3]] = (result[:, [1, 3]] - top) / transform.scale_y + crop_y
    if clip:
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, transform.original_width)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, transform.original_height)
    return result


def restore_boxes(boxes: np.ndarray,
                  transform: ImageTransform,
                  clip: bool = True) -> np.ndarray:
    """Compatibility alias for :func:`inverse_boxes`."""
    return inverse_boxes(boxes, transform, clip=clip)


def scale_boxes_to_original(boxes: np.ndarray,
                            transform: ImageTransform,
                            clip: bool = True) -> np.ndarray:
    """Descriptive alias used by callers that prefer a scaling name."""
    return inverse_boxes(boxes, transform, clip=clip)


def inverse_points(points: np.ndarray, transform: ImageTransform) -> np.ndarray:
    """Map finite (...,2) model points through the actual resize/padding.

    Return owned float32 original-image coordinates clipped to inclusive image
    bounds. Unlike confidence values, coordinates undergo geometry exactly once.
    """
    values = np.asarray(points, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError("points must be finite coordinates with shape (...,2).")
    result = values.copy()
    result[..., 0] = (result[..., 0] - transform.padding[0]) / transform.scale_x + transform.crop_offset[0]
    result[..., 1] = (result[..., 1] - transform.padding[1]) / transform.scale_y + transform.crop_offset[1]
    result[..., 0] = np.clip(result[..., 0], 0, transform.original_size[1])
    result[..., 1] = np.clip(result[..., 1], 0, transform.original_size[0])
    return result

# ====================================================================
# Numeric decode: DFL and direct-LTRB head decoding with NMS.
# ====================================================================

class DecodeError(ValueError):
    """Raised when semantic DFL tensors cannot be decoded safely."""


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used for classification logits."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax used for DFL logits."""
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values - maximum)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64))


def _contract_value(contract: Any, name: str, default: Any = None) -> Any:
    if contract is None:
        return default
    return getattr(contract, name, default)


def _role(outputs: Mapping[str, Any], kind: str, stride: int) -> Any:
    canonical = f"{kind}_{int(stride)}"
    if canonical not in outputs:
        raise DecodeError(f"Semantic output role {canonical!r} is missing.")
    return outputs[canonical]


def _shape_from_input(input_size: Optional[Sequence[int]], stride: int) -> Optional[Tuple[int, int]]:
    if input_size is None:
        return None
    if len(input_size) != 2:
        raise DecodeError("input_size must contain (height, width).")
    height, width = int(input_size[0]), int(input_size[1])
    if height <= 0 or width <= 0 or height % stride or width % stride:
        raise DecodeError(
            f"Stride {stride} does not divide model input {height}x{width}.")
    return height // stride, width // stride


def _as_nhwc(value: Any,
             *,
             grid: Optional[Tuple[int, int]],
             channels: int,
             role: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Normalize one physical/semantic tensor to ``(H, W, C)``."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise DecodeError(f"{role} must be a floating semantic tensor, got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise DecodeError(f"{role} contains NaN or infinity.")
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise DecodeError(f"{role} must have batch size one, got {array.shape}.")
        if array.shape[-1] == channels:
            result = array[0]
        else:
            raise DecodeError(
                f"{role} shape {array.shape} is not the required NHWC tensor with "
                f"{channels} channels.")
        actual_grid = (int(result.shape[0]), int(result.shape[1]))
    else:
        raise DecodeError(f"{role} shape {array.shape} is unsupported.")
    if grid is not None and actual_grid != grid:
        raise DecodeError(f"{role} grid {actual_grid} conflicts with declared grid {grid}.")
    return np.asarray(result, dtype=np.float32), actual_grid


def _probabilities(values: np.ndarray, mode: str, role: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "logits":
        return sigmoid(values)
    if mode != "probabilities":
        raise DecodeError(f"Unsupported {role} activation mode {mode!r}.")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-5) or np.any(values > 1.00001):
        raise DecodeError(f"{role} declares probabilities but contains values outside [0, 1].")
    return values


def _dfl_offsets(values: np.ndarray, mode: str, bins: int) -> np.ndarray:
    values = values.reshape(-1, 4, bins)
    mode = str(mode).lower()
    if mode == "logits":
        probabilities = softmax(values, axis=-1)
    elif mode == "probabilities":
        if not np.all(np.isfinite(values)) or np.any(values < -1e-5):
            raise DecodeError("DFL probabilities must be finite and non-negative.")
        sums = values.sum(axis=-1, keepdims=True)
        if np.any(sums <= 0):
            raise DecodeError("DFL probability distributions must have positive mass.")
        probabilities = values / sums
    else:
        raise DecodeError(f"Unsupported DFL distribution mode {mode!r}.")
    bins_values = np.arange(bins, dtype=np.float32).reshape(1, 1, bins)
    return np.sum(probabilities * bins_values, axis=-1)


def _bound_outputs(outputs: Any, contract: Any, binding: Any,
                   protocol: str) -> Tuple[Mapping[str, Any], Any]:
    """Resolve one physical binding before either protocol decodes it."""
    if binding is not None:
        try:
            outputs = binding.read_outputs(outputs)
            contract = getattr(binding, "contract", contract)
        except Exception as exc:
            if isinstance(exc, DecodeError):
                raise
            raise DecodeError(str(exc)) from exc
    if not isinstance(outputs, Mapping):
        raise DecodeError(
            f"{protocol} outputs must be a mapping keyed by semantic roles.")
    return outputs, contract


def _decode_options(contract: Any,
                    *,
                    strides: Optional[Sequence[int]],
                    classes_num: Optional[int],
                    score_thres: float,
                    nms_thres: Optional[float],
                    nms: Optional[str],
                    protocol: str) -> Tuple[Tuple[int, ...], int, str, Optional[float]]:
    values = tuple(int(value) for value in (strides or _contract_value(
        contract, "strides", (8, 16, 32))))
    classes = int(classes_num if classes_num is not None else _contract_value(
        contract, "classes", 80))
    nms_mode = str(nms if nms is not None else _contract_value(
        contract, "nms", "classwise")).lower()
    if classes <= 0 or not values or any(value <= 0 for value in values):
        raise DecodeError(f"{protocol} classes and strides must be positive.")
    if not 0.0 <= float(score_thres) <= 1.0:
        raise DecodeError("score_thres must be between 0 and 1.")
    if nms_mode not in {"none", "classwise", "agnostic"}:
        raise DecodeError(f"Unsupported NMS policy {nms_mode!r}.")
    if nms_mode != "none":
        if nms_thres is None:
            raise DecodeError(f"NMS policy {nms_mode!r} requires nms_thres.")
        if not 0.0 <= float(nms_thres) <= 1.0:
            raise DecodeError("nms_thres must be between 0 and 1.")
    return values, classes, nms_mode, nms_thres


def _apply_nms(boxes: np.ndarray,
               scores: np.ndarray,
               class_ids: np.ndarray,
               nms_mode: str,
               nms_thres: Optional[float]) -> np.ndarray:
    if nms_mode == "none":
        return np.arange(len(boxes), dtype=np.int64)
    from utils.py_utils.postprocess import NMS
    nms_ids = np.zeros_like(class_ids) if nms_mode == "agnostic" else class_ids
    return np.asarray(NMS(boxes, scores, nms_ids, float(nms_thres)), dtype=np.int64)


def _decode_heads(outputs: Mapping[str, Any],
                  contract: Any,
                  *,
                  input_size: Optional[Sequence[int]],
                  score_thres: float,
                  nms_thres: Optional[float],
                  nms: Optional[str],
                  strides: Optional[Sequence[int]],
                  classes_num: Optional[int],
                  box_channels: int,
                  offset_decoder: Any,
                  protocol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    strides, classes, nms_mode, nms_thres = _decode_options(
        contract, strides=strides, classes_num=classes_num,
        score_thres=score_thres, nms_thres=nms_thres, nms=nms,
        protocol=protocol)
    all_boxes = []
    all_scores = []
    all_ids = []
    for stride in strides:
        grid = _shape_from_input(input_size, stride)
        cls_value = _role(outputs, "cls", stride)
        box_value = _role(outputs, "box", stride)
        cls_array, cls_grid = _as_nhwc(
            cls_value, grid=grid, channels=classes, role=f"cls_{stride}")
        box_array, box_grid = _as_nhwc(
            box_value, grid=grid, channels=box_channels, role=f"box_{stride}")
        if cls_grid != box_grid:
            raise DecodeError(
                f"Classification grid {cls_grid} and box grid {box_grid} differ at stride {stride}.")
        cls_flat = cls_array.reshape(-1, classes)
        box_flat = box_array.reshape(-1, box_channels)
        classification = str(_contract_value(
            contract, "classification", "logits")).lower()
        cls_prob = _probabilities(
            cls_flat, classification, f"cls_{stride}")
        # Choose the class in the source logit domain.  Applying sigmoid first
        # can round large logits to the same float32 value and change the
        # selected class even though sigmoid is monotonic.
        ids = np.argmax(cls_flat, axis=1).astype(np.int64)
        scores = cls_prob[np.arange(cls_prob.shape[0]), ids]
        if classification == "logits":
            threshold = float(score_thres)
            if threshold <= 0.0:
                raw_threshold = -np.inf
            elif threshold >= 1.0:
                raw_threshold = np.inf
            else:
                raw_threshold = -np.log(1.0 / threshold - 1.0)
            raw_scores = cls_flat[np.arange(cls_flat.shape[0]), ids]
            valid = raw_scores >= raw_threshold
        else:
            valid = scores >= float(score_thres)
        if not np.any(valid):
            continue
        offsets = np.asarray(offset_decoder(box_flat[valid]), dtype=np.float32)
        if offsets.shape != (int(np.count_nonzero(valid)), 4):
            raise DecodeError(
                f"{protocol} box decoder returned shape {offsets.shape}; expected four offsets.")
        if not np.all(np.isfinite(offsets)):
            raise DecodeError(f"{protocol} box offsets contain NaN or infinity.")
        gh, gw = cls_grid
        grid_y, grid_x = np.indices((gh, gw), dtype=np.float32)
        anchors = np.stack((grid_x.reshape(-1) + 0.5,
                            grid_y.reshape(-1) + 0.5), axis=-1)[valid]
        selected_boxes = np.concatenate((
            anchors - offsets[:, :2], anchors + offsets[:, 2:]), axis=1
        ) * float(stride)
        if not np.all(np.isfinite(selected_boxes)):
            raise DecodeError(f"{protocol} decoded boxes contain NaN or infinity.")
        all_boxes.append(selected_boxes.astype(np.float32, copy=False))
        all_scores.append(scores[valid].astype(np.float32, copy=False))
        all_ids.append(ids[valid])

    if not all_boxes:
        return _empty()
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_ids, axis=0)
    keep = _apply_nms(boxes, scores, class_ids, nms_mode, nms_thres)
    return boxes[keep], scores[keep], class_ids[keep]


def _direct_ltrb_offsets(values: np.ndarray) -> np.ndarray:
    """Keep already-decoded LTRB distances in grid-cell units."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise DecodeError(
            f"LTRB outputs must provide four direct offsets, got {values.shape}.")
    return values


def decode_dfl(outputs: Mapping[str, Any],
               contract: Any = None,
               *,
               strides: Optional[Sequence[int]] = None,
               reg_bins: Optional[int] = None,
               classes_num: Optional[int] = None,
               input_size: Optional[Sequence[int]] = None,
               score_thres: float = 0.25,
               nms_thres: Optional[float] = None,
               nms: Optional[str] = None,
               binding: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode named DFL outputs while preserving the established API."""
    outputs, contract = _bound_outputs(outputs, contract, binding, "DFL")
    if str(_contract_value(contract, "protocol", "DFL")) != "DFL":
        raise DecodeError("decode_dfl received a non-DFL detection contract.")
    bins = int(reg_bins if reg_bins is not None else _contract_value(
        contract, "reg_bins", 16))
    if bins <= 0:
        raise DecodeError("DFL bins must be positive.")
    box_mode = _contract_value(contract, "box_distribution", "logits")
    return _decode_heads(
        outputs, contract, input_size=input_size, score_thres=score_thres,
        nms_thres=nms_thres, nms=nms, strides=strides,
        classes_num=classes_num, box_channels=4 * bins,
        offset_decoder=lambda values: _dfl_offsets(values, box_mode, bins),
        protocol="DFL",
    )


def decode_ltrb(outputs: Mapping[str, Any],
                contract: Any,
                *,
                input_size: Optional[Sequence[int]] = None,
                score_thres: float = 0.25,
                nms_thres: Optional[float] = None,
                binding: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the reviewed YOLO26 direct four-channel LTRB outputs."""
    outputs, contract = _bound_outputs(outputs, contract, binding, "LTRB")
    if str(_contract_value(contract, "protocol", "DFL")) != "LTRB":
        raise DecodeError("decode_ltrb requires an LTRB detection contract.")
    return _decode_heads(
        outputs, contract, input_size=input_size, score_thres=score_thres,
        nms_thres=nms_thres, nms=None, strides=None, classes_num=None,
        box_channels=4, offset_decoder=_direct_ltrb_offsets, protocol="LTRB",
    )

# ====================================================================
# Shared detection transport: prepared NV12 input, one runner call, semantic output resolution, per-call transform and scheduling.
# ====================================================================

def _read_source_image(source: "str | Path | np.ndarray") -> np.ndarray:
    """Accept one local image path or an in-memory BGR array.

    Path read failures name the exact path.  Arrays pass through
    unchanged (never modified in place); their shape/dtype validation
    happens in :meth:`YoloDetect.preprocess`.
    """

    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        from utils.py_utils import (
            file_io,
        )

        return file_io.load_image(str(source))
    raise TypeError(
        "source must be an image path or a BGR NumPy array, got "
        f"{type(source).__name__}.")


@dataclass(frozen=True)
class PreparedDetection(Mapping[str, Mapping[str, np.ndarray]]):
    """Owned NV12 tensors and immutable geometry for exactly one BGR image.

    Use .tensors for forward and .transform for post_process. Mapping access
    preserves historical pre_process(...)[model_name] transport access; no last
    image is cached on the task. SDK execution is not promised thread-safe.
    """

    tensors: Mapping[str, Mapping[str, np.ndarray]]
    transform: ImageTransform

    def __getitem__(self, key):
        return self.tensors[key]

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)


class DetectionResult(NamedTuple):
    """Owned detection arrays, also unpackable as (boxes, scores, class_ids).

    boxes_xyxy: float32 (N,4), original-image continuous xyxy pixel coordinates
    clipped to [0,width]/[0,height]. scores: float32 (N,) probabilities in [0,1].
    class_ids: int64 (N,) zero-based labels. Empty results keep those dtypes and
    ranks. Ordering follows the selected decoder/NMS policy, not a new global sort.
    """

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    @property
    def boxes(self) -> np.ndarray:
        return self.boxes_xyxy

    @property
    def cls_ids(self) -> np.ndarray:
        return self.class_ids


def _size_from_runner(runner: Any, config: Any) -> Tuple[int, int]:
    value = getattr(runner, "input_size", None)
    if value is not None:
        if len(value) != 2:
            raise ValueError("runner.input_size must be (height, width).")
        return int(value[0]), int(value[1])
    height = getattr(runner, "input_height", None)
    width = getattr(runner, "input_width", None)
    if height is not None and width is not None:
        return int(height), int(width)
    if config.input_shape is not None:
        return int(config.input_shape[0]), int(config.input_shape[1])
    raise ValueError(
        "The injected runner must expose input_size or input_height/input_width."
    )


def _normalise_grids(
    anchor_sizes: Optional[Sequence[Any]],
    input_size: Tuple[int, int],
    strides: Sequence[int],
) -> list:
    expected = []
    for stride in strides:
        stride = int(stride)
        if stride <= 0 or input_size[0] % stride or input_size[1] % stride:
            raise ValueError(
                f"Stride {stride} does not divide model input {input_size[0]}x{input_size[1]}."
            )
        expected.append((input_size[0] // stride, input_size[1] // stride))
    if anchor_sizes is None:
        return expected
    if len(anchor_sizes) != len(expected):
        raise ValueError("anchor_sizes must contain one grid per stride.")
    actual = []
    for value in anchor_sizes:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError(
                    "Each rectangular anchor grid must be (height, width)."
                )
            actual.append((int(value[0]), int(value[1])))
        else:
            actual.append((int(value), int(value)))
    if actual != expected:
        raise ValueError(
            f"anchor_sizes {actual} conflict with model input and strides {expected}."
        )
    return actual


def _build_input(
    runner: Any, input_adapter: Any, y_plane: np.ndarray, uv_plane: np.ndarray
):
    """Build one bound NV12 input through the selected runner."""
    method = getattr(runner, "prepare_input", None)
    if method is not None:
        return method(y_plane, uv_plane)
    if input_adapter is None:
        raise BindingError("The injected runner has no input adapter.")
    return input_adapter.build(y_plane, uv_plane)


def _prepare_image(
    runner: Any,
    input_adapter: Any,
    input_size: Tuple[int, int],
    resize_type: int,
    img: np.ndarray,
    image_format: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], ImageTransform]:
    """Share image validation, resize bookkeeping and NV12 transport."""
    if str(image_format).upper() != "BGR":
        raise ValueError(f"Unsupported image_format: {image_format}")
    if (
        not isinstance(img, np.ndarray)
        or img.ndim != 3
        or img.shape[2] != 3
        or img.dtype != np.uint8
        or min(img.shape[:2]) <= 0
    ):
        raise ValueError("img must be a nonempty BGR uint8 HxWx3 NumPy array.")
    resized, transform = resize_with_transform(img, input_size, resize_type=resize_type)
    y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(resized)
    return _build_input(runner, input_adapter, y_plane, uv_plane), transform


def _forward_runner(runner: Any, input_tensor: Mapping[str, Any]):
    """Call one injected or factory-created runner exactly once."""
    if isinstance(input_tensor, PreparedDetection):
        input_tensor = input_tensor.tensors
    if callable(runner):
        return runner(input_tensor)
    method = getattr(runner, "forward", None) or getattr(runner, "run", None)
    if method is None:
        raise BindingError(
            "The injected runner is not callable and has no forward/run method."
        )
    return method(input_tensor)


def _semantic_outputs(
    binding: Any, contract: Any, outputs: Any, protocol: str
) -> Mapping[str, Any]:
    """Use named semantic outputs or the already validated physical binding."""

    if isinstance(outputs, RawOutputs):
        expected = getattr(binding, "output_adapter", None)
        if expected is not None and outputs.binding is not expected:
            raise BindingError("Raw outputs belong to a different model binding.")
        try:
            return outputs.binding.read(outputs)
        except ValueError as exc:
            raise BindingError(str(exc)) from exc
    if isinstance(outputs, Mapping):
        required = set(contract.required_roles)
        if required.issubset(set(outputs)):
            return outputs
    if binding is not None:
        reader = getattr(binding, "read_outputs", None)
        if reader is not None:
            try:
                return reader(outputs)
            except Exception as exc:
                raise BindingError(str(exc)) from exc
    raise BindingError(
        f"Detector outputs are not keyed by semantic roles and no complete {protocol} "
        "output binding is available."
    )


def _transform_for_postprocess(
    transform: Optional[ImageTransform],
    ori_img_w: Optional[int],
    ori_img_h: Optional[int],
    input_size: Tuple[int, int],
    resize_type: int,
) -> ImageTransform:
    """Use per-call context, or reconstruct from explicit legacy dimensions."""
    if transform is not None:
        if (
            not isinstance(transform, ImageTransform)
            or transform.model_size != input_size
        ):
            raise ValueError("Image transform does not match the model input geometry.")
        if (ori_img_w is not None and int(ori_img_w) != transform.original_size[1]) or (
            ori_img_h is not None and int(ori_img_h) != transform.original_size[0]
        ):
            raise ValueError(
                "Image transform conflicts with explicit original dimensions."
            )
        return transform
    if ori_img_w is None or ori_img_h is None:
        raise ValueError(
            "post_process requires prepared.transform or both original dimensions."
        )
    return make_transform((ori_img_h, ori_img_w), input_size, resize_type)


def _set_scheduling_params(
    runner: Any,
    model: Any,
    model_name: Optional[str],
    priority: Optional[int],
    bpu_cores: Optional[list],
) -> None:
    """Forward explicit scheduler settings and reject unsupported requests."""
    method = getattr(runner, "set_scheduling_params", None)
    if method is not None:
        method(priority=priority, bpu_cores=bpu_cores)
        return
    method = getattr(model, "set_scheduling_params", None)
    if method is None:
        if priority is not None or bpu_cores is not None:
            raise BindingError(
                "The injected runner/runtime does not support explicit scheduling parameters."
            )
        return
    values: Dict[str, Any] = {}
    if priority is not None:
        values["priority"] = {model_name: priority}
    if bpu_cores is not None:
        values["bpu_cores"] = {model_name: bpu_cores}
    if values:
        method(**values)

__all__ = [
    "DecodeError",
    "DetectionResult",
    "ImageTransform",
    "PreparedDetection",
    "YOLO26Detect",
    "YOLO26DetectConfig",
    "YoloDetect",
    "YoloDetectConfig",
    "YoloV10Detect",
    "YoloV10DetectConfig",
    "decode_dfl",
    "decode_ltrb",
    "inverse_boxes",
    "inverse_points",
    "make_transform",
    "resize_with_transform",
    "restore_boxes",
    "scale_boxes_to_original",
    "sigmoid",
    "softmax",
]
