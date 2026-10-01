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

"""The readable YOLO DFL detection model.

:class:`YoloDetect` shows one complete detection pipeline in a single
file: construction resolves the runner and binding, :meth:`preprocess`
turns one BGR image (or local image path) into the bound NV12 tensors
plus this call's resize geometry, :meth:`infer` executes exactly one
model call, :meth:`postprocess` decodes the DFL outputs and restores the
boxes to the original image, and :meth:`predict` chains the three steps.

The reusable pieces stay shared: DFL decoding lives in ``decode.py``,
box geometry in ``geometry.py``, tensor contracts in ``model_binding.py``
and the NV12 transport in ``detection_io.py``/``rdk_yolo_utils``.  The
other protocols keep their own task classes — ``yolo26_det.py`` (direct
LTRB), ``yolo_v10detect.py`` (NMS-free S YOLOv10) and the cls/seg/pose/
obb task modules; they are deliberately not folded into this class.

Minimal library use::

    from samples.vision.ultralytics_yolo.runtime.python.detect import (
        YoloDetect, YoloDetectConfig)
    from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
        resolve_platform)

    config = YoloDetectConfig(
        "model/yolo11n_detect_bayese_640x640_nv12.bin",
        platform=resolve_platform("x5"),
    )
    model = YoloDetect(config)
    result = model.predict("test_data/bus.jpg")   # path or BGR ndarray
    boxes, scores, class_ids = result             # DetectionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.decode import decode_dfl
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
    resize_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    default_dfl_contract,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile

from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    DetectionResult,
    PreparedDetection,
    _build_input,
    _normalise_grids,
    _semantic_outputs,
    _set_scheduling_params,
    _size_from_runner,
    _transform_for_postprocess,
)
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
    preprocess as pre_utils,
)
from samples.vision.ultralytics_yolo.runtime.python.legacy import (
    pre_process_with_transform,
)


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

    pre_process_with_transform = pre_process_with_transform

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


def _read_source_image(source: "str | Path | np.ndarray") -> np.ndarray:
    """Accept one local image path or an in-memory BGR array.

    Path read failures name the exact path.  Arrays pass through
    unchanged (never modified in place); their shape/dtype validation
    happens in :meth:`YoloDetect.preprocess`.
    """

    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
            file_io,
        )

        return file_io.load_image(str(source))
    raise TypeError(
        "source must be an image path or a BGR NumPy array, got "
        f"{type(source).__name__}.")


__all__ = ["DetectionResult", "YoloDetectConfig", "YoloDetect"]
