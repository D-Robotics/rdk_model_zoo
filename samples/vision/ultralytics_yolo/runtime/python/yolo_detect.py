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

"""Reusable YOLO DFL detection task.

The task owns image geometry, decoding and the public result.  A
``ModelRunner`` (or any compatible callable) owns model execution and may be
injected for host tests or another supported runtime.  The legacy constructor
still builds a runner from ``YoloDetectConfig`` when one is not injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

# The repository-root bootstrap belongs to the entrypoints.  Keeping the
# algorithm package-qualified avoids collisions with another sample's
# ``model_runner`` or ``tensor_io`` module when a compatibility wrapper imports
# this task from an arbitrary working directory.
from samples.vision.ultralytics_yolo.runtime.python.decode import decode_dfl
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    default_dfl_contract,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile


from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    DetectionResult, PreparedDetection, _size_from_runner, _normalise_grids,
    _prepare_image, _forward_runner, _semantic_outputs, _transform_for_postprocess,
    _set_scheduling_params, _predict_task,
)
from samples.vision.ultralytics_yolo.runtime.python.legacy import pre_process_with_transform


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
    """DFL detector with a replaceable model execution boundary."""

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

    def pre_process(self, img: np.ndarray, image_format: str = "BGR") -> PreparedDetection:
        """Validate BGR uint8 HxWx3 and return NV12 tensors plus frozen geometry."""
        tensors, transform = _prepare_image(
            self.runner, self.input_adapter, self.input_size,
            self.cfg.resize_type, img, image_format)
        return PreparedDetection(tensors, transform)

    def forward(self, input_tensor: Mapping[str, Any]):
        """Call the injected runner exactly once."""
        return _forward_runner(self.runner, input_tensor)

    def post_process(self,
                     outputs: Any,
                     ori_img_w: Optional[int] = None,
                     ori_img_h: Optional[int] = None,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Decode DFL outputs into owned DetectionResult arrays.

        Supply the matching PreparedDetection.transform, or explicit original
        dimensions for the legacy stateless path. RawOutputs uses its validated
        binding for postprocess transforms; injected semantic mappings must hold
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
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None) -> DetectionResult:
        """Run preprocessing, one model call, and postprocessing."""
        return _predict_task(self, img, image_format, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None) -> DetectionResult:
        """Tuple-compatible alias for :meth:`predict`."""
        return self.predict(img, image_format, score_thres, nms_thres)


__all__ = ["DetectionResult", "YoloDetectConfig", "YoloDetect"]
