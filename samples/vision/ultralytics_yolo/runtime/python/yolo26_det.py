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

"""YOLO26 direct-offset detection task.

YOLO26 detection uses the same image transport and runner boundary as the
reviewed DFL detector, but its box heads are already four-channel LTRB
distances.  The protocol-specific contract and decoder stay explicit here;
the other YOLO26 tasks continue to use their existing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python.decode import (
    DecodeError,
    decode_ltrb,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    LTRBDetectionContract,
    ModelSelection,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import (
    DetectionResult,
    _forward_runner,
    _normalise_grids,
    _prepare_image,
    _predict_task,
    _semantic_outputs,
    _set_scheduling_params,
    _size_from_runner,
    _transform_for_postprocess,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
    PlatformProfile,
    resolve_platform,
)


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


def _selection_from_config(config: YOLO26DetectConfig,
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
            selection = _selection_from_config(config, contract)
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
        self.last_transform: Optional[ImageTransform] = None
        self.last_image_transform: Optional[ImageTransform] = None

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

    def pre_process_with_transform(
            self,
            img: np.ndarray,
            image_format: str = "BGR") -> Tuple[Dict[str, Dict[str, np.ndarray]], ImageTransform]:
        tensors, transform = _prepare_image(
            self.runner, self.input_adapter, self.input_size,
            self.cfg.resize_type, img, image_format)
        self.last_transform = transform
        self.last_image_transform = transform
        return tensors, transform

    def pre_process(self,
                    img: np.ndarray,
                    image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare one image using the legacy tensor-only return shape."""
        tensors, _ = self.pre_process_with_transform(img, image_format)
        return tensors

    def forward(self, input_tensor: Mapping[str, Any]):
        """Call the injected or factory-created runner exactly once."""
        return _forward_runner(self.runner, input_tensor)

    def _semantic_outputs(self, outputs: Any) -> Mapping[str, Any]:
        return _semantic_outputs(self.binding, self.contract, outputs, "LTRB")

    def post_process(self,
                     outputs: Any,
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Decode direct LTRB heads and map boxes to original image pixels."""
        semantic = self._semantic_outputs(outputs)
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
        concrete = transform or _transform_for_postprocess(
            self.last_transform, ori_img_w, ori_img_h,
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
        return _predict_task(self, img, image_format, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None) -> DetectionResult:
        """Tuple-compatible alias for :meth:`predict`."""
        return self.predict(img, image_format, score_thres, nms_thres)


__all__ = ["YOLO26DetectConfig", "YOLO26Detect"]
