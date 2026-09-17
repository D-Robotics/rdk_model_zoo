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
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# The repository-root bootstrap belongs to the entrypoints.  Keeping the
# algorithm package-qualified avoids collisions with another sample's
# ``model_runner`` or ``tensor_io`` module when a compatibility wrapper imports
# this task from an arbitrary working directory.
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import preprocess as pre_utils
from samples.vision.ultralytics_yolo.runtime.python.decode import decode_dfl
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    inverse_boxes,
    make_transform,
    resize_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    default_dfl_contract,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile


class DetectionResult(NamedTuple):
    """Tuple-compatible public detector result."""

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    @property
    def boxes(self) -> np.ndarray:
        return self.boxes_xyxy

    @property
    def cls_ids(self) -> np.ndarray:
        return self.class_ids


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


def _size_from_runner(runner: Any,
                      config: YoloDetectConfig) -> Tuple[int, int]:
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
        "The injected runner must expose input_size or input_height/input_width.")


def _normalise_grids(anchor_sizes: Optional[Sequence[Any]],
                     input_size: Tuple[int, int],
                     strides: Sequence[int]) -> list:
    expected = []
    for stride in strides:
        stride = int(stride)
        if stride <= 0 or input_size[0] % stride or input_size[1] % stride:
            raise ValueError(
                f"Stride {stride} does not divide model input {input_size[0]}x{input_size[1]}.")
        expected.append((input_size[0] // stride, input_size[1] // stride))
    if anchor_sizes is None:
        return expected
    if len(anchor_sizes) != len(expected):
        raise ValueError("anchor_sizes must contain one grid per stride.")
    actual = []
    for value in anchor_sizes:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("Each rectangular anchor grid must be (height, width).")
            actual.append((int(value[0]), int(value[1])))
        else:
            actual.append((int(value), int(value)))
    if actual != expected:
        raise ValueError(
            f"anchor_sizes {actual} conflict with model input and strides {expected}.")
    return actual


def _build_input(runner: Any,
                 input_adapter: Any,
                 y_plane: np.ndarray,
                 uv_plane: np.ndarray):
    """Build one bound NV12 input through the selected runner."""
    method = getattr(runner, "prepare_input", None)
    if method is not None:
        return method(y_plane, uv_plane)
    if input_adapter is None:
        raise BindingError("The injected runner has no input adapter.")
    return input_adapter.build(y_plane, uv_plane)


def _prepare_image(runner: Any,
                   input_adapter: Any,
                   input_size: Tuple[int, int],
                   resize_type: int,
                   img: np.ndarray,
                   image_format: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], ImageTransform]:
    """Share image validation, resize bookkeeping and NV12 transport."""
    if str(image_format).upper() != "BGR":
        raise ValueError(f"Unsupported image_format: {image_format}")
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be a BGR HxWx3 NumPy array.")
    resized, transform = resize_with_transform(
        img, input_size, resize_type=resize_type)
    y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(resized)
    return _build_input(runner, input_adapter, y_plane, uv_plane), transform


def _forward_runner(runner: Any, input_tensor: Mapping[str, Any]):
    """Call one injected or factory-created runner exactly once."""
    if callable(runner):
        return runner(input_tensor)
    method = getattr(runner, "forward", None) or getattr(runner, "run", None)
    if method is None:
        raise BindingError("The injected runner is not callable and has no forward/run method.")
    return method(input_tensor)


def _semantic_outputs(binding: Any,
                      contract: Any,
                      outputs: Any,
                      protocol: str) -> Mapping[str, Any]:
    """Use named semantic outputs or the already validated physical binding."""
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
        "output binding is available.")


def _transform_for_postprocess(last_transform: Optional[ImageTransform],
                               ori_img_w: int,
                               ori_img_h: int,
                               input_size: Tuple[int, int],
                               resize_type: int) -> ImageTransform:
    """Return the concrete transform used for one image or legacy postprocess."""
    if last_transform is not None and last_transform.original_size == (ori_img_h, ori_img_w):
        return last_transform
    return make_transform((ori_img_h, ori_img_w), input_size, resize_type)


def _set_scheduling_params(runner: Any,
                           model: Any,
                           model_name: Optional[str],
                           priority: Optional[int],
                           bpu_cores: Optional[list]) -> None:
    """Forward explicit scheduler settings and reject unsupported requests."""
    method = getattr(runner, "set_scheduling_params", None)
    if method is not None:
        method(priority=priority, bpu_cores=bpu_cores)
        return
    method = getattr(model, "set_scheduling_params", None)
    if method is None:
        if priority is not None or bpu_cores is not None:
            raise BindingError(
                "The injected runner/runtime does not support explicit scheduling parameters.")
        return
    values: Dict[str, Any] = {}
    if priority is not None:
        values["priority"] = {model_name: priority}
    if bpu_cores is not None:
        values["bpu_cores"] = {model_name: bpu_cores}
    if values:
        method(**values)


def _predict_task(task: Any,
                  img: np.ndarray,
                  image_format: str,
                  score_thres: Optional[float],
                  nms_thres: Optional[float]) -> DetectionResult:
    """Run the shared image/runner/task orchestration for one detector."""
    input_tensor, transform = task.pre_process_with_transform(img, image_format)
    outputs = task.forward(input_tensor)
    return task.post_process(
        outputs,
        ori_img_w=int(img.shape[1]),
        ori_img_h=int(img.shape[0]),
        score_thres=score_thres,
        nms_thres=nms_thres,
        transform=transform,
    )


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
        self.last_transform: Optional[ImageTransform] = None
        self.last_image_transform: Optional[ImageTransform] = None

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Forward explicit runtime scheduling parameters to the runner."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores)

    def _build_input(self, y_plane: np.ndarray, uv_plane: np.ndarray):
        return _build_input(self.runner, self.input_adapter, y_plane, uv_plane)

    def pre_process_with_transform(
            self,
            img: np.ndarray,
            image_format: str = "BGR") -> Tuple[Dict[str, Dict[str, np.ndarray]], ImageTransform]:
        """Prepare one image and return its input tensors and actual transform."""
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
        """Call the injected runner exactly once."""
        return _forward_runner(self.runner, input_tensor)

    def _semantic_outputs(self, outputs: Any) -> Mapping[str, Any]:
        return _semantic_outputs(self.binding, self.contract, outputs, "DFL")

    def _transform_for_postprocess(self,
                                   ori_img_w: int,
                                   ori_img_h: int,
                                   transform: Optional[ImageTransform]) -> ImageTransform:
        if transform is not None:
            return transform
        return _transform_for_postprocess(
            self.last_transform, ori_img_w, ori_img_h,
            self.input_size, self.cfg.resize_type)

    def post_process(self,
                     outputs: Any,
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     transform: Optional[ImageTransform] = None) -> DetectionResult:
        """Decode, suppress, and map boxes to original image pixels."""
        semantic = self._semantic_outputs(outputs)
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
        concrete = self._transform_for_postprocess(ori_img_w, ori_img_h, transform)
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
