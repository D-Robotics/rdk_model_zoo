# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared detection transport and per-call geometry, independent of DFL/LTRB math."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils import (
    preprocess as pre_utils,
)
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    ImageTransform,
    make_transform,
    resize_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.model_binding import BindingError


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
    from samples.vision.ultralytics_yolo.runtime.python.tensor_io import RawOutputs

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


def _predict_task(
    task: Any,
    img: np.ndarray,
    image_format: str,
    score_thres: Optional[float],
    nms_thres: Optional[float],
) -> DetectionResult:
    """Compose the exact three public stages with per-call geometry."""
    prepared = task.pre_process(img, image_format)
    outputs = task.forward(prepared.tensors)
    return task.post_process(
        outputs,
        score_thres=score_thres,
        nms_thres=nms_thres,
        transform=prepared.transform,
    )
