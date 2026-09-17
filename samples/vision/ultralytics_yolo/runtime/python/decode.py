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

"""Decoder for the reviewed YOLO DFL output protocol.

The decoder consumes semantic role names (``cls_8``, ``box_8`` …) after the
binding layer has validated physical tensors.  It supports rectangular model
inputs by deriving independent feature-map heights and widths.  Input/output
binding, SDK loading and image-coordinate restoration intentionally live in
other modules.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


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
    from samples.vision.ultralytics_yolo.runtime.python.rdk_yolo_utils.postprocess import NMS
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


__all__ = ["DecodeError", "sigmoid", "softmax", "decode_dfl", "decode_ltrb"]
