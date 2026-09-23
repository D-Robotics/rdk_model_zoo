# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure YOLOWorld task stages for X5's RGB/text protocol."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence
import cv2
import numpy as np
from samples.vision.yoloworld.runtime.python.model_binding import ModelBinding

@dataclass(frozen=True)
class DetectionContext:
    original_height: int
    original_width: int
    scale: float
    prompts: tuple[str, ...]
    class_ids: tuple[int, ...]

@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: DetectionContext

@dataclass(frozen=True)
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    prompts: tuple[str, ...]

class YOLOWorldTask:
    """Open-vocabulary detection with no mutable prompt/geometry state."""
    def __init__(self, runner, binding: ModelBinding, vocabulary: Mapping[str, Sequence[float]],
                 *, score_thres: float = 0.05, nms_thres: float = 0.45):
        if not vocabulary:
            raise ValueError("Offline vocabulary must not be empty.")
        self.runner, self.binding = runner, binding
        self.class_names = tuple(vocabulary)
        self.vocabulary = {str(k): np.asarray(v, dtype=np.float32) for k, v in vocabulary.items()}
        if any(v.shape != (512,) or not np.isfinite(v).all() for v in self.vocabulary.values()):
            raise ValueError("Every offline vocabulary embedding must be finite F32[512].")
        self.score_thres, self.nms_thres = float(score_thres), float(nms_thres)
        if not 0 <= self.score_thres <= 1 or not 0 <= self.nms_thres <= 1:
            raise ValueError("score_thres and nms_thres must be in [0,1].")

    def _prompts(self, prompts: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray]:
        if isinstance(prompts, str):
            raise TypeError("prompts must be a sequence of strings, not one comma string")
        values = tuple(str(p).strip() for p in prompts)
        if not values or any(not p for p in values):
            raise ValueError("At least one nonempty prompt is required; empty prompts are rejected.")
        if len(values) > 32:
            raise ValueError("YOLOWorld accepts at most 32 prompts per call.")
        ids = tuple(self.class_names.index(p) if p in self.vocabulary else -1 for p in values)
        if -1 in ids:
            missing = values[ids.index(-1)]
            raise KeyError(f"Prompt {missing!r} is absent from offline vocabulary.")
        rows = [self.vocabulary[p] for p in values]
        while len(rows) < 32:
            rows.append(rows[-1])
            ids = ids + (ids[-1],)
        return values, ids, np.asarray(rows, dtype=np.float32).reshape(1, 32, 512, 1)

    def pre_process(self, image: np.ndarray, prompts: Sequence[str]) -> PreparedInput:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("image must be a nonempty HxWx3 BGR array")
        if image.dtype not in (np.uint8, np.float32):
            raise ValueError("image must be uint8 or float32")
        image_f = image.astype(np.float32, copy=False)
        if not np.isfinite(image_f).all() or image_f.min() < 0 or image_f.max() > 255:
            raise ValueError("image must contain finite values in [0,255]")
        names, ids, text = self._prompts(prompts)
        h, w = image.shape[:2]
        scale = max(h, w) / 640.0
        resized = cv2.resize(image, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
        canvas = np.zeros((640, 640, 3), dtype=np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1)[None].copy()
        context = DetectionContext(h, w, scale, names, ids)
        return PreparedInput(MappingProxyType({self.binding.image_input_name: rgb,
                                                 self.binding.text_input_name: text}), context)

    def forward(self, prepared: PreparedInput | Mapping[str, np.ndarray]):
        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
        keep: list[int] = []
        for cls in np.unique(classes):
            indices = np.flatnonzero(classes == cls)
            order = indices[np.argsort(scores[indices])[::-1]]
            while len(order):
                current = int(order[0]); keep.append(current)
                if len(order) == 1: break
                rest = order[1:]
                xx1 = np.maximum(boxes[current, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
                xx2 = np.minimum(boxes[current, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area = np.maximum(0, boxes[current, 2]-boxes[current, 0]) * np.maximum(0, boxes[current, 3]-boxes[current, 1])
                rest_area = np.maximum(0, boxes[rest, 2]-boxes[rest, 0]) * np.maximum(0, boxes[rest, 3]-boxes[rest, 1])
                iou = inter / (area + rest_area - inter + 1e-9)
                order = rest[iou < threshold]
        return keep

    def post_process(self, outputs: Mapping[str, np.ndarray], context: DetectionContext) -> DetectionResult:
        score_raw = np.asarray(outputs[self.binding.score_output_name]); box_raw = np.asarray(outputs[self.binding.box_output_name])
        if score_raw.shape not in ((1, 8400, 32), (1, 8400, 32, 1)) or box_raw.shape not in ((1, 8400, 4), (1, 8400, 4, 1)) or score_raw.dtype != np.float32 or box_raw.dtype != np.float32:
            raise ValueError("Raw outputs must preserve native F32 logical score/box shapes.")
        if score_raw.ndim == 4: score_raw = score_raw[..., 0]
        if box_raw.ndim == 4: box_raw = box_raw[..., 0]
        slots = np.argmax(score_raw[0], axis=1)
        scores = score_raw[0, np.arange(8400), slots]
        mask = scores >= self.score_thres
        if not np.any(mask):
            return DetectionResult(np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32), context.prompts)
        boxes = box_raw[0, mask].copy(); scores = scores[mask].copy(); slots = slots[mask].astype(np.int32)
        keep = self._nms(boxes, scores, slots, self.nms_thres)
        boxes, scores, slots = boxes[keep], scores[keep], slots[keep]
        boxes *= context.scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, context.original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, context.original_height)
        ids = np.asarray([context.class_ids[int(slot)] for slot in slots], dtype=np.int32)
        return DetectionResult(boxes, scores, ids, context.prompts)

    def predict(self, image: np.ndarray, prompts: Sequence[str]) -> DetectionResult:
        prepared = self.pre_process(image, prompts)
        return self.post_process(self.forward(prepared), prepared.context)
