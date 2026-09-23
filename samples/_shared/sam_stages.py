# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared, SDK-free numeric stages for the two SAM sample pipelines."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from samples._shared.sam_binding import validate_tensors
from samples._shared.sam_tensor_io import (
    DEFAULT_BOX,
    IMAGE_SIZE,
    PreparedStageInput,
    prepare_decoder_input,
    prepare_encoder_input,
)


class StageError(RuntimeError):
    """A pipeline stage failed; ``stage`` identifies encoder or decoder."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"{stage} stage failed: {message}")


def _output_name(binding: Any, kind: str) -> str:
    expected = "low_res_masks" if kind == "mask" else "iou_predictions"
    names = tuple(binding.output_names)
    if set(names) != {"low_res_masks", "iou_predictions"}:
        raise ValueError(f"SAM decoder output names are not source names: {names!r}")
    return expected


def _owned_float32(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("SAM output must be numeric")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("SAM output must be finite")
    return result


def _metadata_shape(binding: Any, name: str, *, outputs: bool = True) -> tuple[int, ...] | None:
    metadata = binding.metadata
    field = "output_shapes" if outputs else "input_shapes"
    values = getattr(metadata, field, {})
    shape = values.get(name)
    return None if shape is None else tuple(int(dim) for dim in shape)


class EncoderStage:
    """Encode one image using the bound SAM stage.

    ``pre_process`` accepts numeric BGR HWC input and returns owned float32
    RGB NCHW ``(1,3,512,512)`` tensors. ``forward`` preserves native,
    metadata-validated embedding arrays. ``post_process`` owns a float32
    ``(1,256,32,32)`` embedding and performs no dequantization.
    """

    def __init__(self, runner: Any, binding: Any):
        self.runner = runner
        self.binding = binding

    def pre_process(self, image: Any) -> PreparedStageInput:
        return prepare_encoder_input(image, self.binding)

    def forward(self, prepared: PreparedStageInput | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        tensors = prepared.tensors if isinstance(prepared, PreparedStageInput) else prepared
        validated = validate_tensors(self.binding, tensors)
        return validate_tensors(self.binding, self.runner(validated), outputs=True)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        names = tuple(self.binding.output_names)
        if len(names) != 1:
            raise ValueError("SAM encoder must return exactly one metadata-bound output")
        validated = validate_tensors(self.binding, outputs, outputs=True)
        result = _owned_float32(validated[names[0]])
        expected = _metadata_shape(self.binding, names[0])
        if expected is not None and tuple(result.shape) != expected:
            raise ValueError(f"SAM encoder output shape must be {expected}, got {result.shape}")
        return result


class DecoderStage:
    """Decode an embedding and optional box prompt into a 512-square mask.

    ``pre_process`` accepts the bound float32 embedding and, for MobileSAM,
    an ordered finite box in 512-image coordinates. ``forward`` preserves the
    native validated output arrays. ``post_process`` casts only source-accepted
    numeric output to owned float32, selects the highest-IoU candidate, linearly
    resizes its logits to ``(512,512)``, and returns a bool mask, Python float
    IoU, Python int index, and owned low-resolution float32 masks.
    """

    def __init__(self, runner: Any, binding: Any):
        self.runner = runner
        self.binding = binding

    def pre_process(self, embedding: Any, *, box=None) -> PreparedStageInput:
        return prepare_decoder_input(embedding, self.binding, box)

    def forward(self, prepared: PreparedStageInput | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        tensors = prepared.tensors if isinstance(prepared, PreparedStageInput) else prepared
        validated = validate_tensors(self.binding, tensors)
        return validate_tensors(self.binding, self.runner(validated), outputs=True)

    def post_process(self, outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray | float | int]:
        outputs = validate_tensors(self.binding, outputs, outputs=True)
        mask_name = _output_name(self.binding, "mask")
        iou_name = _output_name(self.binding, "iou")
        masks = _owned_float32(outputs[mask_name])
        ious = _owned_float32(outputs[iou_name])
        mask_shape = _metadata_shape(self.binding, mask_name)
        iou_shape = _metadata_shape(self.binding, iou_name)
        if mask_shape is not None and tuple(masks.shape) != mask_shape:
            raise ValueError(f"SAM mask output shape must be {mask_shape}, got {masks.shape}")
        if iou_shape is not None and tuple(ious.shape) != iou_shape:
            raise ValueError(f"SAM IoU output shape must be {iou_shape}, got {ious.shape}")
        if masks.ndim != 4 or masks.shape[0] != 1 or masks.shape[1] <= 0 or masks.shape[2] <= 0 or masks.shape[3] <= 0:
            raise ValueError(f"SAM masks must have shape (1,N,H,W), got {masks.shape}")
        if ious.size != masks.shape[1]:
            raise ValueError("SAM IoU count does not match mask count")
        ious_flat = ious.reshape(-1)
        index = int(np.argmax(ious_flat))
        resized = cv2.resize(masks[0, index], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        threshold = resized >= 0.0 if self.binding.sample == "efficient_sam" else resized > 0.0
        return {
            "mask": np.array(threshold, dtype=bool, copy=True),
            "iou": float(ious_flat[index]),
            "mask_index": index,
            "low_res_masks": np.array(masks, dtype=np.float32, order="C", copy=True),
        }


class SAMPipeline:
    """Explicit encoder → decoder pipeline with per-stage error attribution."""

    def __init__(self, runner: Any, binding: Any):
        self.encoder = EncoderStage(runner.encoder, binding.encoder)
        self.decoder = DecoderStage(runner.decoder, binding.decoder)

    def predict(self, image: Any, *, box=None) -> dict[str, np.ndarray | float | int]:
        try:
            prepared_image = self.encoder.pre_process(image)
            raw_embedding = self.encoder.forward(prepared_image)
            embedding = self.encoder.post_process(raw_embedding)
        except Exception as exc:
            if isinstance(exc, StageError):
                raise
            raise StageError("encoder", str(exc)) from exc
        try:
            prepared_decoder = self.decoder.pre_process(embedding, box=box)
            raw_decoder = self.decoder.forward(prepared_decoder)
            return self.decoder.post_process(raw_decoder)
        except Exception as exc:
            if isinstance(exc, StageError):
                raise
            raise StageError("decoder", str(exc)) from exc


__all__ = ["DEFAULT_BOX", "DecoderStage", "EncoderStage", "SAMPipeline", "StageError"]
