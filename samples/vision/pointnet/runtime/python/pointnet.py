# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Chair part segmentation: raw points → normalized tensors → raw logits → IDs."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from samples._shared.quantization import apply_output_transform
from samples.vision.pointnet.runtime.python.model_binding import ModelBinding


@dataclass(frozen=True)
class PointContext:
    """Per-call centroid and radius; point order is never changed."""
    centroid: tuple[float, float, float]
    radius: float
    point_count: int


@dataclass(frozen=True)
class PreparedInput:
    tensors: Mapping[str, np.ndarray]
    context: PointContext


class PointNetTask:
    """Four-stage PointNet API; no file IO, plotting, downloads or mutable context."""
    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    def pre_process(self, points: np.ndarray) -> PreparedInput:
        """Normalize finite real (N,3) XYZ points to centroid 0 and max radius 1.

        N must equal compiled metadata. No resampling, padding or point reordering.
        Returns owned contiguous float32 (1,3,N) and immutable normalization context.
        ValueError rejects shape/count mismatches, zero radius and nonfinite data.
        """
        n = self.binding.metadata.input_shapes[self.binding.input_name][2]
        if not isinstance(points, np.ndarray) or points.shape != (n, 3) or points.dtype.kind not in 'fiu':
            raise ValueError(f"Expected {n} real XYZ points, shape ({n},3).")
        values = points.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Point coordinates must be finite float32 values.")
        centroid = np.mean(values, axis=0)
        centered = values - centroid[None]
        radius = float(np.max(np.sqrt(np.sum(centered ** 2, axis=1))))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Point cloud normalization requires a finite positive radius.")
        normalized = centered / radius
        tensor = np.ascontiguousarray(normalized.T[None], dtype=np.float32)
        return PreparedInput({self.binding.input_name: tensor},
                             PointContext(tuple(float(x) for x in centroid), radius, n))

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return raw runner-validated (1,N,4) logits, with no numerical transform."""
        return self.runner(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Decode raw (1,N,4) logits to owned int32 (N,) IDs in input point order.

        Integer outputs use validated SCALE dequantization; F32 stays raw even if
        metadata carries a vestigial descriptor. No softmax is needed for argmax.
        Ties choose the lowest part index. Context is not consumed: no geometry
        restoration or point reordering occurs. Invalid raw tensors raise ValueError.
        """
        name = self.binding.output_name
        meta = self.binding.metadata
        if (not isinstance(raw, np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError("PointNet raw output shape/dtype/values differ from binding.")
        transform = 'raw_f32' if raw.dtype == np.float32 else 'dequant'
        decoded = apply_output_transform(transform, {name: raw}, meta.output_quants)[name]
        if not np.isfinite(decoded).all():
            raise ValueError("PointNet dequantization produced nonfinite logits.")
        return np.argmax(decoded[0], axis=1).astype(np.int32)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Run exactly pre_process → forward → post_process on raw XYZ points."""
        prepared = self.pre_process(points)
        return self.post_process(self.forward(prepared.tensors))
