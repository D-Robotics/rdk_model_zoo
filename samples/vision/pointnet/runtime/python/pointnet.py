# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Chair part segmentation: raw points → normalized tensors → raw logits → IDs."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from utils.py_utils.quantization import dequantize_tensor
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


@dataclass(frozen=True)
class PointNetPredictionDetails:
    """One predict call's owned labels plus its prepared normalized points.

    The prepared record carries the exact normalized ``(1,3,N)`` tensor and the
    centroid/radius context needed to interpret or archive it; callers request
    it with ``return_details=True`` instead of recomputing stages.  It
    describes only its own call; the task never retains a last cloud.
    """

    labels: np.ndarray
    prepared: PreparedInput


class PointNetTask:
    """Four-stage PointNet API; no file IO, plotting, downloads or mutable context.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.
    """
    def __init__(self, runner, binding: ModelBinding):
        self.runner = runner
        self.binding = binding

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, points: np.ndarray) -> PreparedInput:
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

    def infer(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return raw runner-validated (1,N,4) logits, with no numerical transform."""
        return self.runner(tensors)

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Decode raw (1,N,4) logits to owned int32 (N,) IDs in input point order.

        Integer outputs use validated SCALE dequantization computed in float64,
        so distinct int8..int32 raw values keep their ordering for argmax
        (float32 decoding rounds large integers into artificial ties; see
        POINTNET-R2). F32 stays raw even if metadata carries a vestigial
        descriptor. No softmax is needed for argmax. Ties choose the lowest
        part index. Context is not consumed: no geometry restoration or point
        reordering occurs. Invalid raw tensors raise ValueError.
        """
        name = self.binding.output_name
        meta = self.binding.metadata
        if (not isinstance(raw, np.ndarray) or raw.shape != meta.output_shapes[name]
                or raw.dtype != np.dtype(meta.output_dtypes[name]) or not np.isfinite(raw).all()):
            raise ValueError("PointNet raw output shape/dtype/values differ from binding.")
        if raw.dtype == np.float32:
            decoded = raw
        else:
            quant_info = meta.output_quants.get(name)
            if quant_info is None:
                raise ValueError(
                    "PointNet integer output requires a SCALE quantization descriptor.")
            decoded = dequantize_tensor(raw, quant_info, dtype="float64")
        if not np.isfinite(decoded).all():
            raise ValueError("PointNet dequantization produced nonfinite logits.")
        return np.argmax(decoded[0], axis=1).astype(np.int32)

    def predict(self, points: np.ndarray, *, return_details: bool = False):
        """Run exactly preprocess → infer → postprocess on raw XYZ points.

        ``return_details=True`` wraps the usual ``(N,)`` int32 labels with
        this call's prepared record (normalized tensor plus centroid/radius
        context), so plotting and archiving need no second pass; the default
        return stays the plain labels array.
        """
        prepared = self.preprocess(points)
        labels = self.postprocess(self.infer(prepared.tensors))
        if return_details:
            return PointNetPredictionDetails(labels, prepared)
        return labels

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, points: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(points)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw)
