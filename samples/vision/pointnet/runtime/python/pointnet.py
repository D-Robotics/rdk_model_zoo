# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet chair part segmentation: load, preprocess, infer, postprocess, predict.

``PointNetSegmenter`` owns the fixed per-point contract end to end:
construction loads the model through the shared lazy transport, and each
``predict`` call runs preprocess -> infer -> postprocess visible in this
file. Catalog selection, evidence writing and plotting live in ``cli.py``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import dequantize_tensor, validate_scale_quantization
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.vision.pointnet.runtime.python.cli import ModelSelection


@dataclass(frozen=True)
class ModelBinding:
    """Validated PointNet tensor protocol; N comes from fixed artifact metadata.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_name: Bound normalized-points F32 input tensor name.
        output_name: Bound per-point logits output tensor name.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source PointNet tensor protocol.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated tensor names for the (1,3,N) F32 contract.

    Raises:
        BindingError: The selection does not match the exact manifest asset.
        MetadataMismatchError: Model, tensor, geometry, or dtype contract fails.
    """
    from samples.vision.pointnet.runtime.python.cli import BindingError, resolve_selection

    # Re-resolve caller-created selections: identity and path stay inseparable.
    resolved = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
        or selection.explicit_model_path != resolved.explicit_model_path
    ):
        raise BindingError("ModelSelection does not match the exact manifest asset and path.")

    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("PointNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1:
        raise MetadataMismatchError("PointNet requires exactly one point input.")
    if len(meta.output_names) == 1:
        output_name = meta.output_names[0]
    elif len(meta.output_names) == 2 and set(meta.output_names) == {"pred", "trans"}:
        # The published HBM also exposes its learned XYZ transform. It is
        # an auxiliary output, not another set of chair-part logits.
        output_name = "pred"
        if meta.output_shapes.get("trans") != (1, 3, 3) or meta.output_dtypes.get("trans") != "float32":
            raise MetadataMismatchError("PointNet trans must be float32 (1,3,3).")
    else:
        raise MetadataMismatchError("PointNet requires part logits, optionally with the named XYZ transform.")
    input_name = meta.input_names[0]
    shape = meta.input_shapes.get(input_name, ())
    if len(shape) != 3 or shape[:2] != (1, 3) or type(shape[2]) is not int or shape[2] <= 0:
        raise MetadataMismatchError("PointNet input must have fixed shape (1,3,N), N > 0.")
    if meta.input_dtypes.get(input_name) != "float32":
        raise MetadataMismatchError("PointNet input must be float32.")
    if meta.output_shapes.get(output_name) != (1, shape[2], 4):
        raise MetadataMismatchError("PointNet output must be (1,N,4), matching the input point count.")
    dtype = meta.output_dtypes.get(output_name)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError(f"Unsupported PointNet output dtype {dtype!r}.")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(output_name), (1, shape[2], 4))
    return ModelBinding(selection, meta, input_name, output_name)


class PointNetRunner(NamedArrayRunner):
    """Validate all published SDK outputs and return the named part logits.

    The auxiliary XYZ transform is checked for shape, dtype and finite values
    by the shared named transport. Part decoding uses only ``pred``; output
    ordering never selects a different tensor as the segmentation result.
    """

    def __call__(self, tensors):
        """Return owned raw part logits after every bound output is validated."""
        binding = self.load()
        return super().__call__(tensors)[binding.output_name]


def create_runner(selection: ModelSelection, *, runtime_factory=None, runtime=None) -> PointNetRunner:
    """Construct the lazy PointNet transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        PointNetRunner: Lazy runner bound to the normalized-points F32
        physical input contract; loading gates board identity and the
        published file hash.
    """
    return PointNetRunner(
        selection,
        binding_loader=bind_model,
        physical_input=lambda binding: (binding.metadata.input_shapes[binding.input_name], "float32"),
        task_name="PointNet",
        runtime_factory=runtime_factory,
        runtime=runtime,
        execution_target_gate=require_execution_target,
    )


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


class PointNetSegmenter:
    """Segment one chair point cloud into 4 parts with a compiled PointNet.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations. No file IO, plotting, downloads or
    mutable context happens here.

    Attributes:
        runner (PointNetRunner): Shared transport with explicit output roles.
        binding (ModelBinding): Validated tensor names and runtime metadata.
    """
    def __init__(self, selection: ModelSelection, *, runner=None):
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            runner: Optional injected transport (host-test seam); defaults to
                the shared lazy runner with the published-file hash gate.

        Returns:
            None.

        Raises:
            ValueError: The selection or its local file is invalid.
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()

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

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional non-empty list of nonnegative BPU core indexes.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, points: np.ndarray) -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(points)

    def forward(self, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw)
