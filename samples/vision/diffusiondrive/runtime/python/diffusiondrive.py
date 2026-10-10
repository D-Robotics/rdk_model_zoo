# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""DiffusionDrive planning: load, preprocess, infer, postprocess, predict.

``DiffusionDrivePlanner`` owns the fixed four-input/four-output planning
contract end to end: construction loads the model through the shared lazy
transport, and each ``predict`` call runs preprocess -> infer ->
postprocess visible in this file, together with the validated affine IO
transforms (float32 rounding, then float64 clipping for int32/uint32
exactly as the source). Catalog selection, npz feature IO and presentation
live in ``cli.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import dequantize_tensor, validate_scale_quantization
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import NamedArrayRunner


# ======================================================================
# Validated affine IO transforms; no SDK, activation, rendering or data
# loading.
# ======================================================================

DTYPES = frozenset(
    ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32")
)


@dataclass(frozen=True)
class AffineTransform:
    dtype: str
    shape: tuple[int, ...]
    scale: tuple[float, ...] = ()
    zero: tuple[float, ...] = ()
    axis: int = 0


def transform(dtype, shape, quant, *, input_tensor=False):
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported physical dtype: {dtype}")
    integer = np.issubdtype(np.dtype(dtype), np.integer)
    scales = np.asarray(getattr(quant, "scale", []), dtype=np.float32)
    if not scales.size:
        if integer:
            raise ValueError("Integer physical tensors require explicit SCALE metadata")
        zeros = np.asarray(getattr(quant, "zero_point", []))
        if (
            not np.isfinite(zeros).all()
            or np.any(zeros != 0)
            or str(
                getattr(
                    getattr(quant, "quant_type", None),
                    "name",
                    getattr(quant, "quant_type", None),
                )
            )
            not in ("None", "NONE", "0")
        ):
            raise ValueError(
                "Floating tensor has inconsistent empty quantization metadata"
            )
        return AffineTransform(dtype, tuple(shape))
    validate_scale_quantization(quant, shape)
    if input_tensor and scales.size != 1:
        raise ValueError("Source input contract supports per-tensor quantization only")
    zeros = np.asarray(getattr(quant, "zero_point", []), dtype=np.float64).reshape(-1)
    if integer:
        limits = np.iinfo(dtype)
        if (
            np.any(zeros != np.rint(zeros))
            or np.any(zeros < limits.min)
            or np.any(zeros > limits.max)
        ):
            raise ValueError(
                "Integer zero points must be integral and within dtype range"
            )
    axis = int(getattr(quant, "axis", 0)) if scales.size > 1 else 0
    return AffineTransform(
        dtype,
        tuple(shape),
        tuple(float(x) for x in scales.reshape(-1)),
        tuple(float(x) for x in zeros),
        axis,
    )


def quantize(value, spec):
    tensor = np.asarray(value)
    if (
        tensor.shape != spec.shape
        or tensor.dtype != np.dtype("float32")
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("Logical input requires exact finite float32 shape")
    if not spec.scale:
        result = tensor.astype(spec.dtype, copy=True)
    else:
        zero = spec.zero[0] if spec.zero else 0.0
        # Preserve source float32 arithmetic; clip in float64 to avoid an int32/
        # uint32 upper bound rounding beyond the dtype range before integer cast.
        with np.errstate(over="ignore"):
            values = np.rint(tensor / float(spec.scale[0]) + zero)
        if np.issubdtype(np.dtype(spec.dtype), np.integer):
            limits = np.iinfo(spec.dtype)
            values = np.clip(values.astype(np.float64), limits.min, limits.max)
        result = values.astype(spec.dtype)
    if not np.isfinite(result).all():
        raise ValueError("Physical input conversion produced nonfinite values")
    return np.ascontiguousarray(result)


def decode(value, spec):
    tensor = np.asarray(value)
    if (
        tensor.shape != spec.shape
        or tensor.dtype != np.dtype(spec.dtype)
        or not np.isfinite(tensor).all()
    ):
        raise ValueError("Raw output differs from physical metadata")
    if spec.scale:
        q = SimpleNamespace(
            quant_type="SCALE",
            scale=np.asarray(spec.scale, np.float32),
            zero_point=np.asarray(spec.zero, np.float32),
            axis=spec.axis,
        )
        result = dequantize_tensor(tensor, q)
    else:
        result = tensor.astype(np.float32, copy=True)
    result = np.array(result, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("Decoded output contains nonfinite values")
    return result

from samples.vision.diffusiondrive.runtime.python.cli import ModelSelection

INPUT_SHAPES = MappingProxyType(
    {
        "camera": (1, 3, 256, 1024),
        "lidar": (1, 1, 256, 256),
        "status": (1, 8),
        "noise": (1, 20, 8, 2),
    }
)
OUTPUT_SHAPES = MappingProxyType(
    {
        "trajectory": (1, 8, 3),
        "agent_states": (1, 30, 5),
        "agent_labels": (1, 30),
        "bev_semantic_map": (1, 7, 128, 256),
    }
)


@dataclass(frozen=True)
class ModelBinding:
    """Validated planning tensor protocol with per-tensor transforms.

    Attributes:
        selection: The manifest-backed selection this binding was built from.
        metadata: Board-observed model metadata validated against the contract.
        input_transforms: Frozen input-name to physical transform mapping.
        output_transforms: Frozen output-name to decode transform mapping.
    """

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_transforms: object
    output_transforms: object

    @property
    def model_name(self):
        """Return the single submodel name declared by the artifact."""
        return self.metadata.model_name


def bind_model(selection, metadata):
    """Validate the exact named four-input/four-output planning contract.

    Args:
        selection: Manifest-backed selection whose asset and path must match.
        metadata: Board-observed metadata mapping or ``RuntimeMetadata``.

    Returns:
        ModelBinding: Validated per-tensor input/output transforms.

    Raises:
        ValueError: The selection differs from the manifest contract.
        MetadataMismatchError: Names, shapes, or dtypes violate the contract.
    """
    from samples.vision.diffusiondrive.runtime.python.cli import resolve_selection

    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if selection != expected:
        raise ValueError("Selection differs from manifest contract")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if len(meta.model_names) != 1:
        raise MetadataMismatchError("Expected one planning model")
    transforms = []
    for direction, shapes in (("input", INPUT_SHAPES), ("output", OUTPUT_SHAPES)):
        names = getattr(meta, direction + "_names")
        actual_shapes = getattr(meta, direction + "_shapes")
        dtypes = getattr(meta, direction + "_dtypes")
        quants = getattr(meta, direction + "_quants")
        if len(names) != len(shapes) or set(names) != set(shapes):
            raise MetadataMismatchError(f"Exact named {direction} set required")
        result = {}
        for name, shape in shapes.items():
            if actual_shapes.get(name) != shape:
                raise MetadataMismatchError(f"Unexpected {direction} shape for {name}")
            result[name] = transform(
                dtypes.get(name),
                shape,
                quants.get(name),
                input_tensor=direction == "input",
            )
        transforms.append(MappingProxyType(result))
    return ModelBinding(selection, meta, *transforms)


def create_runner(selection, *, runtime_factory=None, runtime=None) -> NamedArrayRunner:
    """Construct the lazy DiffusionDrive transport for a resolved selection.

    Args:
        selection: Manifest-backed selection carrying target, asset, and path.
        runtime_factory: Optional model-path-to-SDK-object factory (host seam).
        runtime: Optional prebuilt SDK object; overrides the factory.

    Returns:
        NamedArrayRunner: Lazy four-input runner bound to the planning
        physical IO contract; loading gates board identity and the
        published file hash.
    """
    return NamedArrayRunner(
        selection,
        binding_loader=bind_model,
        physical_inputs=lambda b: {
            n: (s.shape, s.dtype) for n, s in b.input_transforms.items()
        },
        task_name="DiffusionDrive",
        runtime=runtime,
        runtime_factory=runtime_factory,
        execution_target_gate=require_execution_target,
    )


@dataclass(frozen=True)
class DiffusionDriveDetails:
    """One predict call's decoded result plus its physical inputs and raw outputs.

    Callers that archive ``physical_inputs.npz``/``raw_outputs.npz`` request
    this record with ``return_details=True`` instead of recomputing stages.
    It describes only its own call; the task never retains a last output.
    """

    result: dict
    physical: dict
    raw: dict


class DiffusionDrivePlanner:
    """Plan one driving step's trajectory and agents with a compiled model.

    ``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
    established ``pre_process``/``forward``/``post_process`` names stay thin
    aliases of those implementations.

    Attributes:
        runner (NamedArrayRunner): Lazy shared transport used by infer.
        binding (ModelBinding): Validated tensor protocol and transforms.
        agent_score_threshold: Sigmoid cutoff for the agent mask.
    """

    def __init__(self, selection: ModelSelection, *, agent_score_threshold=0.5, runner=None):
        """Load the compiled model and validate its tensor protocol.

        Args:
            selection: Manifest-backed selection from ``cli.resolve_selection``.
            agent_score_threshold: Finite value in [0, 1] for the agent mask.
            runner: Optional injected transport (host-test seam); defaults to
                the shared lazy runner with the published-file hash gate.

        Returns:
            None.

        Raises:
            ValueError: The threshold or selection is invalid.
            MetadataMismatchError: Runtime metadata violates the contract.
            RuntimeError: Board identity or SDK loading fails.
        """
        if (
            not np.isfinite(agent_score_threshold)
            or not 0 <= agent_score_threshold <= 1
        ):
            raise ValueError("Agent score threshold must be finite and within [0,1]")
        self.runner = runner if runner is not None else create_runner(selection)
        self.binding = self.runner.load()
        self.agent_score_threshold = float(agent_score_threshold)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, features):
        """Exact four float32 feature arrays to metadata-declared physical IO."""
        if set(features) != set(self.binding.input_transforms):
            raise ValueError("Exact four logical input names required")
        return {
            name: quantize(features[name], spec)
            for name, spec in self.binding.input_transforms.items()
        }

    def infer(self, prepared):
        """Return all raw named outputs; never dequantize or render here."""
        return self.runner(prepared)

    def postprocess(self, outputs):
        """Raw physical tensors to owned trajectory/agents/BEV, no filesystem IO."""
        if set(outputs) != set(self.binding.output_transforms):
            raise ValueError("Exact four raw output names required")
        decoded = {
            name: decode(outputs[name], spec)
            for name, spec in self.binding.output_transforms.items()
        }
        scores = 1 / (1 + np.exp(-np.clip(decoded["agent_labels"], -60, 60)))
        return {
            "trajectory": decoded["trajectory"],
            "agent_states": decoded["agent_states"],
            "agent_scores": scores,
            "agent_mask": scores >= self.agent_score_threshold,
            "bev_logits": decoded["bev_semantic_map"],
            "bev_labels": np.argmax(decoded["bev_semantic_map"], axis=1).astype(
                np.uint8
            ),
        }

    def predict(self, features, *, return_details=False):
        """Compose the same three stages once with fixed caller-provided noise.

        ``return_details=True`` wraps the usual decoded mapping with this
        call's physical inputs and raw outputs, so one production inference
        also serves callers that archive the raw IO; the default return stays
        the decoded mapping alone.
        """
        physical = self.preprocess(features)
        raw = self.infer(physical)
        result = self.postprocess(raw)
        if return_details:
            return DiffusionDriveDetails(result, physical, raw)
        return result

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

    def pre_process(self, features):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(features)

    def forward(self, prepared):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)
