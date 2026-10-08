# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""DiffusionDrive planning: load, preprocess, infer, postprocess, predict.

``DiffusionDrivePlanner`` owns the fixed four-input/four-output planning
contract end to end: construction loads the model through the shared lazy
transport, and each ``predict`` call runs preprocess -> infer ->
postprocess visible in this file. Catalog selection and presentation live
in ``cli.py``; tensor quantization/decoding math lives in
``quantization.py``; npz feature IO lives in ``data_io.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from utils.py_utils.platforms import require_execution_target
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import NamedArrayRunner

from samples.vision.diffusiondrive.runtime.python.cli import ModelSelection
from samples.vision.diffusiondrive.runtime.python.quantization import quantize, decode, transform

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
