# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Offline HIMLoco policy stages and raw runner construction.

No robot control, SDK loading or file handling happens here: the board SDK
stays a lazy import inside the shared transport, and this module only
validates the fixed policy tensors, calls the runner and returns owned
actions. The model-owned loader :meth:`HimLocoTask.from_model` keeps runner
construction, load and binding inside the model class so callers never
assemble runners themselves. Offline policy inference only — never
actuators.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
import sys
import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.robotics.himloco.runtime.python.model_binding import (
    bind_model,
    validate_selection,
)

INPUT_NAME = "obs_history"
OUTPUT_NAME = "actions"


class RuntimeModelRunner(NamedArrayRunner):
    """Lazy raw transport using shared board/hash gates and scheduling support."""

    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        validate_selection(selection)
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: (
                binding.metadata.input_shapes[binding.input_name], "float32"
            ),
            task_name="HIMLoco",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )


    def __call__(self, tensors):
        """Adapt semantic policy vectors to the bound singleton SDK layout.

        Args:
            tensors: Exactly obs_history finite float32[1,270].

        Returns:
            dict: Owned actions float32[1,12], with values and joint order
            unchanged from the validated physical SDK output.

        Raises:
            ValueError: The semantic observation contract is violated.
            MetadataMismatchError: Physical runtime tensors changed.
        """
        binding = self.load()
        observation = _physical(tensors, INPUT_NAME, (1, 270))
        shape = binding.metadata.input_shapes[binding.input_name]
        outputs = super().__call__({binding.input_name: observation.reshape(shape)})
        return {OUTPUT_NAME: outputs[binding.output_name].reshape(1, 12)}


@dataclass(frozen=True)
class PreparedInput:
    """Owned float32 [1,270] tensors; no hidden history or mutable task context."""

    tensors: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RawOutputs:
    """Owned raw model arrays and latency from this call, never instance state."""

    tensors: Mapping[str, np.ndarray]
    latency_ms: float


@dataclass(frozen=True)
class HimLocoResult:
    """Owned float32 [1,12] raw policy actions and synchronous runner latency."""

    actions: np.ndarray
    latency_ms: float


def _physical(values, name, shape):
    if not isinstance(values, Mapping) or set(values) != {name}:
        raise ValueError(f"Expected exactly one physical tensor {name!r}")
    value = values[name]
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite float32 {shape}")
    return np.array(value, order="C", copy=True)


class HimLocoTask:
    """Pure policy boundary for a separately bound, injected single-model runner.

    The caller supplies all six observations, current first. No history updates,
    joint reordering, input scaling, output clipping or action application occurs.
    Runner metadata/target checks belong to the SDK adapter. Instances keep no
    per-call result state; the runner itself may require serial access.
    """

    def __init__(self, runner):
        if not callable(runner):
            raise TypeError("runner must be callable")
        self.runner = runner

    @classmethod
    def from_model(cls, selection, *, runtime=None, runtime_factory=None):
        """Construct the loaded policy task for one resolved selection.

        The model class owns the low-level assembly: it creates the raw
        runner, loads and binds the model (running the board/asset gates
        unless a ``runtime`` or ``runtime_factory`` host seam is supplied)
        and returns a task whose ``predict`` is ready to call.

        Args:
            selection: ModelSelection from ``resolve_selection``; its target
                must match the executing board on real execution.
            runtime: Optional injected SDK runtime (host-test seam).
            runtime_factory: Optional SDK factory callable (host-test seam).

        Returns:
            HimLocoTask: Loaded task constructed with the pure injected
            constructor; ``metadata``/``runtime_module_source`` expose the
            bound SDK evidence for reports.

        Raises:
            ValueError: Board identity or publication checks fail, or the
                selection differs from the declared X5 publication.
            MetadataMismatchError: SDK metadata differs from the policy
                contract.
            RuntimeError: The board SDK is unavailable or loading fails.
        """
        runner = RuntimeModelRunner(
            selection, runtime=runtime, runtime_factory=runtime_factory
        )
        runner.load()
        return cls(runner)

    @property
    def metadata(self):
        """Bound SDK metadata of the loaded model (evidence for reports)."""
        return self.runner.binding.metadata

    @property
    def runtime_module_source(self):
        """File (or synthesizing module) of the loaded board SDK runtime."""
        sdk_module = sys.modules.get("hbm_runtime")
        return str(
            getattr(sdk_module, "__file__", type(self.runner.runtime).__module__)
        )

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional list of nonnegative BPU core indexes.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, observation):
        """Pack exactly 270 finite real numeric values into owned float32 [1,270].

        Preserve source flattening order and values (apart from float32 casting).
        Accept a flat vector, [1,270], [6,45], or another 270-value numeric shape.
        Reject complex, boolean/string/object values and float32 overflow. This
        boundary consumes prepared training observations, not raw sensor data.
        """
        value = np.asarray(observation)
        if value.size != 270 or value.dtype.kind not in "fiu":
            raise ValueError("Expected exactly 270 real numeric observation values")
        with np.errstate(over="ignore", invalid="ignore"):
            tensor = np.array(value, dtype=np.float32, order="C", copy=True).reshape(
                1, 270
            )
        if not np.isfinite(tensor).all():
            raise ValueError("Observation must contain finite float32 values")
        return PreparedInput({INPUT_NAME: tensor})

    def infer(self, tensors):
        """Call one runner with float32 obs_history [1,270], preserve raw actions.

        The runner returns exactly actions float32 [1,12]. Names, shape, dtype
        and finite values are checked; no decoding/scaling/clipping is applied.
        Latency measures only the synchronous runner call, not this method's
        copies/validation, preprocessing or postprocessing. No device-only timing
        is claimed. Returned arrays own storage independent of runner buffers.
        """
        value = _physical(tensors, INPUT_NAME, (1, 270))
        start = perf_counter()
        outputs = self.runner({INPUT_NAME: value})
        elapsed = (perf_counter() - start) * 1000
        actions = _physical(outputs, OUTPUT_NAME, (1, 12))
        return RawOutputs({OUTPUT_NAME: actions}, elapsed)

    def postprocess(self, outputs):
        """Return owned actions [1,12] unchanged, with this raw call's latency.

        Requires RawOutputs, finite nonnegative latency and exact float32 actions.
        Does not multiply by 0.25 or add default joint positions: those are external
        control-system responsibilities, not offline inference behavior.
        """
        if not isinstance(outputs, RawOutputs):
            raise ValueError("Expected RawOutputs from the corresponding forward call")
        if (
            isinstance(outputs.latency_ms, bool)
            or not isinstance(outputs.latency_ms, (int, float))
            or not np.isfinite(outputs.latency_ms)
            or outputs.latency_ms < 0
        ):
            raise ValueError("Runner latency must be finite and nonnegative")
        return HimLocoResult(
            _physical(outputs.tensors, OUTPUT_NAME, (1, 12)), outputs.latency_ms
        )

    def predict(self, observation):
        """Execute preprocess → infer → postprocess with no retained call state."""
        prepared = self.preprocess(observation)
        return self.postprocess(self.infer(prepared.tensors))

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, observation):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(observation)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)
