# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Offline HIMLoco policy stages; no robot control, SDK loading or file handling."""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
import numpy as np

INPUT_NAME = "obs_history"
OUTPUT_NAME = "actions"


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

    def pre_process(self, observation):
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

    def forward(self, tensors):
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

    def post_process(self, outputs):
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
        """Execute pre_process → forward → post_process with no retained call state."""
        prepared = self.pre_process(observation)
        return self.post_process(self.forward(prepared.tensors))
