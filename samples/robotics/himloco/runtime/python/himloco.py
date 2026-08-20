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

"""Reusable RDK X5 Python Runtime wrapper for the fused HIMLoco policy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

INPUT_NAME = "obs_history"
INPUT_WIDTH = 270
OUTPUT_NAME = "actions"
OUTPUT_WIDTH = 12


def _dtype_name(value: Any) -> str:
    """Return a stable name for an hbm_runtime dtype enum."""

    name = getattr(value, "name", None)
    return str(name if name is not None else value).split(".")[-1]


def _shape(value: Any, tensor_name: str) -> tuple[int, ...]:
    """Normalize and validate one Runtime tensor shape."""

    try:
        shape = tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"invalid shape for {tensor_name}: {value!r}") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"invalid shape for {tensor_name}: {shape}")
    return shape


@dataclass(frozen=True)
class HimLocoResult:
    """Store one policy action and the measured Runtime call latency."""

    actions: np.ndarray
    latency_ms: float


class HimLoco:
    """Run the fused HIMLoco Go2 policy with ``HB_HBMRuntime`` on RDK X5.

    The wrapper enforces the deployment contract before the first inference:
    one float32 ``obs_history`` input containing 270 values and one float32
    ``actions`` output containing 12 values. Runtime layouts such as
    ``[1, 270]`` and ``[1, 1, 1, 270]`` are both accepted when their element
    counts match the compiled contract.
    """

    def __init__(
        self,
        model_path: str | Path,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if self.model_path.suffix != ".bin":
            raise ValueError(f"RDK X5 Runtime requires a .bin model: {self.model_path}")

        if runtime_factory is None:
            try:
                import hbm_runtime
            except ImportError as error:
                raise RuntimeError(
                    "hbm_runtime is unavailable; run on RDK X5 OS >= 3.5.0 with "
                    "the matching board package installed"
                ) from error
            runtime_factory = hbm_runtime.HB_HBMRuntime
            self.runtime_module_source = str(
                getattr(hbm_runtime, "__file__", "unreported")
            )
        else:
            self.runtime_module_source = (
                f"{runtime_factory.__module__}.{runtime_factory.__qualname__}"
            )

        self.runtime = runtime_factory(str(self.model_path))
        model_names = list(self.runtime.model_names)
        if len(model_names) != 1:
            raise ValueError(f"expected exactly one packed model, got {model_names}")
        self.model_name = model_names[0]

        input_names = list(self.runtime.input_names[self.model_name])
        output_names = list(self.runtime.output_names[self.model_name])
        if input_names != [INPUT_NAME]:
            raise ValueError(f"expected input {[INPUT_NAME]}, got {input_names}")
        if output_names != [OUTPUT_NAME]:
            raise ValueError(f"expected output {[OUTPUT_NAME]}, got {output_names}")
        self.input_name = input_names[0]
        self.output_name = output_names[0]

        self.input_shape = _shape(
            self.runtime.input_shapes[self.model_name][self.input_name],
            self.input_name,
        )
        self.output_shape = _shape(
            self.runtime.output_shapes[self.model_name][self.output_name],
            self.output_name,
        )
        self.input_dtype = _dtype_name(
            self.runtime.input_dtypes[self.model_name][self.input_name]
        )
        self.output_dtype = _dtype_name(
            self.runtime.output_dtypes[self.model_name][self.output_name]
        )
        self._validate_contract()

    def _validate_contract(self) -> None:
        """Reject a model whose Runtime metadata differs from the policy contract."""

        if int(np.prod(self.input_shape)) != INPUT_WIDTH or self.input_shape[0] != 1:
            raise ValueError(
                f"expected {INPUT_NAME} batch-one shape with {INPUT_WIDTH} values, "
                f"got {self.input_shape}"
            )
        if int(np.prod(self.output_shape)) != OUTPUT_WIDTH or self.output_shape[0] != 1:
            raise ValueError(
                f"expected {OUTPUT_NAME} batch-one shape with {OUTPUT_WIDTH} values, "
                f"got {self.output_shape}"
            )
        if self.input_dtype not in {"F32", "FLOAT"}:
            raise TypeError(f"expected float32 input, got {self.input_dtype}")
        if self.output_dtype not in {"F32", "FLOAT"}:
            raise TypeError(f"expected float32 output, got {self.output_dtype}")

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable model and Runtime metadata."""

        runtime_version = getattr(self.runtime, "version", "unreported")
        return {
            "runtime_version": str(runtime_version),
            "runtime_module_source": self.runtime_module_source,
            "model_name": self.model_name,
            "input": {
                "name": self.input_name,
                "shape": list(self.input_shape),
                "dtype": self.input_dtype,
            },
            "output": {
                "name": self.output_name,
                "shape": list(self.output_shape),
                "dtype": self.output_dtype,
            },
            "scheduling": self.scheduling_metadata(),
        }

    def scheduling_metadata(self) -> dict[str, Any] | None:
        """Return the effective Runtime scheduling state when it is exposed."""

        sched_params = getattr(self.runtime, "sched_params", None)
        if not sched_params or self.model_name not in sched_params:
            return None
        sched = sched_params[self.model_name]
        result = {}
        for report_name, attribute_name in (
            ("priority", "priority"),
            ("custom_id", "customId"),
            ("bpu_cores", "bpu_cores"),
            ("device_id", "deviceId"),
        ):
            value = getattr(sched, attribute_name, None)
            if value is not None:
                result[report_name] = (
                    [int(item) for item in value]
                    if report_name == "bpu_cores"
                    else int(value)
                )
        return result or None

    def set_scheduling_params(
        self,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
    ) -> None:
        """Apply explicitly requested scheduling parameters.

        No priority or core binding is applied by default, leaving scheduling
        to the board Runtime.
        """

        kwargs: dict[str, dict[str, Any]] = {}
        if priority is not None:
            if not 0 <= priority <= 255:
                raise ValueError("priority must be in [0,255]")
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            if not bpu_cores or any(core < 0 for core in bpu_cores):
                raise ValueError("bpu_cores must contain non-negative indexes")
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.runtime.set_scheduling_params(**kwargs)

    def prepare_input(self, observation: np.ndarray) -> np.ndarray:
        """Validate one observation and reshape it to the Runtime layout."""

        array = np.asarray(observation)
        if array.size != INPUT_WIDTH:
            raise ValueError(
                f"expected one {INPUT_WIDTH}-value observation, got shape {array.shape}"
            )
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("observation contains NaN/Inf")
        return np.ascontiguousarray(array.reshape(self.input_shape))

    def infer(self, observation: np.ndarray) -> HimLocoResult:
        """Run one policy inference and return float32 actions with latency.

        ``latency_ms`` measures only the synchronous ``runtime.run`` Python call;
        file reads, input validation, and action dump writes are excluded.
        """

        input_tensor = self.prepare_input(observation)
        inputs = {self.model_name: {self.input_name: input_tensor}}
        started = time.perf_counter()
        results = self.runtime.run(inputs)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            raw_actions = results[self.model_name][self.output_name]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Runtime output does not match the model metadata"
            ) from error
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.size != OUTPUT_WIDTH:
            raise ValueError(
                f"expected {OUTPUT_WIDTH} action values, got shape {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Runtime output contains NaN/Inf")
        actions = np.ascontiguousarray(actions.reshape(1, OUTPUT_WIDTH))
        return HimLocoResult(actions=actions, latency_ms=latency_ms)

    def warmup(self, observation: np.ndarray, count: int) -> None:
        """Run unmeasured warmup inferences with one representative input."""

        if count < 0:
            raise ValueError("warmup count must be non-negative")
        for _ in range(count):
            self.infer(observation)
