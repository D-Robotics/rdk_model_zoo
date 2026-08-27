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

"""Provide reusable RDK X5 inference for the fused HIMLoco policy.

The module validates the compiled model contract and exposes the standard
Model Zoo pipeline: ``pre_process -> forward -> post_process -> predict``.

Notes:
    The BSP-provided X5 ``hbm_runtime`` package is required on the board. The
    unrelated PyPI package with the same name must not be installed.
"""

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
SAMPLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = SAMPLE_ROOT / "model" / "bayes-e" / "himloco_go2_bayese_1x270.bin"

RuntimeInputs = dict[str, dict[str, np.ndarray]]
RuntimeOutputs = dict[str, dict[str, np.ndarray]]


def _dtype_name(value: Any) -> str:
    """Return a stable name for an ``hbm_runtime`` dtype enum.

    Args:
        value: Runtime dtype enum or another string-compatible value.

    Returns:
        Stable enum member name without a module prefix.
    """

    name = getattr(value, "name", None)
    return str(name if name is not None else value).split(".")[-1]


def _shape(value: Any, tensor_name: str) -> tuple[int, ...]:
    """Normalize and validate one Runtime tensor shape.

    Args:
        value: Iterable Runtime shape description.
        tensor_name: Tensor name used in validation errors.

    Returns:
        Positive tensor dimensions as an immutable tuple.

    Raises:
        TypeError: If dimensions cannot be converted to integers.
        ValueError: If the shape is empty or contains a non-positive dimension.
    """

    try:
        shape = tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"invalid shape for {tensor_name}: {value!r}") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"invalid shape for {tensor_name}: {shape}")
    return shape


@dataclass(frozen=True)
class HimLocoConfig:
    """Configure HIMLoco model loading and Runtime scheduling.

    Attributes:
        model_path: Path to the RDK X5 Bayes-e ``.bin`` model.
        priority: Optional DNN task priority in the inclusive range [0, 255].
        bpu_cores: Optional BPU core indexes; ``None`` keeps Runtime scheduling.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    priority: int | None = None
    bpu_cores: tuple[int, ...] | None = None


@dataclass(frozen=True)
class HimLocoResult:
    """Store one policy action and measured Runtime call latency.

    Attributes:
        actions: Float32 policy actions with shape ``(1, 12)``.
        latency_ms: Synchronous ``HB_HBMRuntime.run`` latency in milliseconds.
    """

    actions: np.ndarray
    latency_ms: float


class HimLoco:
    """Run the fused HIMLoco Go2 policy with ``HB_HBMRuntime`` on RDK X5.

    Args:
        config: Model path and optional Runtime scheduling configuration.
        runtime_factory: Optional Runtime constructor used for controlled tests.

    Attributes:
        config: Effective immutable HIMLoco configuration.
        model_name: Packed model name reported by the Runtime.
        input_name: Validated model input tensor name.
        output_name: Validated model output tensor name.

    Notes:
        The instance owns one Runtime object and is intended for serial use.
    """

    def __init__(
        self,
        config: HimLocoConfig | None = None,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """Load and validate one fused HIMLoco Runtime model.

        Args:
            config: Model path and optional Runtime scheduling configuration.
            runtime_factory: Optional Runtime constructor used for controlled
                tests. ``None`` selects the BSP-provided ``HB_HBMRuntime``.

        Raises:
            FileNotFoundError: If the configured model file does not exist.
            RuntimeError: If the RDK X5 Runtime package is unavailable.
            TypeError: If the model tensor dtypes do not match the contract.
            ValueError: If the model suffix or metadata does not match.
        """

        self.config = config or HimLocoConfig()
        self.model_path = Path(self.config.model_path).expanduser().resolve()
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
            module = getattr(runtime_factory, "__module__", "unreported")
            name = getattr(
                runtime_factory,
                "__qualname__",
                type(runtime_factory).__name__,
            )
            self.runtime_module_source = f"{module}.{name}"

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
        self._last_latency_ms: float | None = None
        self._validate_contract()
        self.set_scheduling_params(
            priority=self.config.priority,
            bpu_cores=(
                list(self.config.bpu_cores)
                if self.config.bpu_cores is not None
                else None
            ),
        )

    def _validate_contract(self) -> None:
        """Reject a model whose metadata differs from the policy contract.

        Returns:
            None.

        Raises:
            TypeError: If an input or output tensor is not float32.
            ValueError: If a tensor name, batch, shape, or element count differs.
        """

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
        """Return JSON-serializable model and Runtime metadata.

        Returns:
            Model name, I/O contract, Runtime version, and scheduling metadata.
        """

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
        """Return effective Runtime scheduling state when exposed.

        Returns:
            Scheduling fields reported by the Runtime, or ``None`` if unavailable.
        """

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
        """Apply explicitly requested Runtime scheduling parameters.

        Args:
            priority: Optional DNN task priority in the range [0, 255].
            bpu_cores: Optional non-empty list of non-negative core indexes.

        Returns:
            None.

        Raises:
            ValueError: If priority or core indexes are outside valid ranges.
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

    def pre_process(self, observation: np.ndarray) -> RuntimeInputs:
        """Validate one observation and construct Runtime input tensors.

        Args:
            observation: One finite observation containing exactly 270 values.

        Returns:
            Nested model/input dictionary accepted directly by ``runtime.run``.

        Raises:
            ValueError: If the observation size or finite-value check fails.
        """

        array = np.asarray(observation)
        if array.size != INPUT_WIDTH:
            raise ValueError(
                f"expected one {INPUT_WIDTH}-value observation, got shape {array.shape}"
            )
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("observation contains NaN/Inf")
        tensor = np.ascontiguousarray(array.reshape(self.input_shape))
        return {self.model_name: {self.input_name: tensor}}

    def forward(self, inputs: RuntimeInputs) -> RuntimeOutputs:
        """Execute one synchronous BPU inference call.

        Args:
            inputs: Nested model/input dictionary produced by ``pre_process``.

        Returns:
            Direct nested output dictionary returned by ``runtime.run``.

        Raises:
            ValueError: If the input dictionary does not match the model contract.
        """

        if set(inputs) != {self.model_name}:
            raise ValueError(f"expected model input key {self.model_name!r}")
        model_inputs = inputs[self.model_name]
        if set(model_inputs) != {self.input_name}:
            raise ValueError(f"expected input tensor key {self.input_name!r}")
        tensor = np.asarray(model_inputs[self.input_name])
        if tensor.shape != self.input_shape or tensor.dtype != np.float32:
            raise ValueError(
                f"expected float32 input shape {self.input_shape}, "
                f"got {tensor.dtype} {tensor.shape}"
            )
        if not tensor.flags.c_contiguous:
            raise ValueError("Runtime input tensor must be C-contiguous")

        started = time.perf_counter()
        outputs = self.runtime.run(inputs)
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        return outputs

    def post_process(self, outputs: RuntimeOutputs) -> HimLocoResult:
        """Convert raw Runtime output into policy actions.

        Args:
            outputs: Direct nested output dictionary returned by ``forward``.

        Returns:
            Float32 actions with the measured synchronous Runtime latency.

        Raises:
            RuntimeError: If ``forward`` was not called before post-processing.
            ValueError: If output keys, size, or finite-value checks fail.
        """

        if self._last_latency_ms is None:
            raise RuntimeError("forward must be called before post_process")
        try:
            raw_actions = outputs[self.model_name][self.output_name]
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
        return HimLocoResult(actions=actions, latency_ms=self._last_latency_ms)

    def predict(self, observation: np.ndarray) -> HimLocoResult:
        """Run the complete preprocessing, inference, and output pipeline.

        Args:
            observation: One finite observation containing exactly 270 values.

        Returns:
            Float32 actions and synchronous Runtime call latency.
        """

        return self.post_process(self.forward(self.pre_process(observation)))

    def __call__(self, observation: np.ndarray) -> HimLocoResult:
        """Run ``predict`` through the callable model interface.

        Args:
            observation: One finite observation containing exactly 270 values.

        Returns:
            Float32 actions and synchronous Runtime call latency.
        """

        return self.predict(observation)

    def infer(self, observation: np.ndarray) -> HimLocoResult:
        """Run one policy inference through the compatibility API.

        Args:
            observation: One finite observation containing exactly 270 values.

        Returns:
            The same result as ``predict``.
        """

        return self.predict(observation)

    def warmup(self, observation: np.ndarray, count: int) -> None:
        """Run unmeasured warmup inferences with one representative input.

        Args:
            observation: Representative finite 270-value observation.
            count: Non-negative number of warmup calls.

        Returns:
            None.

        Raises:
            ValueError: If ``count`` is negative.
        """

        if count < 0:
            raise ValueError("warmup count must be non-negative")
        for _ in range(count):
            self.predict(observation)
