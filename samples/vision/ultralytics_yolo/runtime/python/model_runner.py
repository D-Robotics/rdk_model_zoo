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

"""The small, replaceable model-call boundary for the YOLO task.

``ModelRunner`` is responsible for loading one model once, reading its actual
runtime metadata, binding that metadata, and executing it.  It deliberately
does not know about images, confidence thresholds, NMS, or drawing.  Tests can
inject a callable runner into :class:`yolo_detect.YoloDetect`; the board SDK is
only touched by this module's factory.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    DFLDetectionContract,
    ModelBinding,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
    default_dfl_contract,
)


class RunnerError(RuntimeError):
    """Raised when the runtime cannot be loaded or called."""


def _selection_from_config(config: Any) -> ModelSelection:
    """Build a selection from the legacy detector configuration object."""
    profile = getattr(config, "platform", None)
    target = getattr(profile, "key", None) if profile is not None else None
    contract = getattr(config, "contract", None)
    if contract is None:
        contract = default_dfl_contract(
            classes=int(getattr(config, "classes_num", 80)),
            reg_bins=int(getattr(config, "reg", 16)),
            strides=tuple(getattr(config, "strides", (8, 16, 32))),
        )
    return ModelSelection(
        model_path=str(config.model_path),
        target=target,
        platform=profile,
        task="detect",
        contract=contract,
        input_shape=getattr(config, "input_shape", None),
    )


class ModelRunner:
    """Load and call one compiled model with its validated binding."""

    def __init__(self,
                 model: Any,
                 binding: ModelBinding,
                 metadata: Optional[RuntimeMetadata] = None) -> None:
        self.model = model
        self.binding = binding
        self.metadata = metadata or binding.metadata
        self.model_name = self.metadata.model_name
        self.input_adapter = binding.input_adapter
        self.output_adapter = binding.output_adapter
        self.input_names = tuple(self.input_adapter.input_names)
        self.output_names = tuple(self.metadata.output_names)
        self.input_shapes = dict(self.metadata.input_shapes)
        self.output_shapes = dict(self.metadata.output_shapes)
        self.input_dtypes = dict(self.metadata.input_dtypes)
        self.output_dtypes = dict(self.metadata.output_dtypes)
        self.input_height = self.input_adapter.input_height
        self.input_width = self.input_adapter.input_width
        self.input_size = (self.input_height, self.input_width)

    @property
    def contract(self) -> Any:
        return self.binding.contract

    @classmethod
    def from_selection(cls,
                       selection: ModelSelection,
                       runtime_loader: Optional[Callable[[], Any]] = None) -> "ModelRunner":
        """Load a selected artifact, inspect it, and bind its actual metadata."""
        if not isinstance(selection, ModelSelection):
            if not all(hasattr(selection, name)
                       for name in ("model_path", "task", "contract")):
                raise RunnerError("ModelRunner.from_selection expects ModelSelection.")
            selection = ModelSelection(
                model_path=selection.model_path,
                target=getattr(selection, "target", None),
                task=selection.task,
                contract=selection.contract,
                input_shape=getattr(selection, "input_shape", None),
                family=getattr(selection, "family", None),
                artifact_id=getattr(selection, "artifact_id", None),
                platform=getattr(selection, "platform", None),
            )
        if runtime_loader is None:
            # Direct library use is an execution path too.  The canonical CLI
            # performs the same check before model selection; repeating it here
            # prevents a caller from loading a board model on an unknown or
            # mismatched host by bypassing that entrypoint.
            from samples._shared.platforms import require_execution_target
            requested = selection.target
            if requested is None:
                requested = getattr(selection.platform, "key", selection.platform)
            try:
                require_execution_target(requested)
            except Exception as exc:
                raise RunnerError(str(exc)) from exc
            # Keep the import lazy: --help, dry-runs and host-side inspection do
            # not need the board-only SDK.  Import the system module directly
            # so this package does not inherit the legacy top-level module
            # graph used by compatibility tasks.
            try:
                import hbm_runtime
            except ImportError as exc:
                raise RunnerError(
                    "hbm_runtime is not installed; on-board inference requires "
                    "the RDK system image.") from exc
            runtime_loader = lambda: hbm_runtime
        try:
            runtime_module = runtime_loader()
            model = runtime_module.HB_HBMRuntime(selection.model_path)
        except Exception as exc:
            if isinstance(exc, RunnerError):
                raise
            raise RunnerError(f"Unable to load model {selection.model_path!r}: {exc}") from exc
        try:
            metadata = RuntimeMetadata.from_runtime(model)
        except Exception as exc:
            raise RunnerError(f"Runtime metadata is not readable: {exc}") from exc
        try:
            binding = bind_model(selection, metadata)
        except BindingError as exc:
            raise RunnerError(str(exc)) from exc
        return cls(model, binding, metadata)

    @classmethod
    def from_config(cls,
                    config: Any,
                    runtime_loader: Optional[Callable[[], Any]] = None) -> "ModelRunner":
        return cls.from_selection(_selection_from_config(config), runtime_loader)

    def prepare_input(self, y_plane: Any, uv_plane: Any):
        """Pack preprocessed NV12 planes through the bound adapter."""
        return self.input_adapter.build(y_plane, uv_plane)

    def __call__(self, prepared_input: Mapping[str, Any]):
        """Execute once and return role-keyed raw arrays without changing their values."""
        try:
            raw_outputs = self.model.run(prepared_input)
        except Exception as exc:
            raise RunnerError(f"Model execution failed: {exc}") from exc
        try:
            return self.binding.read_raw_outputs(raw_outputs)
        except BindingError as exc:
            raise RunnerError(str(exc)) from exc

    def run(self, prepared_input: Mapping[str, Any]):
        """Compatibility alias for the callable runner interface."""
        return self(prepared_input)

    def forward(self, prepared_input: Mapping[str, Any]):
        """Compatibility alias used by older task wrappers."""
        return self(prepared_input)

    def set_scheduling_params(self, **kwargs: Any) -> None:
        """Forward only explicitly requested scheduling parameters."""
        values = {}
        if "priority" in kwargs and kwargs["priority"] is not None:
            values["priority"] = {self.model_name: kwargs["priority"]}
        if "bpu_cores" in kwargs and kwargs["bpu_cores"] is not None:
            values["bpu_cores"] = {self.model_name: kwargs["bpu_cores"]}
        if not values:
            return
        method = getattr(self.model, "set_scheduling_params", None)
        if method is None:
            raise RunnerError(
                "The selected runtime does not support explicit scheduling parameters.")
        method(**values)


def build_runner(selection: Any,
                 runtime_loader: Optional[Callable[[], Any]] = None) -> ModelRunner:
    """Factory accepting either a ``ModelSelection`` or legacy config."""
    if isinstance(selection, ModelSelection) or all(
            hasattr(selection, name) for name in ("model_path", "task", "contract")):
        return ModelRunner.from_selection(selection, runtime_loader)
    return ModelRunner.from_config(selection, runtime_loader)


__all__ = ["RunnerError", "ModelRunner", "build_runner"]
