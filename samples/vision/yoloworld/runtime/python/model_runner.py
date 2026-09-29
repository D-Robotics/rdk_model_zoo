# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lazy X5 runtime adapter; it returns native raw tensors unchanged."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable
import numpy as np
from samples._shared.model_runner import _default_runtime_factory
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata
from samples.vision.yoloworld.runtime.python.model_binding import ModelBinding, ModelSelection, bind_model

class RuntimeModelRunner:
    """Load hbm_runtime only on execution and preserve both raw output arrays."""
    def __init__(self, selection: ModelSelection, *, runtime: Any = None,
                 runtime_factory: Callable[[str], Any] | None = None):
        self.selection, self._runtime, self._runtime_factory = selection, runtime, runtime_factory
        self.binding: ModelBinding | None = None
        self.metadata: RuntimeMetadata | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None and self.binding is not None

    def load(self) -> ModelBinding:
        if self.loaded:
            return self.binding  # type: ignore[return-value]
        if self._runtime is None and self._runtime_factory is None:
            # Real execution path: identity and publication gates run before the
            # SDK factory is constructed.  An injected factory is the documented
            # host seam and the only way to skip them.
            from samples._shared.platforms import require_execution_target
            from samples._shared.assets import verify_asset_file
            require_execution_target(self.selection.target)
            verify_asset_file(self.selection.asset, self.selection.model_path)
        try:
            if self._runtime is None:
                factory = self._runtime_factory or _default_runtime_factory()
                self._runtime = factory(str(self.selection.model_path))
            self.metadata = RuntimeMetadata.from_runtime(self._runtime)
            self.binding = bind_model(self.selection, self.metadata)
            return self.binding
        except Exception:
            self._runtime = None
            self.metadata = None
            self.binding = None
            raise

    def set_scheduling_params(self, *, priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        binding = self.load()
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be in 0..255")
        if bpu_cores is not None and (not bpu_cores or any(core < 0 for core in bpu_cores)):
            raise ValueError("bpu_cores must be a nonempty list of nonnegative indexes")
        if priority is None and bpu_cores is None:
            return
        setter = getattr(self._runtime, "set_scheduling_params", None)
        if not callable(setter):
            raise RuntimeError("Runtime has no set_scheduling_params API")
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {binding.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {binding.model_name: list(bpu_cores)}
        setter(**kwargs)

    def __call__(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        binding = self.load()
        if set(tensors) != {binding.image_input_name, binding.text_input_name}:
            raise MetadataMismatchError("YOLOWorld requires exactly image and text input tensors.")
        image = np.asarray(tensors[binding.image_input_name])
        text = np.asarray(tensors[binding.text_input_name])
        if image.shape != (1, 3, 640, 640) or image.dtype != np.float32 or not np.isfinite(image).all():
            raise MetadataMismatchError("Image input must be finite float32[1,3,640,640].")
        if text.shape != (1, 32, 512, 1) or text.dtype != np.float32 or not np.isfinite(text).all():
            raise MetadataMismatchError("Text input must be finite float32[1,32,512,1].")
        outputs = self._runtime.run({binding.model_name: dict(tensors)})
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError("Runtime returned a non-mapping output.")
        flat = outputs.get(binding.model_name, outputs)
        if not isinstance(flat, Mapping) or set(flat) != {binding.score_output_name, binding.box_output_name}:
            raise MetadataMismatchError("Runtime output names changed from the bound protocol.")
        score = np.asarray(flat[binding.score_output_name]); box = np.asarray(flat[binding.box_output_name])
        if score.shape not in ((1, 8400, 32), (1, 8400, 32, 1)) or box.shape not in ((1, 8400, 4), (1, 8400, 4, 1)) or score.dtype != np.float32 or box.dtype != np.float32:
            raise MetadataMismatchError("Runtime outputs must preserve native F32 score/box shapes (logical [1,8400,32] and [1,8400,4], with an optional terminal singleton).")
        if not np.isfinite(score).all() or not np.isfinite(box).all():
            raise MetadataMismatchError("Runtime outputs must be finite.")
        return {binding.score_output_name: score, binding.box_output_name: box}
