"""Compatibility adapters for the former X5 and S-series ResNet wrappers.

The old samples exposed two slightly different Python APIs even though they
implemented the same image operation and score decoding.  This module keeps
those public method signatures and return shapes while delegating all model
selection, metadata validation, NV12 conversion, execution, and Top-K math to
the canonical ResNet18 runtime.

The optional ``runtime`` and ``runtime_factory`` keyword arguments are host
test seams.  They are deliberately absent from the old configuration objects;
normal callers still construct the board SDK through ``hbm_runtime`` only when
the adapter is instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from samples.vision.resnet.runtime.python.classification import topk_from_scores
from samples.vision.resnet.runtime.python.labels import load_labels
from samples.vision.resnet.runtime.python.model_binding import resolve_selection
from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner


_ASSET_IDS = {
    "x5": "x5:resnet:resnet18_224x224_nv12.bin",
    "s100": "s:resnet18:s100/resnet18_224x224_nv12.hbm",
    "s600": "s:resnet18:s600/resnet18_224x224_nv12.hbm",
}


@dataclass
class ResNetConfig:
    """Configuration retained for the legacy X5 ``ResNet`` API."""

    model_path: str
    label_file: Optional[str] = None
    resize_type: int = 1
    topk: int = 5
    target: str = "x5"
    asset_id: Optional[str] = None


@dataclass
class Resnet18Config:
    """Configuration retained for the legacy S-series ``Resnet18`` API."""

    model_path: str = "../../model/s100/resnet18_224x224_nv12.hbm"
    resize_type: int = 1
    target: Optional[str] = None
    asset_id: Optional[str] = None


class _AdapterBase:
    """Small shared adapter surface; numerical work stays in canonical code."""

    def _load_runtime(
        self,
        *,
        target: str,
        model_path: str,
        asset_id: Optional[str],
        runtime: Any,
        runtime_factory: Any,
    ) -> None:
        qualified_id = asset_id or _ASSET_IDS[target]
        selection = resolve_selection(
            target,
            asset_id=qualified_id,
            model_path=model_path,
        )
        self._runner = RuntimeModelRunner(
            selection,
            runtime=runtime,
            runtime_factory=runtime_factory,
        )
        self._binding = self._runner.load()
        # The old classes exposed these metadata attributes directly.  Keep
        # their values sourced from the validated canonical binding.
        self.model = self._runner.runtime
        self.model_name = self._binding.model_name
        self.input_names = self._binding.input_names
        self.output_names = (self._binding.output_name,)
        self.input_shapes = self._binding.input_shapes
        self.input_h = self._binding.contract.input_height
        self.input_w = self._binding.contract.input_width

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Preserve the former positional scheduling method."""

        self._runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def _forward_flat(self, inputs: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
        if not isinstance(inputs, Mapping):
            raise ValueError("Prepared inputs must be a mapping.")
        nested = inputs.get(self.model_name, inputs)
        if not isinstance(nested, Mapping):
            raise ValueError("Prepared inputs must contain a tensor mapping.")
        return self._runner(nested)

    def _extract_scores(self, outputs: Any) -> np.ndarray:
        if isinstance(outputs, np.ndarray):
            return outputs
        if not isinstance(outputs, Mapping):
            raise ValueError("Runtime outputs must be a mapping.")
        if self._binding.output_name in outputs:
            return np.asarray(outputs[self._binding.output_name])
        nested = outputs.get(self.model_name)
        if isinstance(nested, Mapping) and self._binding.output_name in nested:
            return np.asarray(nested[self._binding.output_name])
        raise ValueError(
            f"Runtime output does not contain {self._binding.output_name!r}."
        )

    @staticmethod
    def _effective_topk(value: Optional[int], configured: int) -> int:
        # X5's former implementation used ``topk or cfg.topk``; retain that
        # documented default for callers that pass zero explicitly.
        return configured if value is None or value == 0 else int(value)


class ResNet(_AdapterBase):
    """Source-compatible X5 wrapper backed by the canonical runtime."""

    def __init__(
        self,
        config: ResNetConfig,
        *,
        runtime: Any = None,
        runtime_factory: Any = None,
    ) -> None:
        self.cfg = config
        target = str(config.target).strip().lower()
        if target != "x5":
            raise ValueError("The legacy ResNet wrapper only accepts target='x5'.")
        self._load_runtime(
            target=target,
            model_path=config.model_path,
            asset_id=config.asset_id,
            runtime=runtime,
            runtime_factory=runtime_factory,
        )
        self.labels = (
            load_labels(Path(config.label_file).expanduser())
            if config.label_file
            else {}
        )

    def pre_process(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        from samples.vision.resnet.runtime.python.classification import ClassificationTask

        task = ClassificationTask(
            self._runner,
            self._binding,
            top_k=self.cfg.topk,
            resize_type=self.cfg.resize_type if resize_type is None else resize_type,
        )
        prepared = task.pre_process(image)
        return {self.model_name: dict(prepared.tensors)}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return dict(self._forward_flat(inputs))

    def post_process(
        self,
        outputs: Dict[str, np.ndarray],
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        result = topk_from_scores(
            self._extract_scores(outputs),
            self._effective_topk(topk, self.cfg.topk),
            self.labels,
        )
        return result.class_ids, result.scores, list(result.labels)

    def predict(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        start = time.perf_counter()
        inputs = self.pre_process(image, resize_type)
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        outputs = self.forward(inputs)
        inference_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = self.post_process(outputs, topk)
        postprocess_ms = (time.perf_counter() - start) * 1000
        print(
            f"\n[Log] Pre-process: {preprocess_ms:.2f} ms | "
            f"Inference: {inference_ms:.2f} ms | "
            f"Post-process: {postprocess_ms:.2f} ms"
        )
        return result

    def __call__(
        self,
        image: np.ndarray,
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return self.predict(image, resize_type, topk)


class Resnet18(_AdapterBase):
    """Source-compatible S100/S600 wrapper backed by the canonical runtime."""

    def __init__(
        self,
        config: Resnet18Config,
        *,
        runtime: Any = None,
        runtime_factory: Any = None,
    ) -> None:
        self.cfg = config
        target = _resolve_s_target(config.target, config.model_path)
        self._load_runtime(
            target=target,
            model_path=config.model_path,
            asset_id=config.asset_id,
            runtime=runtime,
            runtime_factory=runtime_factory,
        )

    def pre_process(
        self,
        img: np.ndarray,
        resize_type: Optional[int] = None,
        image_format: Optional[str] = "BGR",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")
        from samples.vision.resnet.runtime.python.classification import ClassificationTask

        if resize_type is not None:
            self.cfg.resize_type = resize_type
        task = ClassificationTask(
            self._runner,
            self._binding,
            top_k=5,
            resize_type=self.cfg.resize_type,
        )
        prepared = task.pre_process(img)
        return {self.model_name: dict(prepared.tensors)}

    def forward(
        self,
        input_tensor: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        return {self.model_name: dict(self._forward_flat(input_tensor))}

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        topk: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        result = topk_from_scores(
            self._extract_scores(outputs),
            self._effective_topk(topk, 5),
        )
        return [
            (int(class_id), float(score))
            for class_id, score in zip(result.class_ids, result.scores)
        ]

    def predict(
        self,
        img: np.ndarray,
        image_format: str = "BGR",
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        input_tensor = self.pre_process(img, resize_type, image_format)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, topk)

    def __call__(
        self,
        img: np.ndarray,
        image_format: str = "BGR",
        resize_type: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        return self.predict(img, image_format, resize_type, topk)


def _resolve_s_target(target: Optional[str], model_path: str) -> str:
    if target is not None:
        value = str(target).strip().lower()
        if value not in ("s100", "s600"):
            raise ValueError("S-series ResNet18 target must be 's100' or 's600'.")
        return value
    # The old wrapper had no target flag.  Resolve that legacy omission from
    # the shared board identity reader; a model filename or arbitrary custom
    # path must never be treated as chip evidence.  The compatibility command
    # wrapper supplies an explicit target when it is invoked with an old
    # documented S100/S600 path.
    from samples._shared.platforms import resolve_target

    detected = resolve_target("auto")
    if detected not in ("s100", "s600"):
        raise ValueError(
            f"Detected target {detected!r} has no S-series ResNet18 compatibility asset."
        )
    return detected


__all__ = ["ResNet", "ResNetConfig", "Resnet18", "Resnet18Config"]
