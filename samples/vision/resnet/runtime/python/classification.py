"""Reusable single-image classification flow for the ResNet pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from samples.vision.resnet.runtime.python.model_binding import (
    MetadataMismatchError,
    ModelBinding,
)
from samples.vision.resnet.runtime.python.tensor_io import PreparedInput, prepare_nv12


@dataclass(frozen=True)
class ClassificationResult:
    """Top-K result with stable class IDs, scores, and optional labels."""

    class_ids: np.ndarray
    scores: np.ndarray
    labels: tuple[str, ...]

    @property
    def topk_idx(self) -> np.ndarray:
        """X5-compatible name for the class index array."""

        return self.class_ids

    @property
    def topk_prob(self) -> np.ndarray:
        """X5-compatible name for the probability array."""

        return self.scores

    @property
    def topk_labels(self) -> tuple[str, ...]:
        """X5-compatible name for the label tuple."""

        return self.labels

    def as_legacy_tuple(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return the tuple shape used by the former X5 wrapper."""

        return self.class_ids, self.scores, self.labels

    def __iter__(self):
        # Keep unpacking ``ids, scores, labels = result`` useful to callers of
        # the old X5 entry while retaining a typed value object for new code.
        yield self.class_ids
        yield self.scores
        yield self.labels


class ClassificationTask:
    """One classification pipeline driven by an injected callable runner.

    The task owns image preprocessing and Top-K decoding.  A runner owns model
    loading and one model invocation; this keeps the same flow usable with a
    host fixture or a board runtime without importing the board SDK here.
    """

    def __init__(
        self,
        runner: Callable[[Mapping[str, np.ndarray]], Any],
        binding: ModelBinding,
        *,
        top_k: int = 5,
        labels: Optional[Mapping[int, str] | Sequence[str]] = None,
        resize_type: Optional[int] = None,
    ) -> None:
        if not callable(runner):
            raise TypeError("runner must be callable.")
        if top_k <= 0 or top_k > binding.contract.class_count:
            raise ValueError(
                f"top_k must be between 1 and {binding.contract.class_count}, got {top_k}.")
        self.runner = runner
        self.binding = binding
        self.top_k = top_k
        self.labels = labels
        self.resize_type = resize_type

    def pre_process(self, image: np.ndarray) -> PreparedInput:
        """Convert an application BGR image to the bound NV12 input tensors."""

        return prepare_nv12(image, self.binding, resize_type=self.resize_type)

    def forward(self, prepared: PreparedInput | Mapping[str, np.ndarray]) -> Any:
        """Invoke the injected runner on prepared physical tensors."""

        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    def post_process(self, outputs: Any) -> ClassificationResult:
        """Decode the bound score vector using the source policy.

        Both legacy wrappers apply softmax to their output vector.  The
        contract calls that policy ``legacy_softmax`` because X5 board evidence
        currently looks normalized already and does not prove graph semantics.
        """

        scores = _extract_output(outputs, self.binding)
        if self.binding.contract.output_score_policy != "legacy_softmax":
            raise MetadataMismatchError(
                "Unsupported classification output score policy "
                f"{self.binding.contract.output_score_policy!r}."
            )
        return topk_from_scores(scores, self.top_k, self.labels)

    def predict(self, image: np.ndarray) -> ClassificationResult:
        """Run preprocessing, model invocation, and post-processing."""

        prepared = self.pre_process(image)
        outputs = self.forward(prepared)
        return self.post_process(outputs)

    def __call__(self, image: np.ndarray) -> ClassificationResult:
        return self.predict(image)


def topk_from_scores(
    logits: np.ndarray | Sequence[float],
    top_k: int = 5,
    labels: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> ClassificationResult:
    """Convert one finite score vector into stable descending Top-K results.

    The source wrappers apply softmax to a single output vector.  This
    deliberately retains that behavior for the pilot even while the X5 graph
    output semantics remain unverified.  The function accepts the known
    singleton-batch/singleton-spatial forms and rejects ambiguous batches or
    extra dimensions instead of silently flattening them.
    """

    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an integer, got {top_k!r}.")
    array = np.asarray(logits)
    if array.size == 0:
        raise ValueError("Classification logits are empty.")
    if array.ndim > 1:
        if array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"Expected one logits vector, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Classification logits must be numeric, got {array.dtype}.")
    vector = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Classification logits contain NaN or infinity.")
    if top_k <= 0 or top_k > vector.shape[0]:
        raise ValueError(f"top_k must be between 1 and {vector.shape[0]}, got {top_k}.")

    shifted = vector - np.max(vector)
    probabilities = np.exp(shifted)
    denominator = float(np.sum(probabilities))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Classification logits cannot be normalized to probabilities.")
    probabilities /= denominator
    # Stable descending ordering preserves lower class IDs on equal scores.
    indices = np.argsort(-probabilities, kind="stable")[: int(top_k)]
    scores = probabilities[indices].astype(np.float32, copy=True)
    class_ids = indices.astype(np.int64, copy=True)
    resolved_labels = tuple(_label_for(labels, int(index)) for index in class_ids)
    return ClassificationResult(class_ids=class_ids, scores=scores, labels=resolved_labels)


def topk_from_logits(
    logits: np.ndarray | Sequence[float],
    top_k: int = 5,
    labels: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> ClassificationResult:
    """Compatibility alias for callers of the former X5 helper.

    The pilot does not use this name to assert that a runtime tensor is a
    verified logits tensor; :func:`topk_from_scores` is the canonical API.
    """

    return topk_from_scores(logits, top_k, labels)


def _extract_output(outputs: Any, binding: ModelBinding) -> np.ndarray:
    value: Any
    if isinstance(outputs, np.ndarray):
        value = outputs
    elif isinstance(outputs, Mapping):
        if binding.output_name in outputs:
            value = outputs[binding.output_name]
        else:
            nested = outputs.get(binding.model_name)
            if isinstance(nested, Mapping) and binding.output_name in nested:
                value = nested[binding.output_name]
            else:
                raise MetadataMismatchError(
                    f"Runner output does not contain bound tensor {binding.output_name!r}."
                )
    else:
        raise MetadataMismatchError(
            f"Runner output does not contain bound tensor {binding.output_name!r}."
        )

    array = np.asarray(value)
    if tuple(array.shape) != tuple(binding.output_shape):
        raise MetadataMismatchError(
            f"Runner output {binding.output_name!r} shape {array.shape} does not "
            f"match the bound shape {binding.output_shape}."
        )
    if array.dtype != np.dtype("float32"):
        raise MetadataMismatchError(
            f"Runner output {binding.output_name!r} dtype {array.dtype} does not "
            "match the bound F32 contract."
        )
    return array


def _label_for(labels: Optional[Mapping[int, str] | Sequence[str]], index: int) -> str:
    if labels is None:
        return str(index)
    if isinstance(labels, Mapping):
        return str(labels.get(index, index))
    if 0 <= index < len(labels):
        return str(labels[index])
    return str(index)


__all__ = [
    "ClassificationResult",
    "ClassificationTask",
    "topk_from_logits",
    "topk_from_scores",
]
