"""Reusable single-image classification flow for the unified samples.

The task owns image preprocessing and Top-K decoding.  A runner owns model
loading and one model invocation; this keeps the same flow usable with a
host fixture or a board runtime without importing the board SDK here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from samples._shared.cls_binding import (
    SCORE_POLICIES,
    MetadataMismatchError,
    ModelBinding,
    score_vector_shape,
)
from samples._shared.quantization import apply_output_transform
from samples._shared.tensor_io import PreparedInput, prepare_nv12


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

    Stage data flow (inference-contract §2), concretized for classification:

    - ``Input``: one BGR ``uint8`` image, shape ``(height, width, 3)``.
    - ``Tensors``: the bound NV12 physical mapping (packed single tensor on
      X5, split Y/UV pair on S-series), ``uint8``, validated shapes.
    - ``Context``: :class:`ImageTransform` carried by
      :class:`PreparedInput.transform` — per-call geometry (resize, scale,
      padding).  Classification consumes no geometry in ``post_process``, so
      the stage omits the context argument (see inference-contract §1) and
      the transform exists for callers and debugging.
    - ``RawOutputs``: the runner's validated F32 score tensor in the bound
      shape; ``forward`` performs container adaptation only — no decode,
      activation, or file access.
    - ``Result``: :class:`ClassificationResult` with descending Top-K class
      IDs, scores, and labels.
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
        """Decode the bound score vector using the declared source policy.

        Phase 1.5 H1: ``post_process`` owns the declared
        ``output_transform`` (``raw_f32`` passthrough or ``dequant`` with the
        binding's quantization descriptors); ``forward``/the runner only
        adapt containers.  The contract's ``output_score_policy`` then
        selects the source-proven math: ``legacy_softmax``/``softmax``
        normalize before Top-K (the legacy name keeps the pilot's
        unverified-graph-semantics warning), ``none`` keeps the raw values
        exactly as the sources that print already-activated outputs.
        """

        raw = _extract_output(outputs, self.binding)
        values = apply_output_transform(
            self.binding.contract.output_transform,
            {self.binding.output_name: raw},
            self.binding.output_quants,
        )
        scores = values[self.binding.output_name]
        policy = self.binding.contract.output_score_policy
        if policy not in SCORE_POLICIES:
            raise MetadataMismatchError(
                "Unsupported classification output score policy "
                f"{policy!r}."
            )
        return topk_from_scores(
            scores, self.top_k, self.labels, softmax=policy != "none"
        )

    def predict(self, image: np.ndarray) -> ClassificationResult:
        """Run preprocessing, model invocation, and post-processing."""

        prepared = self.pre_process(image)
        outputs = self.forward(prepared)
        return self.post_process(outputs)

    def __call__(self, image: np.ndarray) -> ClassificationResult:
        return self.predict(image)


def topk_from_scores(
    scores: np.ndarray | Sequence[float],
    top_k: int = 5,
    labels: Optional[Mapping[int, str] | Sequence[str]] = None,
    *,
    softmax: bool = True,
) -> ClassificationResult:
    """Convert one finite score vector into stable descending Top-K results.

    ``softmax=True`` reproduces the source wrappers that normalize the
    vector before Top-K (numerically stable); ``softmax=False`` preserves
    the wrappers that sort and print already-activated values untouched.
    Both paths accept the known singleton-batch/singleton-spatial forms and
    reject ambiguous batches or extra dimensions instead of silently
    flattening them.
    """

    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an integer, got {top_k!r}.")
    array = np.asarray(scores)
    if array.size == 0:
        raise ValueError("Classification scores are empty.")
    if array.ndim > 1:
        if array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"Expected one score vector, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Classification scores must be numeric, got {array.dtype}.")
    vector = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Classification scores contain NaN or infinity.")
    if top_k <= 0 or top_k > vector.shape[0]:
        raise ValueError(f"top_k must be between 1 and {vector.shape[0]}, got {top_k}.")

    if softmax:
        shifted = vector - np.max(vector)
        probabilities = np.exp(shifted)
        denominator = float(np.sum(probabilities))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Classification scores cannot be normalized to probabilities.")
        probabilities /= denominator
    else:
        probabilities = vector
    # Stable descending ordering preserves lower class IDs on equal scores.
    indices = np.argsort(-probabilities, kind="stable")[: int(top_k)]
    top_scores = probabilities[indices].astype(np.float32, copy=True)
    class_ids = indices.astype(np.int64, copy=True)
    resolved_labels = tuple(_label_for(labels, int(index)) for index in class_ids)
    return ClassificationResult(class_ids=class_ids, scores=top_scores, labels=resolved_labels)


def topk_from_logits(
    scores: np.ndarray | Sequence[float],
    top_k: int = 5,
    labels: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> ClassificationResult:
    """Compatibility alias for callers of the former X5 helper.

    The samples do not use this name to assert that a runtime tensor is a
    verified logits tensor; :func:`topk_from_scores` is the canonical API.
    """

    return topk_from_scores(scores, top_k, labels)


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
    # H4 rank rule replaces the dual (1,1000,1,1)/(1,1000) hard-code.
    if not score_vector_shape(array.shape, binding.contract.class_count):
        raise MetadataMismatchError(
            f"Runner output {binding.output_name!r} shape {array.shape} does not "
            f"squeeze to the bound ({binding.contract.class_count},) score vector."
        )
    if binding.output_transform == "raw_f32" and array.dtype != np.dtype("float32"):
        raise MetadataMismatchError(
            f"Runner output {binding.output_name!r} dtype {array.dtype} does not "
            "match the declared raw_f32 contract."
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
