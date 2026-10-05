"""The readable EfficientViT classification model.

:class:`EfficientViTClassifier` shows one complete classification pipeline in a
single file: construction resolves and loads the model through the sample
runner, :meth:`preprocess` turns one BGR image into the bound NV12 input
tensors, :meth:`infer` executes exactly one model call, :meth:`postprocess`
decodes the score vector into Top-K results, and :meth:`predict` chains the
three steps.  The reusable pieces stay shared: NV12 packing lives in
``samples/_shared/tensor_io.py``, the stable Top-K math in
``samples/_shared/classification.py``, and model loading/identity checks in
the sample runner (backed by the thin SDK session
``samples/_shared/runtime.py``).

Selection: use
:func:`samples.vision.efficientvit.runtime.python.model_binding.resolve_selection`
for a manifest-published model; ``model_path`` is accepted only together
with an exact qualified manifest reference.

Minimal library use::

    from samples.vision.efficientvit.runtime.python.classify import EfficientViTClassifier
    from samples.vision.efficientvit.runtime.python.model_binding import resolve_selection

    model = EfficientViTClassifier(resolve_selection("x5"),
                               labels=my_labels)      # labels optional
    result = model.predict("image.jpg")  # path or BGR ndarray
    print(result.class_ids, result.scores, result.labels)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from samples._shared.classification import (
    ClassificationResult,
    extract_score_tensor,
    topk_from_scores,
)
from samples._shared.cls_binding import (
    SCORE_POLICIES,
    MetadataMismatchError,
    ModelSelection,
)
from samples._shared.quantization import apply_output_transform
from samples._shared.tensor_io import PreparedInput, prepare_nv12

from samples.vision.efficientvit.runtime.python.model_runner import RuntimeModelRunner


class EfficientViTClassifier:
    """EfficientViT image classifier with the full pipeline readable in one place.

    The object loads the model once and reuses it across ``predict`` calls.
    Each call carries its own resize/letterbox context on the prepared
    input, so consecutive images of different sizes never reuse a stale
    transform.  Prediction itself prints nothing, draws nothing and writes
    no files; presentation belongs to the caller.

    Args:
        selection: The resolved model selection (a manifest-published
            reference; a ``model_path`` override requires the exact
            qualified asset id).
        top_k: Number of results to decode (default 5).
        labels: Optional class names — a sequence of exactly
            ``class_count`` names, or a mapping of class index to name.
            Without labels the result keeps the raw class IDs.
        resize_type: 0 stretch or 1 letterbox; ``None`` (default) follows
            the bound source contract.
        runner: Optional injected runner (host-test seam).  Production
            callers leave it unset and get the sample runner backed by the
            shared SDK session.
    """

    def __init__(
        self,
        selection: ModelSelection,
        *,
        top_k: int = 5,
        labels: Optional[Mapping[int, str] | Sequence[str]] = None,
        resize_type: Optional[int] = None,
        runner: Optional[RuntimeModelRunner] = None,
    ) -> None:
        self.selection = selection
        self.runner = runner if runner is not None else RuntimeModelRunner(selection)
        # Loading validates the board identity (via the shared SDK session),
        # imports the SDK, constructs the model and binds its actual tensor
        # metadata against the selection contract.
        self.binding = self.runner.load()
        class_count = self.binding.contract.class_count
        if top_k <= 0 or top_k > class_count:
            raise ValueError(
                f"top_k must be between 1 and {class_count}, got {top_k}.")
        self.top_k = int(top_k)
        self.labels = _checked_labels(labels, class_count)
        self.resize_type = resize_type

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, source: "str | Path | np.ndarray") -> PreparedInput:
        """Read one image (path or BGR array) and pack the bound NV12 tensors.

        The input array is never modified in place; the returned
        :class:`PreparedInput` carries this call's resize/letterbox
        geometry on ``.transform``.
        """

        image = _read_source_image(source)
        return prepare_nv12(image, self.binding, resize_type=self.resize_type)

    def infer(self, prepared: PreparedInput) -> object:
        """Execute exactly one model call on the prepared NV12 tensors."""

        return self.runner(prepared.tensors)

    def postprocess(self, outputs: object) -> ClassificationResult:
        """Apply the declared output transform and decode the Top-K results."""

        raw = extract_score_tensor(outputs, self.binding)
        values = apply_output_transform(
            self.binding.contract.output_transform,
            {self.binding.output_name: raw},
            self.binding.output_quants,
        )
        scores = values[self.binding.output_name]
        policy = self.binding.contract.output_score_policy
        if policy not in SCORE_POLICIES:
            raise MetadataMismatchError(
                f"Unsupported classification output score policy {policy!r}.")
        return topk_from_scores(
            scores, self.top_k, self.labels, softmax=policy != "none")

    def predict(self, source: "str | Path | np.ndarray") -> ClassificationResult:
        """Run the full pipeline for one image path or BGR array."""

        prepared = self.preprocess(source)
        outputs = self.infer(prepared)
        return self.postprocess(outputs)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # delegates of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, source: "str | Path | np.ndarray") -> PreparedInput:
        """Compatibility alias for :meth:`preprocess`."""

        return self.preprocess(source)

    def forward(self, prepared: "PreparedInput | Mapping[str, np.ndarray]") -> object:
        """Compatibility alias for :meth:`infer` (raw tensors also accepted)."""

        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    def post_process(self, outputs: object) -> ClassificationResult:
        """Compatibility alias for :meth:`postprocess`."""

        return self.postprocess(outputs)

    def __call__(self, source: "str | Path | np.ndarray") -> ClassificationResult:
        return self.predict(source)

    def set_scheduling_params(self, *, priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """Forward explicit runtime scheduling options to the runner."""

        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)


def _read_source_image(source: "str | Path | np.ndarray") -> np.ndarray:
    """Accept one local image path or an in-memory BGR array.

    Path read failures name the exact path.  Arrays pass through
    unchanged (never modified in place); their shape/dtype validation
    happens in the shared preprocessing.
    """

    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        image = _imread_color(str(path))
        if image is None:
            raise FileNotFoundError(f"image not found or unreadable: {path}")
        return image
    raise TypeError(
        "source must be an image path or a BGR NumPy array, got "
        f"{type(source).__name__}.")


def _imread_color(path: str) -> Optional[np.ndarray]:
    """Read one BGR image; OpenCV stays a lazy, execution-time import."""

    import cv2

    return cv2.imread(path, cv2.IMREAD_COLOR)


def _checked_labels(
    labels: Optional[Mapping[int, str] | Sequence[str]], class_count: int
) -> Optional[Mapping[int, str] | Sequence[str]]:
    """Reject labels that cannot address the bound class count."""

    if labels is None:
        return None
    if isinstance(labels, Mapping):
        for key in labels:
            if not isinstance(key, (int, np.integer)) or not 0 <= int(key) < class_count:
                raise ValueError(
                    f"Label key {key!r} is not a valid class index for the bound "
                    f"{class_count}-class output.")
        return labels
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        if len(labels) != class_count:
            raise ValueError(
                f"{len(labels)} labels do not match the bound {class_count}-class "
                "output; labels must cover every class exactly.")
        return labels
    raise TypeError(
        "labels must be a mapping of class index to name, or a sequence of names.")


__all__ = ["EfficientViTClassifier"]
