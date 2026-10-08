"""Run MobileNetV2Classifier preprocessing, board inference, and Top-K decoding.

Shared utilities provide image reading, labels, tensor preparation, and SDK calls.
"""

from __future__ import annotations

from samples.vision.mobilenetv2.runtime.python.model_binding import BINDING_TABLE

from pathlib import Path
from utils.py_utils.image import read_bgr_image
from utils.py_utils.labels import validate_labels
from typing import Mapping, Optional, Sequence

import numpy as np

from utils.py_utils.classification import (
    ClassificationResult,
    extract_score_tensor,
    topk_from_scores,
)
from utils.py_utils.cls_binding import (
    SCORE_POLICIES,
    MetadataMismatchError,
    ModelSelection,
)
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.tensor_io import PreparedInput, prepare_nv12

from utils.py_utils.model_runner import RuntimeModelRunner


class MobileNetV2Classifier:
    """MobileNetV2 image classifier with the full pipeline readable in one place.

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
            callers leave it unset and get the shared runner backed by the
            shared SDK session.

    Attributes:
        selection (ModelSelection): Selected model path, target, and contract.
        runner (RuntimeModelRunner): Loaded shared runtime used by infer.
        binding (ModelBinding): Validated tensor names and input/output metadata.
        labels (Mapping[int, str] | Sequence[str] | None): Optional class names.
        top_k (int): Number of ranked classes per image.
        resize_type (int | None): Explicit resize policy or the contract default.
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
        """Load the selected model and configure its classification stages.

        Args:
            selection: Published model path, concrete target, and input/output contract
                returned by this sample's resolve_selection.
            top_k: Number of ranked classes in [1, class_count]; defaults to 5.
            labels: Optional full class-name sequence or sparse integer/name mapping.
                Missing names are rendered as class IDs.
            resize_type: 0 stretches; 1 applies letterbox; None uses the model contract.
            runner: Optional injected runner; otherwise use the shared runtime with
                this sample's binding table.

        Returns:
            None.

        Raises:
            ValueError: Top-K, labels, or execution target is invalid.
            TypeError: Labels are not a supported sequence or mapping.
            BindingError: Model metadata violates the selected contract.
            RuntimeError: The SDK cannot load the model.
        """
        self.selection = selection
        self.runner = runner if runner is not None else RuntimeModelRunner(selection, table=BINDING_TABLE)
        # Loading validates the board identity (via the shared SDK session),
        # imports the SDK, constructs the model and binds its actual tensor
        # metadata against the selection contract.
        self.binding = self.runner.load()
        class_count = self.binding.contract.class_count
        if top_k <= 0 or top_k > class_count:
            raise ValueError(
                f"top_k must be between 1 and {class_count}, got {top_k}.")
        self.top_k = int(top_k)
        self.labels = validate_labels(labels, class_count)
        self.resize_type = resize_type


    def preprocess(self, source: "str | Path | np.ndarray") -> PreparedInput:
        """Prepare one image as the model's physical NV12 input tensors.

        Args:
            source: Image path or uint8 BGR array shaped (H, W, 3), with values
                in [0, 255] and positive dimensions. The array is not modified.

        Returns:
            PreparedInput: Tensor-name mapping of contiguous uint8 NV12 bytes,
            plus this call's resize/padding transform. At model size (h, w), X5
            uses one flat (h*w*3//2,) tensor; S uses Y (1, h, w, 1) and
            UV (1, h//2, w//2, 2). Byte values lie in [0, 255].

        Raises:
            TypeError: Source is neither a path nor a NumPy array.
            FileNotFoundError: The image cannot be read.
            ValueError: Image shape/dtype, resize options, or tensor binding is invalid.
        """

        image = read_bgr_image(source) if isinstance(source, (str, Path)) else source
        if not isinstance(image, np.ndarray):
            raise TypeError("source must be an image path or a BGR NumPy array.")
        return prepare_nv12(image, self.binding, resize_type=self.resize_type)

    def infer(self, prepared: "PreparedInput | Mapping[str, np.ndarray]") -> object:
        """Execute one inference call with the prepared physical tensors.

        Args:
            prepared: PreparedInput from preprocess or its physical tensor mapping.

        Returns:
            Mapping[str, np.ndarray]: Raw tensors keyed by bound output name.
            The score tensor squeezes to (class_count,); raw_f32 uses float32,
            while dequant keeps the SDK's raw dtype. Values are not activated here.

        Raises:
            ValueError: Input names, shapes, dtype, or contiguity violate the binding.
            MetadataMismatchError: Output structure, shape, or dtype is invalid.
            RuntimeError: SDK execution fails.
        """

        tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
        return self.runner(tensors)

    def postprocess(self, outputs: object) -> ClassificationResult:
        """Transform model scores and select the highest-ranked classes.

        Args:
            outputs: Raw output mapping returned by infer. The bound score tensor
                must squeeze to (class_count,); raw_f32 requires float32.

        Returns:
            ClassificationResult: class_ids is int64 (top_k,), scores is float32
            (top_k,), and labels is a tuple of top_k strings. Scores descend;
            ties prefer lower class IDs. Softmax policies produce values in [0, 1];
            none preserves unnormalized values. Arrays are owned by the result.

        Raises:
            MetadataMismatchError: Output container, shape, dtype, or policy is invalid.
            OutputTransformError: Output data or quantization metadata is invalid.
            ValueError: Scores are nonnumeric, nonfinite, or cannot be ranked.

        Notes:
            Classification does not consume image coordinates, so no resize context
            is required. Top-K probabilities alone need not sum to one.
        """

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
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            source: Image path or uint8 BGR array shaped (H, W, 3), values [0, 255].

        Returns:
            ClassificationResult: Ranked int64 class_ids and float32 scores of
            shape (top_k,), with a matching label tuple; see postprocess for scoring.

        Raises:
            FileNotFoundError: The input image cannot be read.
            TypeError: The input type is unsupported.
            ValueError: Image data, tensors, or classification scores are invalid.
            BindingError: Model outputs violate the declared contract.
            RuntimeError: SDK execution fails.

        Notes:
            Propagates stage errors. Does not print, save, or download results.
        """

        prepared = self.preprocess(source)
        outputs = self.infer(prepared)
        return self.postprocess(outputs)


    def pre_process(self, source: "str | Path | np.ndarray") -> PreparedInput:
        """Delegate to preprocess with the same input and error contract.

        Args:
            source: Input accepted by preprocess; see that method for shape and dtype.

        Returns:
            PreparedInput: NV12 tensors and the per-call image transform.
        """

        return self.preprocess(source)

    def forward(self, prepared: "PreparedInput | Mapping[str, np.ndarray]") -> object:
        """Delegate to infer with the same input and error contract.

        Args:
            prepared: Input accepted by infer; see that method for shape and dtype.

        Returns:
            Mapping[str, np.ndarray]: Raw bound output tensors.
        """

        return self.infer(prepared)

    def post_process(self, outputs: object) -> ClassificationResult:
        """Delegate to postprocess with the same input and error contract.

        Args:
            outputs: Input accepted by postprocess; see that method for shape and dtype.

        Returns:
            ClassificationResult: Ranked class IDs, scores, and labels.
        """

        return self.postprocess(outputs)

    def __call__(self, source: "str | Path | np.ndarray") -> ClassificationResult:
        """Delegate to predict with the same input and error contract.

        Args:
            source: Input accepted by predict; see that method for shape and dtype.

        Returns:
            ClassificationResult: Ranked class IDs, scores, and labels.
        """
        return self.predict(source)

    def set_scheduling_params(self, *, priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional list of nonnegative BPU core indexes. The SDK
                determines which indexes are supported on the selected board.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """

        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)


__all__ = ["MobileNetV2Classifier"]
