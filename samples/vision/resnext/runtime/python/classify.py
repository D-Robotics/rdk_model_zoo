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
"""ResNeXt model: load, preprocess, infer, postprocess, and predict."""

from __future__ import annotations


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
)
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.tensor_io import PreparedInput, prepare_nv12

from utils.py_utils.model_runner import RuntimeModelRunner


class ResNeXtClassifier:
    """ResNeXt image classifier.

    Args:
        model_path: Compiled model path; see __init__ for tensor and target options.

    Attributes:
        runner: Shared classification Runtime adapter.
        binding: Input and output metadata validated when the model loads.
        top_k: Number of ranked classes returned by predict.
    """

    def __init__(
        self, model_path: str | Path, *, target: str,
        input_size: tuple[int, int] = (224, 224), class_count: int = 1000,
        top_k: int = 5, labels: Mapping[int, str] | Sequence[str] | None = None,
        resize_type: int | None = None, resize_interpolation: str | None = None,
        score_policy: str = "softmax", output_transform: str = "raw_f32",
        runner: RuntimeModelRunner | None = None,
    ) -> None:
        """Load the compiled model and validate its classification settings.

        Args:
            model_path: Local compiled model path; leading ~ is expanded.
            target: Artifact target: x5, s100, s100p, or s600. Must match the board.
            input_size: Positive, even (height, width) in pixels; defaults to (224, 224).
            class_count: Positive output class count; defaults to 1000.
            top_k: Number of results in [1, class_count]; defaults to 5.
            labels: Full class-name sequence, sparse index/name mapping, or None.
                Missing names are rendered as class IDs.
            resize_type: 0 stretches; 1 preserves aspect ratio with letterbox padding.
                None follows an injected binding or uses 1 for a local model.
            resize_interpolation: Direct-resize interpolation; None uses linear on
                X5 and nearest on S. Letterbox uses linear interpolation.
            score_policy: softmax or legacy_softmax normalizes scores; none preserves
                values. Defaults to softmax.
            output_transform: raw_f32 passes through float32 output; dequant applies
                SDK quantization metadata before scoring.
            runner: Optional injected runner. Its loaded binding supplies the model
                contract instead of model_path, target, and tensor settings.

        Returns:
            None.

        Raises:
            ValueError: Class count, Top-K, labels, or local model settings are invalid.
            TypeError: Labels are neither a mapping nor a class-name sequence.
            FileNotFoundError: The local model file does not exist.
            BindingError: Runtime metadata or the selected target violates the contract.
            RuntimeError: Board identity, SDK availability, or model loading fails.
        """
        from utils.py_utils.model_runner import RuntimeModelRunner

        self.resize_type = resize_type
        self.runner = runner if runner is not None else RuntimeModelRunner.from_file(
            model_path, target=target, input_size=input_size, class_count=class_count,
            resize_type=1 if resize_type is None else resize_type,
            resize_interpolation=resize_interpolation or ("linear" if target == "x5" else "nearest"),
            score_policy=score_policy, output_transform=output_transform)
        self.binding = self.runner.load()
        class_count = self.binding.contract.class_count
        if not 1 <= top_k <= class_count:
            raise ValueError(f"top_k must be between 1 and {class_count}, got {top_k}.")
        self.labels = validate_labels(labels, class_count)
        self.top_k = int(top_k)


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


__all__ = ["ResNeXtClassifier"]
