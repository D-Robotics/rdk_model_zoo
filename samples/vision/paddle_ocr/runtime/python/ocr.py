# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Readable two-stage PaddleOCR composition.

``ocr.py`` owns the complete pipeline locally: the injectable two-stage
:class:`OCRPipeline`, the pure CTC decoding, the crop/rotate geometry and the
target-local NV12/tensor input preparation. Physical runners and tensor
contracts live in ``backend.py``; published pair selection lives in
``cli.py``.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from samples.vision.paddle_ocr.runtime.python.backend import RuntimeStageRunner
from samples.vision.paddle_ocr.runtime.python.cli import OCRPair, resolve_pair

# ======================================================================
# Pure, finite CTC post-processing for the recognizers.
# ======================================================================

def ctc_decode_indices(
    indices: np.ndarray,
    tokens: Sequence[str],
    *,
    raw: bool = False,
    blank_symbol: str = "",
) -> str:
    """Decode an already-argmaxed CTC index sequence.

    This is the shared primitive used by the canonical recognizer and the
    legacy compatibility converter.  ``raw=True`` intentionally keeps every
    class, including blanks, for callers that expose the historical diagnostic
    string; normal decoding applies the usual blank and repeat collapse.
    """

    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"CTC indices must be one-dimensional, got {values.shape}.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"CTC indices must be integers, got {values.dtype}.")
    token_table = tuple(tokens)
    if not token_table or token_table[0] != "blank":
        raise ValueError("CTC token table must contain 'blank' at class index 0.")
    if any(not isinstance(token, str) for token in token_table):
        raise ValueError("CTC token table entries must be strings.")

    result: list[str] = []
    previous = -1
    for value in values:
        index = int(value)
        if index < 0 or index >= len(token_table):
            raise ValueError(
                f"CTC class index {index} is outside the token table of length "
                f"{len(token_table)}."
            )
        if raw:
            result.append(blank_symbol if index == 0 else token_table[index])
        elif index != 0 and index != previous:
            result.append(token_table[index])
        previous = index
    return "".join(result)


def ctc_greedy_decode(scores: np.ndarray, tokens: Sequence[str]) -> str:
    """Decode one CTC score tensor using the source-compatible best path.

    ``scores`` may be ``[T,V]`` or ``[1,T,V]``.  Class zero is the blank.  A
    blank resets the repeated-token state, matching both audited Python
    wrappers; no softmax or other activation is inserted.

    Args:
        scores: Float32 score/logit tensor with one sequence.
        tokens: Token strings indexed by the final class dimension.

    Returns:
        The decoded Unicode text.

    Raises:
        ValueError: If rank, class count, dtype, or finite-value contracts are
            not satisfied.
    """

    values = np.asarray(scores)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"CTC decoding accepts one batch, got {values.shape}.")
        values = values[0]
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError(f"CTC scores must have shape (T,V) or (1,T,V), got {values.shape}.")
    if values.dtype != np.dtype("float32"):
        raise ValueError(f"CTC scores must be float32, got {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("CTC scores contain NaN or infinity.")

    token_table = tuple(tokens)
    if len(token_table) != values.shape[1]:
        raise ValueError(
            f"CTC class count {values.shape[1]} does not match {len(token_table)} tokens."
        )
    if not token_table or token_table[0] != "blank":
        raise ValueError("CTC token table must contain 'blank' at class index 0.")
    if any(not isinstance(token, str) for token in token_table):
        raise ValueError("CTC token table entries must be strings.")

    indices = np.argmax(values, axis=1)
    return ctc_decode_indices(indices, token_table)

# ======================================================================
# Target-local geometry policies used by the detector stage.
# ======================================================================

def dilate_contours(
    contours: Iterable[np.ndarray],
    *,
    target: str,
    ratio_prime: float = 2.7,
) -> tuple[np.ndarray, ...]:
    """Expand contours with the target's source DB/pyclipper policy."""

    if target not in ("x5", "s100"):
        raise ValueError(f"Unsupported OCR geometry target {target!r}.")
    if not np.isfinite(ratio_prime) or ratio_prime < 0:
        raise ValueError("ratio_prime must be a finite non-negative number.")
    import cv2
    contours = tuple(contours)
    if not contours:
        return ()
    try:
        import pyclipper
    except ImportError as exc:  # pragma: no cover - exercised on SDK hosts
        raise RuntimeError(
            "pyclipper is required for OCR detection post-processing; install it "
            "for the execution environment."
        ) from exc

    expanded: list[np.ndarray] = []
    for contour in contours:
        polygon = np.asarray(contour)[:, 0, :]
        arc_length = cv2.arcLength(polygon, True)
        if arc_length == 0:
            continue
        distance = cv2.contourArea(polygon) * ratio_prime / arc_length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = offset.Execute(distance)
        if target == "s100":
            # S100's source filters multi-polygons before conversion because
            # recent NumPy rejects ragged nested lists.
            if len(solution) != 1 or len(solution[0]) == 0:
                continue
            polygon_array = np.array(solution, dtype=np.int64)
        else:
            # X5's source converts first.  A ragged multi-polygon therefore
            # retains its observable NumPy ValueError behavior.
            polygon_array = np.array(solution)
        if polygon_array.size == 0 or len(polygon_array) != 1:
            continue
        expanded.append(polygon_array)
    return tuple(expanded)


def get_bounding_boxes(
    polygons: Iterable[np.ndarray],
    *,
    min_area: float = 100,
) -> tuple[np.ndarray, ...]:
    """Return source-compatible integer minimum-area rectangles."""

    if not np.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be a finite non-negative number.")
    import cv2

    boxes: list[np.ndarray] = []
    for polygon in polygons:
        if cv2.contourArea(polygon) < min_area:
            continue
        rectangle = cv2.minAreaRect(polygon)
        boxes.append(cv2.boxPoints(rectangle).astype(np.int64))
    return tuple(boxes)


def crop_and_rotate_image(
    image: np.ndarray,
    box: np.ndarray,
    *,
    target: str,
) -> np.ndarray:
    """Perspective-crop one detected box with the target's edge policy."""

    if target not in ("x5", "s100"):
        raise ValueError(f"Unsupported OCR geometry target {target!r}.")
    import cv2

    rectangle = cv2.minAreaRect(box)
    box_points = cv2.boxPoints(rectangle).astype(np.intp)
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    angle = rectangle[2]

    if target == "x5" and (width == 0 or height == 0):
        # This is an observable legacy X5 behavior and is retained as a local
        # policy.  DetectionResult takes an owned copy before returning it.
        return image
    src_points = box_points.astype("float32")
    dst_points = np.array(
        [
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, transform, (width, height))
    if angle >= 45:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

# ======================================================================
# Target-local OCR image and tensor preparation.
# ======================================================================

def prepare_detection(image: np.ndarray, pair: OCRPair) -> dict[str, np.ndarray]:
    """Prepare one BGR image for the selected detector's NV12 protocol.

    X5 uses a single packed ``[1,960,640,1]`` buffer.  S100 exposes separate
    Y and UV planes.  The resize policy is part of the pair contract: X5 uses
    linear interpolation and S100 uses area interpolation.
    """

    _validate_bgr_image(image)
    import cv2

    interpolation = cv2.INTER_LINEAR if pair.target == "x5" else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (pair.detector.runtime_input_shapes[pair.detector.input_names[0]][2]
         if pair.target != "x5" else 640,
         pair.detector.runtime_input_shapes[pair.detector.input_names[0]][1]
         if pair.target != "x5" else 640),
        interpolation=interpolation,
    )
    y, uv = _bgr_to_nv12_planes(resized)

    if pair.target == "x5":
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0).reshape(
            pair.detector.runtime_input_shapes["x"]
        )
        return {"x": packed.astype(np.uint8, copy=False)}

    return {
        "x_y": y.astype(np.uint8, copy=False),
        "x_uv": uv.astype(np.uint8, copy=False),
    }


def prepare_recognition(image: np.ndarray, pair: OCRPair) -> dict[str, np.ndarray]:
    """Prepare one BGR crop as the audited RGB float32 NCHW input."""

    _validate_bgr_image(image)
    import cv2

    shape = pair.recognizer.runtime_input_shapes["x"]
    height, width = shape[2], shape[3]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Match both legacy Python wrappers' operation order: division produces a
    # float64 temporary and conversion to float32 happens before channel swap.
    resized = (resized / 255.0).astype(np.float32)
    rgb = resized[:, :, [2, 1, 0]]
    return {"x": rgb[None].transpose(0, 3, 1, 2)}


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("OCR image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"OCR image must have shape (H,W,3), got {image.shape}.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("OCR image must have non-zero height and width.")
    if image.dtype != np.dtype("uint8"):
        raise ValueError(f"OCR image must be uint8 BGR, got {image.dtype}.")


def _bgr_to_nv12_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the shared I420-to-NV12 conversion; preserve this entry's checks."""
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV12 preparation requires even image dimensions.")
    from utils.py_utils.image import bgr_to_nv12_planes as convert

    return convert(image)

# ======================================================================
# The injectable two-stage composition.
# ======================================================================

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



from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from samples.vision.paddle_ocr.runtime.python.backend import (
    StageContract,
    validate_stage_inputs,
    validate_stage_output,
)
from samples.vision.paddle_ocr.runtime.python.cli import MetadataMismatchError, OCRPair


StageRunner = Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class DetectionResult:
    """Owned, ordered detector boxes and perspective crops."""

    boxes: tuple[np.ndarray, ...]
    crops: tuple[np.ndarray, ...]
    polygons: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if len(self.boxes) != len(self.crops):
            raise ValueError("Detection boxes and crops must have equal lengths.")
        # ``polygons`` are the complete, ordered output of contour dilation.
        # ``get_bounding_boxes`` can then discard a polygon below ``min_area``;
        # consequently the two source lists are intentionally independent.
        object.__setattr__(
            self,
            "boxes",
            tuple(np.array(box, copy=True) for box in self.boxes),
        )
        object.__setattr__(
            self,
            "crops",
            tuple(np.array(crop, copy=True) for crop in self.crops),
        )
        object.__setattr__(
            self,
            "polygons",
            tuple(np.array(polygon, copy=True) for polygon in self.polygons),
        )


@dataclass(frozen=True)
class OCRResult:
    """Owned, ordered boxes and recognition strings from one image."""

    target: str
    detector_asset: str
    recognizer_asset: str
    boxes: tuple[np.ndarray, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.boxes) != len(self.texts):
            raise ValueError("OCR boxes and texts must have equal lengths.")
        object.__setattr__(
            self,
            "boxes",
            tuple(np.array(box, copy=True) for box in self.boxes),
        )
        object.__setattr__(self, "texts", tuple(str(text) for text in self.texts))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for the native CLI."""

        return {
            "target": self.target,
            "detector_asset": self.detector_asset,
            "recognizer_asset": self.recognizer_asset,
            "boxes": [box.tolist() for box in self.boxes],
            "texts": list(self.texts),
        }


class OCRPipeline:
    """Compose preparation, an injected detector, crops and an injected rec.

    The callables receive and return flat physical tensor-name mappings.  This
    keeps host tests independent from ``hbm_runtime`` and makes each stage
    individually comparable with the original source captures.

    Stage data flow (inference-contract §2), concretized for two-stage OCR:

    - ``Input``: one BGR ``uint8`` image (pipeline level); one crop
      (recognition stage level).
    - ``Tensors``: detection — target-specific NV12 mapping (packed on X5,
      split on S100); recognition — shared RGB float32 NCHW crop tensor.
    - ``Context``: detection keeps the original image geometry (the mask is
      resized back to it in ``postprocess_detection``); recognition carries
      the crop identity through the per-crop loop in ``predict``.
    - ``RawOutputs``: each stage's validated flat output mapping;
      ``forward_*`` performs structural validation and container adaptation
      only — no thresholding, decoding, or file access.
    - ``Result``: :class:`DetectionResult` (stage) / :class:`OCRResult`
      (pipeline), owned and ordered.

    Each stage exposes the public three-step interface
    (``prepare_*`` / ``forward_*`` / ``postprocess_*`` or ``decode_*``);
    ``run_*`` composes the three steps of one stage and ``predict`` composes
    detection → ordered cropping → recognition with per-stage, per-crop error
    attribution.
    """

    def __init__(
        self,
        pair: OCRPair,
        detector_runner: StageRunner,
        recognizer_runner: StageRunner,
        *,
        vocabulary_path: str | None = None,
        threshold: float = 0.5,
        ratio_prime: float = 2.7,
        min_area: float = 100,
    ) -> None:
        """Create a pipeline around one finite pair and two runner callables."""

        if not callable(detector_runner) or not callable(recognizer_runner):
            raise TypeError("detector_runner and recognizer_runner must be callable.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number in [0, 1].")
        if not np.isfinite(ratio_prime) or ratio_prime < 0:
            raise ValueError("ratio_prime must be a finite non-negative number.")
        if not np.isfinite(min_area) or min_area < 0:
            raise ValueError("min_area must be a finite non-negative number.")
        self.pair = pair
        self.detector_runner = detector_runner
        self.recognizer_runner = recognizer_runner
        self.vocabulary_path = vocabulary_path
        self.threshold = float(threshold)
        self.ratio_prime = float(ratio_prime)
        self.min_area = float(min_area)
        self._tokens: tuple[str, ...] | None = None

    @classmethod
    def from_models(
        cls,
        pair: OCRPair,
        *,
        vocabulary_path: str | None = None,
        threshold: float = 0.5,
        ratio_prime: float = 2.7,
        min_area: float = 100,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
        detector_runtime_factory=None,
        recognizer_runtime_factory=None,
        detector_runtime=None,
        recognizer_runtime=None,
    ) -> "OCRPipeline":
        """Construct the pipeline with two lazy stage runtimes.

        Builds the lazy detector and recognizer runners for the pair and wraps
        them in the readable composition, so ``main`` visibly constructs the
        pipeline and calls ``predict`` instead of creating runners or loading
        models itself. Runtime factories and prebuilt runtimes are the
        documented host-test seams.

        Args:
            pair: Resolved finite detector/recognizer pair from
                ``resolve_pair``.
            vocabulary_path: Optional recognizer vocabulary path override.
            threshold: Detection binarization threshold (source default 0.5).
            ratio_prime: Unclip ratio for box expansion (source default 2.7).
            min_area: Minimum kept contour area (source default 100).
            priority: Optional integer in [0, 255] applied to both stages.
            bpu_cores: Optional non-empty list of nonnegative BPU core
                indexes applied to both stages.
            detector_runtime_factory: Optional detector SDK-object factory.
            recognizer_runtime_factory: Optional recognizer SDK-object factory.
            detector_runtime: Optional prebuilt detector SDK object.
            recognizer_runtime: Optional prebuilt recognizer SDK object.

        Returns:
            OCRPipeline: Composition ready for ordered ``predict`` calls.

        Raises:
            TypeError: An injected runner is not callable.
            ValueError: Pipeline parameters or local model settings are
                invalid.

        Notes:
            Each stage loads its model, validates metadata and applies scheduling
            before its first inference call. If detection yields no crops, the
            recognizer stays unloaded. Board or SDK failures propagate from
            stage execution through predict.
        """
        from samples.vision.paddle_ocr.runtime.python.backend import (
            create_stage_runners,
        )

        detector_runner, recognizer_runner = create_stage_runners(
            pair,
            priority=priority,
            bpu_cores=bpu_cores,
            detector_runtime_factory=detector_runtime_factory,
            recognizer_runtime_factory=recognizer_runtime_factory,
            detector_runtime=detector_runtime,
            recognizer_runtime=recognizer_runtime,
        )
        return cls(
            pair,
            detector_runner,
            recognizer_runner,
            vocabulary_path=vocabulary_path,
            threshold=threshold,
            ratio_prime=ratio_prime,
            min_area=min_area,
        )

    def prepare_detection(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare the exact target-specific detector input mapping."""

        return prepare_detection(image, self.pair)

    def prepare_recognition(self, crop: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare one crop as the shared RGB float32 NCHW input."""

        return prepare_recognition(crop, self.pair)

    def forward_detection(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Invoke the detector runner and validate its raw output mapping.

        Container adaptation only (inference-contract §3): no thresholding,
        decoding, drawing, or file access happens here.
        """

        return self._call_runner(
            self.detector_runner, self.pair.detector, inputs, "detector"
        )

    def forward_recognition(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Invoke the recognizer runner and validate its raw output mapping.

        Container adaptation only (inference-contract §3): no CTC decoding,
        activation, or file access happens here.
        """

        return self._call_runner(
            self.recognizer_runner, self.pair.recognizer, inputs, "recognizer"
        )

    def run_detection(self, image: np.ndarray) -> DetectionResult:
        """Run detection and return boxes/crops for independent comparison."""

        _validate_bgr_image(image)
        try:
            inputs = self.prepare_detection(image)
            outputs = self.forward_detection(inputs)
            return self.postprocess_detection(outputs, image)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("detector stage"):
                raise
            raise RuntimeError(f"detector stage failed: {exc}") from exc

    def postprocess_detection(
        self,
        outputs: Mapping[str, Any],
        image: np.ndarray,
    ) -> DetectionResult:
        """Validate a detector output and expose ordered boxes and crops."""

        _validate_bgr_image(image)
        validated = validate_stage_output(self.pair.detector, outputs)
        raw = validated[self.pair.detector.output_name]
        # The static binding guarantees [1,1,640,640].  Keep the reshape
        # explicit so an accidental rank-compatible class map cannot slip in.
        mask = np.where(raw.reshape(1, 640, 640)[0] > self.threshold, 255, 0).astype(
            np.uint8
        )
        import cv2

        height, width = image.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = dilate_contours(
            contours, target=self.pair.target, ratio_prime=self.ratio_prime
        )
        boxes = get_bounding_boxes(polygons, min_area=self.min_area)
        crops = tuple(
            crop_and_rotate_image(image, box, target=self.pair.target)
            for box in boxes
        )
        return DetectionResult(boxes=boxes, crops=crops, polygons=polygons)

    def run_recognition(self, crop: np.ndarray) -> str:
        """Run recognition for one crop through the bound recognizer."""

        try:
            inputs = self.prepare_recognition(crop)
            outputs = self.forward_recognition(inputs)
            return self.decode_recognition(outputs)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("recognizer stage"):
                raise
            raise RuntimeError(f"recognizer stage failed: {exc}") from exc

    def decode_recognition(self, outputs: Mapping[str, Any]) -> str:
        """Validate and decode one recognizer output without activation."""

        validated = validate_stage_output(self.pair.recognizer, outputs)
        raw = validated[self.pair.recognizer.output_name]
        if self.pair.target == "x5":
            scores = raw.reshape(1, 40, 97)
        else:
            scores = raw.reshape(1, 40, 18710)
        if self._tokens is None:
            self._tokens = self.pair.vocabulary.load_tokens(self.vocabulary_path)
        return ctc_greedy_decode(scores, self._tokens)

    def predict(self, image: np.ndarray) -> OCRResult:
        """Run detection, ordered cropping and recognition for one BGR image."""

        detection = self.run_detection(image)
        texts: list[str] = []
        for index, crop in enumerate(detection.crops):
            try:
                texts.append(self.run_recognition(crop))
            except Exception as exc:
                raise RuntimeError(
                    f"recognizer stage failed for crop {index}: {exc}"
                ) from exc
        return OCRResult(
            target=self.pair.target,
            detector_asset=self.pair.detector_asset,
            recognizer_asset=self.pair.recognizer_asset,
            boxes=detection.boxes,
            texts=tuple(texts),
        )

    def __call__(self, image: np.ndarray) -> OCRResult:
        """Call ``predict`` using the functional pipeline spelling."""

        return self.predict(image)

    @staticmethod
    def _call_runner(
        runner: StageRunner,
        contract: StageContract,
        inputs: Mapping[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        prepared = validate_stage_inputs(contract, inputs)
        try:
            outputs = runner(prepared)
        except Exception as exc:
            raise RuntimeError(f"{stage} runner call failed: {exc}") from exc
        if not isinstance(outputs, Mapping):
            raise MetadataMismatchError(
                f"{stage} runner must return a flat mapping, got {type(outputs).__name__}."
            )
        return validate_stage_output(contract, outputs)


__all__ = ["DetectionResult", "OCRPipeline", "OCRResult", "StageRunner"]
