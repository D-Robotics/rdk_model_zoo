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

"""Injectable two-stage PaddleOCR composition for the audited target pairs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from samples.vision.paddle_ocr.runtime.python.decode import ctc_greedy_decode
from samples.vision.paddle_ocr.runtime.python.geometry import (
    crop_and_rotate_image,
    dilate_contours,
    get_bounding_boxes,
)
from samples.vision.paddle_ocr.runtime.python.model_binding import (
    MetadataMismatchError,
    OCRPair,
    StageContract,
    validate_stage_inputs,
    validate_stage_output,
)
from samples.vision.paddle_ocr.runtime.python.tensor_io import (
    _validate_bgr_image,
    prepare_detection,
    prepare_recognition,
)


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

    def prepare_detection(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare the exact target-specific detector input mapping."""

        return prepare_detection(image, self.pair)

    def prepare_recognition(self, crop: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare one crop as the shared RGB float32 NCHW input."""

        return prepare_recognition(crop, self.pair)

    def run_detection(self, image: np.ndarray) -> DetectionResult:
        """Run detection and return boxes/crops for independent comparison."""

        _validate_bgr_image(image)
        try:
            inputs = self.prepare_detection(image)
            outputs = self._call_runner(
                self.detector_runner,
                self.pair.detector,
                inputs,
                "detector",
            )
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
            outputs = self._call_runner(
                self.recognizer_runner,
                self.pair.recognizer,
                inputs,
                "recognizer",
            )
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
