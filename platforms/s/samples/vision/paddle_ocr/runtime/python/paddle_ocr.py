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

"""Compatibility API for the legacy S100 PaddleOCR sample.

The canonical implementation is maintained under
``samples.vision.paddle_ocr``.  The classes below retain the old nested
tensor dictionaries and return values while delegating model binding,
preparation, geometry, and CTC decoding to the canonical implementation.
PP-OCRv6's split NV12 detector protocol remains selected by the S100 pair;
this adapter never aliases it to the X5 packed protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parents[7]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from samples.vision.paddle_ocr.runtime.python.decode import (  # noqa: E402
    ctc_greedy_decode as _canonical_ctc_greedy_decode,
)
from samples.vision.paddle_ocr.runtime.python.geometry import (  # noqa: E402
    crop_and_rotate_image,
    dilate_contours as _canonical_dilate_contours,
    get_bounding_boxes,
)
from samples.vision.paddle_ocr.runtime.python.model_binding import (  # noqa: E402
    list_available_pairs,
    resolve_pair,
)
from samples.vision.paddle_ocr.runtime.python.model_runner import (  # noqa: E402
    create_stage_runners,
)
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline  # noqa: E402


@dataclass
class PaddleOCRDetConfig:
    """Configuration retained for the original S100 detector API."""

    model_path: str = (
        "/opt/hobot/model/s100/basic/"
        "PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
    )
    ratio_prime: float = 2.7
    threshold: float = 0.5


@dataclass
class PaddleOCRRecConfig:
    """Configuration retained for the original S100 recognizer API."""

    model_path: str = (
        "/opt/hobot/model/s100/basic/"
        "PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
    )


def dilate_contours(
    contours: List[np.ndarray], ratio_prime: float
) -> List[np.ndarray]:
    """Delegate S100's source-specific contour expansion policy."""

    return list(
        _canonical_dilate_contours(
            contours,
            target="s100",
            ratio_prime=ratio_prime,
        )
    )


def ctc_greedy_decode(logits: np.ndarray, char_list: List[str]) -> str:
    """Delegate CTC best-path decoding while retaining the old name."""

    return _canonical_ctc_greedy_decode(logits, char_list)


def _pair_for_detector(path: str):
    base = list_available_pairs("s100")[0]
    return resolve_pair(
        "s100",
        det_asset_id=base.detector_asset,
        rec_asset_id=base.recognizer_asset,
        det_model_path=path,
        rec_model_path=str(base.recognizer_model_path),
    )


def _pair_for_recognizer(path: str):
    base = list_available_pairs("s100")[0]
    return resolve_pair(
        "s100",
        det_asset_id=base.detector_asset,
        rec_asset_id=base.recognizer_asset,
        det_model_path=str(base.detector_model_path),
        rec_model_path=path,
    )


def _flat_inputs(value: Dict[str, Dict[str, np.ndarray]], model_name: str):
    if model_name not in value or not isinstance(value[model_name], dict):
        raise ValueError(f"Expected nested tensors for model {model_name!r}.")
    return value[model_name]


def _nested_outputs(value: Dict[str, np.ndarray], model_name: str):
    return {model_name: value}


def _draw_boxes(image: np.ndarray, boxes: List[np.ndarray]) -> np.ndarray:
    """Draw the historical green rectangle overlay without utility imports."""

    import cv2

    result = image.copy()
    for box in boxes:
        points = np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return result


class PaddleOCRDet:
    """Legacy S100 detector wrapper backed by the canonical stage runner."""

    def __init__(self, config: PaddleOCRDetConfig) -> None:
        self.cfg = config
        self.pair = _pair_for_detector(config.model_path)
        self._det_runner, self._rec_runner = create_stage_runners(self.pair)
        binding = self._det_runner.load()
        self.model = self._det_runner.runtime
        self.model_name = binding.model_name
        self.input_names = list(binding.input_names)
        self.output_names = list(binding.contract.output_name for _ in [0])
        self.input_shapes = dict(binding.runtime_input_shapes)
        self.output_quants = getattr(self.model, "output_quants", {})
        self.input_H = self.input_shapes[self.input_names[0]][1]
        self.input_W = self.input_shapes[self.input_names[0]][2]
        self.threshold = config.threshold
        self.ratio_prime = config.ratio_prime
        self._pipeline = OCRPipeline(
            self.pair,
            self._det_runner,
            self._rec_runner,
            threshold=config.threshold,
            ratio_prime=config.ratio_prime,
        )

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        self._det_runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        return {self.model_name: self._pipeline.prepare_detection(img)}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        flat = self._det_runner(_flat_inputs(input_tensor, self.model_name))
        return _nested_outputs(flat, self.model_name)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        img: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        # ``img_w``/``img_h`` remain accepted for source compatibility.  The
        # canonical postprocessor uses the actual image shape, as the old
        # caller always supplied these values from ``img.shape``.
        result = self._pipeline.postprocess_detection(
            outputs[self.model_name], img
        )
        return _draw_boxes(img, list(result.boxes)), list(result.crops), list(result.boxes)

    def predict(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        img_h, img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, img, img_w, img_h)

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        return self.predict(img)


class PaddleOCRRec:
    """Legacy S100 recognizer wrapper backed by the canonical stage runner."""

    def __init__(self, config: PaddleOCRRecConfig) -> None:
        self.cfg = config
        self.pair = _pair_for_recognizer(config.model_path)
        self._det_runner, self._rec_runner = create_stage_runners(self.pair)
        binding = self._rec_runner.load()
        self.model = self._rec_runner.runtime
        self.model_name = binding.model_name
        self.input_names = list(binding.input_names)
        self.output_names = list(binding.contract.output_name for _ in [0])
        self.input_shapes = dict(binding.runtime_input_shapes)
        self.output_quants = getattr(self.model, "output_quants", {})
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        output_shape = binding.output_shape
        self.seq_len = output_shape[1]
        self.num_classes = output_shape[2]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        self._rec_runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        from samples.vision.paddle_ocr.runtime.python.tensor_io import (
            prepare_recognition,
        )

        return {self.model_name: prepare_recognition(img, self.pair)}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        flat = self._rec_runner(_flat_inputs(input_tensor, self.model_name))
        return _nested_outputs(flat, self.model_name)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        char_list: List[str],
    ) -> str:
        logits = outputs[self.model_name][self.output_names[0]]
        return ctc_greedy_decode(logits, char_list)

    def predict(self, img: np.ndarray, char_list: List[str]) -> str:
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, char_list)

    def __call__(self, img: np.ndarray, char_list: List[str]) -> str:
        return self.predict(img, char_list)


__all__ = [
    "PaddleOCRDet",
    "PaddleOCRDetConfig",
    "PaddleOCRRec",
    "PaddleOCRRecConfig",
    "ctc_greedy_decode",
    "dilate_contours",
]
