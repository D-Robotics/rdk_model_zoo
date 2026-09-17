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

"""Compatibility API for the legacy X5 PaddleOCR sample.

The maintained implementation lives in ``samples.vision.paddle_ocr``.  This
module keeps the historical class and tensor-dictionary spellings used by
older X5 scripts while forwarding all preparation, geometry, runtime binding,
and CTC work to that implementation.  It intentionally does not import
``hbm_runtime`` at module import time, so host tools can inspect the module.
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
    ctc_decode_indices,
)
from samples.vision.paddle_ocr.runtime.python.geometry import (  # noqa: E402
    crop_and_rotate_image,
    dilate_contours,
    get_bounding_boxes,
)
from samples.vision.paddle_ocr.runtime.python.model_binding import (  # noqa: E402
    X5_ALPHABET,
    list_available_pairs,
    resolve_pair,
)
from samples.vision.paddle_ocr.runtime.python.model_runner import (  # noqa: E402
    create_stage_runners,
)
from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline  # noqa: E402


ALPHABET = X5_ALPHABET


@dataclass
class PaddleOCRConfig:
    """Configuration retained for the original X5 ``PaddleOCR`` API."""

    det_model_path: str = ""
    rec_model_path: str = ""
    det_threshold: float = 0.5
    det_ratio_prime: float = 2.7
    det_min_area: int = 100
    rec_input_size: Tuple[int, int] = (48, 320)
    rec_output_size: Tuple[int, int] = (40, 97)


class _CTCLabelConverter:
    """Legacy index decoder backed by the canonical CTC implementation."""

    def __init__(self, alphabet: str, ignore_case: bool = False) -> None:
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + "-"
        self.dict = {char: i + 1 for i, char in enumerate(alphabet)}

    def decode(self, t: np.ndarray, length: np.ndarray, raw: bool = False):
        values = np.asarray(t)
        lengths = np.asarray(length)
        if lengths.ndim != 1:
            raise ValueError("CTC lengths must be a one-dimensional array.")
        if values.ndim != 1:
            values = values.reshape(-1)
        if lengths.size == 1:
            size = int(lengths[0])
            if size < 0 or size > values.size:
                raise ValueError("CTC length is outside the index sequence.")
            indices = values[:size]
            # The old API exposed '-' for blank in raw mode.  Keep that
            # presentation while sharing validation/collapse with canonical.
            return ctc_decode_indices(
                indices,
                ("blank",) + tuple(self.alphabet[:-1]),
                raw=raw,
                blank_symbol="-",
            )
        result = []
        offset = 0
        for item in lengths:
            size = int(item)
            if size < 0 or offset + size > values.size:
                raise ValueError("CTC length is outside the index sequence.")
            result.append(
                self.decode(values[offset : offset + size], np.array([size]), raw=raw)
            )
            offset += size
        if offset != values.size:
            raise ValueError("CTC lengths do not cover the index sequence.")
        return result


def _pair_for_config(config: PaddleOCRConfig):
    if not config.det_model_path and not config.rec_model_path:
        return resolve_pair("x5")
    if not config.det_model_path or not config.rec_model_path:
        raise ValueError("det_model_path and rec_model_path must be supplied together.")
    pair = list_available_pairs("x5")[0]
    return resolve_pair(
        "x5",
        det_asset_id=pair.detector_asset,
        rec_asset_id=pair.recognizer_asset,
        det_model_path=config.det_model_path,
        rec_model_path=config.rec_model_path,
    )


def _flat_inputs(value: Dict[str, Dict[str, np.ndarray]], model_name: str):
    if model_name not in value or not isinstance(value[model_name], dict):
        raise ValueError(f"Expected nested tensors for model {model_name!r}.")
    return value[model_name]


def _nested_outputs(value: Dict[str, np.ndarray], model_name: str):
    return {model_name: value}


class PaddleOCR:
    """Legacy X5 two-stage wrapper delegating to the canonical pipeline."""

    def __init__(self, config: PaddleOCRConfig) -> None:
        self.cfg = config
        self.pair = _pair_for_config(config)
        self._det_runner, self._rec_runner = create_stage_runners(self.pair)
        det_binding = self._det_runner.load()
        rec_binding = self._rec_runner.load()

        # Preserve the model/metadata attributes consumed by historical code.
        self.det_model = self._det_runner.runtime
        self.rec_model = self._rec_runner.runtime
        self.det_model_name = det_binding.model_name
        self.rec_model_name = rec_binding.model_name
        self.det_input_name = det_binding.input_names[0]
        self.rec_input_name = rec_binding.input_names[0]
        self.det_input_shape = dict(det_binding.runtime_input_shapes)
        self.rec_input_shape = dict(rec_binding.runtime_input_shapes)
        self.det_output_name = det_binding.output_name
        self.rec_output_name = rec_binding.output_name
        self.det_input_h = 640
        self.det_input_w = 640
        self.converter = _CTCLabelConverter(ALPHABET)
        self._pipeline = OCRPipeline(
            self.pair,
            self._det_runner,
            self._rec_runner,
            threshold=config.det_threshold,
            ratio_prime=config.det_ratio_prime,
            min_area=config.det_min_area,
        )

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        self._det_runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
        self._rec_runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def pre_process(self, image: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        flat = self._pipeline.prepare_detection(image)
        return {self.det_model_name: flat}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        flat = self._det_runner(_flat_inputs(input_tensor, self.det_model_name))
        return _nested_outputs(flat, self.det_model_name)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        image: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if image is None:
            image = np.zeros((self.det_input_h, self.det_input_w, 3), dtype=np.uint8)
        result = self._pipeline.postprocess_detection(
            outputs[self.det_model_name], image
        )
        return list(result.polygons), list(result.boxes)

    def _dilate_contours(self, contours: list) -> List[np.ndarray]:
        return list(
            dilate_contours(
                contours,
                target="x5",
                ratio_prime=self.cfg.det_ratio_prime,
            )
        )

    def _get_bounding_boxes(
        self, dilated_polys: List[np.ndarray], min_area: int
    ) -> List[np.ndarray]:
        return list(get_bounding_boxes(dilated_polys, min_area=min_area))

    def _rec_pre_process(self, cropped_img: np.ndarray) -> np.ndarray:
        return self._pipeline.prepare_recognition(cropped_img)[self.rec_input_name]

    def _rec_forward(
        self, input_tensor: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        outputs = self._rec_runner({self.rec_input_name: input_tensor})
        return _nested_outputs(outputs, self.rec_model_name)

    def _rec_post_process(
        self, outputs: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[str, str]:
        logits = np.asarray(outputs[self.rec_model_name][self.rec_output_name])
        scores = logits.reshape(1, 40, 97)
        indices = np.argmax(scores, axis=2).reshape(-1)
        lengths = np.array([indices.size], dtype=np.int32)
        raw = self.converter.decode(indices, lengths, raw=True)
        simplified = self.converter.decode(indices, lengths, raw=False)
        return raw, simplified

    @staticmethod
    def _crop_and_rotate(img: np.ndarray, box: np.ndarray) -> np.ndarray:
        return crop_and_rotate_image(img, box, target="x5")

    def predict(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        detection = self._pipeline.run_detection(image)
        texts = [self._pipeline.run_recognition(crop) for crop in detection.crops]
        return list(detection.boxes), texts

    def __call__(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        return self.predict(image)


__all__ = ["ALPHABET", "PaddleOCR", "PaddleOCRConfig", "_CTCLabelConverter"]
