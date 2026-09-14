# Copyright (c) 2025 D-Robotics Corporation
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

"""
Ultralytics YOLO Classification Runtime Wrapper.

This module implements the maintained Python classification wrapper for the
`ultralytics_yolo` sample on `RDK X5`. The wrapper is shared by the two
classification families delivered in this sample:

    - YOLOv8
    - YOLO11

The exported X5 models use one fixed output protocol:

    - output[0]: classification logits with shape `(1, 1000, 1, 1)`

The wrapper keeps classification behavior consistent with the mature sample
style used by the Model Zoo:

    - load one BGR image
    - convert it to packed NV12
    - run `hbm_runtime`
    - decode Top-K results
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hbm_runtime
import numpy as np
from scipy.special import softmax

sys.path.append(os.path.abspath("../../../../../"))

import utils.py_utils.preprocess as pre_utils


logger = logging.getLogger("Ultralytics_YOLO")


@dataclass
class UltralyticsYOLOClsConfig:
    """Configuration used by the classification wrapper.

    Args:
        model_path: Path to the X5 `.bin` model.
        topk: Number of Top-K results returned by `predict()`.
        resize_type: Resize policy, `0` for direct resize and `1` for
            letterbox.
    """

    model_path: str
    topk: int = 5
    resize_type: int = 0


class UltralyticsYOLOCls:
    """Classification wrapper built on top of `hbm_runtime`."""

    def __init__(self, config: UltralyticsYOLOClsConfig):
        """Initialize the classification runtime wrapper.

        Args:
            config: Classification runtime configuration for the current model.
        """
        self.cfg = config

        t0 = time.time()
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info("\033[1;31mLoad Model time = %.2f ms\033[0m", 1000 * (time.time() - t0))

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        if input_shape[1] == 3:
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        else:
            self.input_h = input_shape[1]
            self.input_w = input_shape[2]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[list] = None,
    ) -> None:
        """Set optional BPU scheduling parameters."""
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(
        self,
        img: np.ndarray,
        resize_type: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert one BGR image into packed NV12 model input."""
        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type

        resized = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)

        logger.info("\033[1;31mPre-process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return {self.model_name: {self.input_names[0]: packed_nv12}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run BPU inference for one input image."""
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info("\033[1;31mForward time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return outputs

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> List[Tuple[int, float]]:
        """Convert classification logits into Top-K prediction results.

        Args:
            outputs: Raw runtime outputs returned by `forward`.

        Returns:
            A list of `(class_id, score)` tuples sorted from high to low.
        """
        t0 = time.time()
        raw_outputs = outputs[self.model_name]
        logits = raw_outputs[self.output_names[0]].reshape(-1).astype(np.float32)
        probs = softmax(logits)
        top_indices = np.argsort(probs)[::-1][:self.cfg.topk]
        results = [(int(idx), float(probs[idx])) for idx in top_indices]
        logger.info("\033[1;31mPost Process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return results

    def predict(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """Run the full classification pipeline on one image."""
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs)

    def __call__(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """Provide function-style inference for one image."""
        return self.predict(img)
