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

"""SigLIP vision encoder runtime wrapper for RDK S100/S100P.

The packaged SigLIP HBM file contains two fixed submodels:

- ``pooler_output`` for image embedding vectors.
- ``last_hidden_state`` for per-patch visual embedding tensors.

Both submodels use a single float32 NCHW RGB input tensor named ``_input_0``
with value range ``[-1, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import hbm_runtime
import numpy as np


POOLER_OUTPUT = "pooler_output"
LAST_HIDDEN_STATE = "last_hidden_state"
INPUT_NAME = "_input_0"
OUTPUT_NAME = "_output_0"


@dataclass
class SigLIPConfig:
    """Configuration for a SigLIP vision encoder HBM model.

    Args:
        model_path: Path to the packed SigLIP `.hbm` model.
        image_size: Square input resolution expected by the selected model.
        submodel: Submodel to run: `pooler_output` or `last_hidden_state`.
    """

    model_path: str
    image_size: int = 224
    submodel: str = POOLER_OUTPUT


class SigLIP:
    """SigLIP vision encoder wrapper based on HB_HBMRuntime."""

    def __init__(self, config: SigLIPConfig):
        """Initialize the SigLIP HBM runtime.

        Args:
            config: Runtime configuration with model path, input size, and
                selected submodel.
        """

        if config.submodel not in (POOLER_OUTPUT, LAST_HIDDEN_STATE):
            raise ValueError(f"Unsupported submodel: {config.submodel}")

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set optional runtime scheduling parameters for both submodels."""

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {
                POOLER_OUTPUT: priority,
                LAST_HIDDEN_STATE: priority,
            }
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {
                POOLER_OUTPUT: bpu_cores,
                LAST_HIDDEN_STATE: bpu_cores,
            }
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray, image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Convert one image to SigLIP float32 NCHW RGB input.

        The preprocessing follows the source sample: keep aspect ratio, pad
        with RGB value `(127, 127, 127)`, then normalize by `/127.5 - 1.0`.

        Args:
            img: Input image as a NumPy array.
            image_format: Input image format. Only `BGR` is supported.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.
        """

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = self.cfg.image_size / max(h, w)
        new_h = max(int(h * scale), 1)
        new_w = max(int(w * scale), 1)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = self.cfg.image_size - new_h
        pad_w = self.cfg.image_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(127, 127, 127),
        )
        tensor = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
        tensor = tensor / 127.5 - 1.0
        return {self.cfg.submodel: {INPUT_NAME: tensor}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run SigLIP inference with hbm_runtime."""

        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Extract the embedding tensor from hbm_runtime output.

        Args:
            outputs: Raw nested output dictionary returned by hbm_runtime.

        Returns:
            np.ndarray: The output embedding tensor.
                - For ``pooler_output``: shape ``(1, D)`` global image embedding.
                - For ``last_hidden_state``: shape ``(1, N, D)`` per-patch features.

        Raises:
            ValueError: If the output tensor contains NaN or Inf values.
        """

        tensor = outputs[self.cfg.submodel][OUTPUT_NAME]
        if not np.isfinite(tensor).all():
            raise ValueError("SigLIP output contains NaN or Inf values.")
        return tensor

    def predict(self, img: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            img: Input image as a NumPy array.
            image_format: Input image color format. Only ``BGR`` is supported.

        Returns:
            np.ndarray: The output embedding tensor from ``post_process``.
        """

        inputs = self.pre_process(img, image_format)
        outputs = self.forward(inputs)
        return self.post_process(outputs)

    def __call__(self, img: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Callable interface for ``predict()``."""

        return self.predict(img, image_format)
