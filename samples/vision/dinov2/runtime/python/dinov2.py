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

"""DINOv2 vision encoder inference wrapper based on HB_HBMRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import hbm_runtime
import numpy as np

CLS_FEAT = "cls_feat"
PATCH_FEAT = "patch_feat"
INPUT_NAME = "input"

# ImageNet channel statistics used by the DINOv2 pretraining pipeline.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Dinov2Config:
    """Configuration for a DINOv2 ViT-S/14 HBM model.

    Args:
        model_path: Path to the quantized `.hbm` model.
        image_size: Square input resolution expected by the model.
        output: Model output to return: `cls_feat` or `patch_feat`.
    """

    model_path: str
    image_size: int = 224
    output: str = CLS_FEAT


class Dinov2:
    """DINOv2 vision encoder wrapper based on HB_HBMRuntime.

    The model consumes one preprocessed float32 NCHW RGB tensor and produces
    two outputs: a global image embedding ``cls_feat`` of shape ``(1, 384)``
    and per-patch features ``patch_feat`` of shape ``(1, 256, 384)``.
    """

    def __init__(self, config: Dinov2Config):
        """Initialize the DINOv2 HBM runtime.

        Args:
            config: Runtime configuration with model path, input size, and
                selected output.

        Raises:
            ValueError: If the requested output name is not supported.
        """

        if config.output not in (CLS_FEAT, PATCH_FEAT):
            raise ValueError(f"Unsupported output: {config.output}")

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set optional runtime scheduling parameters.

        Args:
            priority: Runtime priority in the range 0 to 255.
            bpu_cores: BPU core indexes used by hbm_runtime.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = priority
        if bpu_cores is not None:
            kwargs["bpu_cores"] = bpu_cores
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray, image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Convert one image to the DINOv2 float32 NCHW RGB input.

        The preprocessing matches the DINOv2 evaluation pipeline: square
        resize, BGR to RGB, scale to ``[0, 1]``, then ImageNet mean/std
        normalization.

        Args:
            img: Input image as a NumPy array.
            image_format: Input image format. Only `BGR` is supported.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.

        Raises:
            ValueError: If the input image format is not supported.
        """

        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.cfg.image_size, self.cfg.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
        tensor = tensor / 255.0
        tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        return {self._model_name(): {INPUT_NAME: np.ascontiguousarray(tensor)}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run DINOv2 inference with hbm_runtime.

        Args:
            input_tensor: Nested input dictionary produced by `pre_process`.

        Returns:
            Raw nested output dictionary returned by `hbm_runtime.run()`.
        """

        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Extract the requested feature tensor from hbm_runtime output.

        Args:
            outputs: Raw nested output dictionary returned by `forward`.

        Returns:
            np.ndarray: The selected feature tensor.
                - For ``cls_feat``: shape ``(1, 384)`` global image embedding.
                - For ``patch_feat``: shape ``(1, 256, 384)`` per-patch features.

        Raises:
            ValueError: If the output tensor contains NaN or Inf values.
        """

        tensor = outputs[self._model_name()][self.cfg.output]
        if not np.isfinite(tensor).all():
            raise ValueError("DINOv2 output contains NaN or Inf values.")
        return tensor

    def predict(self, img: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Run preprocessing, inference, and postprocessing for one image.

        Args:
            img: Input image as a NumPy array.
            image_format: Input image color format. Only ``BGR`` is supported.

        Returns:
            np.ndarray: The selected feature tensor from `post_process`.
        """

        inputs = self.pre_process(img, image_format)
        outputs = self.forward(inputs)
        return self.post_process(outputs)

    def __call__(self, img: np.ndarray, image_format: str = "BGR") -> np.ndarray:
        """Callable interface for `predict()`."""

        return self.predict(img, image_format)

    def _model_name(self) -> str:
        """Return the single submodel name of the loaded HBM model."""

        names = self.model.model_names
        return names[0] if isinstance(names, list) and names else ""
