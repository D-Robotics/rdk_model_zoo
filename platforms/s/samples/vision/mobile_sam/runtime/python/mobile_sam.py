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

"""MobileSAM dual-HBM runtime using hbm_runtime.

The image encoder and box-prompt mask decoder are both compiled for RDK-S as
`.hbm` models. This module follows the Model Zoo runtime pattern with config,
model initialization, preprocessing, inference, postprocessing, and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from hbm_runtime import HB_HBMRuntime


@dataclass
class MobileSAMConfig:
    """Runtime configuration for MobileSAM full-mask inference.

    Args:
        encoder_model_path: Path to the compiled TinyViT image encoder `.hbm`.
        decoder_model_path: Path to the compiled box-prompt decoder `.hbm`.
        input_size: Square input size used by the exported ONNX and `.hbm` models.
        box: Box prompt in `x1,y1,x2,y2` coordinates on the resized image.
    """

    encoder_model_path: str
    decoder_model_path: str
    input_size: int = 512
    box: Tuple[float, float, float, float] = (185.0, 120.0, 380.0, 445.0)


class MobileSAMSegment:
    """MobileSAM full-mask pipeline backed by two hbm_runtime models.

    The class owns the encoder and decoder runtimes and exposes the standard
    `pre_process`, `forward`, `post_process`, and `predict` stages used by RDK
    Model Zoo Python demos.
    """

    def __init__(self, config: MobileSAMConfig) -> None:
        """Load encoder and decoder `.hbm` models.

        Args:
            config: Runtime configuration with model paths and prompt settings.
        """

        self.config = config
        self.encoder = HB_HBMRuntime(config.encoder_model_path)
        self.decoder = HB_HBMRuntime(config.decoder_model_path)
        self.encoder_name = self.encoder.model_names[0]
        self.decoder_name = self.decoder.model_names[0]

    def set_scheduling_params(
        self,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
    ) -> None:
        """Set S-series scheduling parameters for both packed models."""

        for model, model_name in (
            (self.encoder, self.encoder_name),
            (self.decoder, self.decoder_name),
        ):
            kwargs: Dict[str, Dict[str, int | list[int]]] = {}
            if priority is not None:
                kwargs["priority"] = {model_name: priority}
            if bpu_cores is not None:
                kwargs["bpu_cores"] = {model_name: bpu_cores}
            if kwargs:
                model.set_scheduling_params(**kwargs)

    def pre_process(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare the normalized encoder input tensor.

        Args:
            image: Input image in BGR HWC format.

        Returns:
            Dictionary containing `normalized_images` in NCHW float32 format.

        Raises:
            ValueError: If the input is not a three-channel image.
        """

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR HWC image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_LINEAR)
        chw = rgb.transpose(2, 0, 1).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        tensor = ((chw - mean) / std)[None]
        return {"normalized_images": np.ascontiguousarray(tensor, dtype=np.float32)}

    def forward(self, input_tensors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run encoder and decoder inference.

        Args:
            input_tensors: Encoder input dictionary from `pre_process`.

        Returns:
            Nested dictionaries containing raw encoder and decoder outputs.
        """

        encoder_outputs = self.encoder.run({self.encoder_name: input_tensors})
        image_embeddings = encoder_outputs[self.encoder_name]["image_embeddings"].astype(np.float32)
        x1, y1, x2, y2 = self.config.box
        # The exported decoder's ONNX contract is ``boxes: [1, 4]``.
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        decoder_outputs = self.decoder.run({
            self.decoder_name: {
                "image_embeddings": image_embeddings,
                "boxes": boxes,
            }
        })
        return {"encoder": encoder_outputs[self.encoder_name], "decoder": decoder_outputs[self.decoder_name]}

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray | float | int]:
        """Convert decoder logits to a full-resolution binary mask.

        Args:
            outputs: Raw outputs returned by `forward`.

        Returns:
            Dictionary with binary mask, selected IoU, selected mask index, and
            raw low-resolution masks.
        """

        low_res_masks = outputs["decoder"]["low_res_masks"].astype(np.float32)
        ious = outputs["decoder"]["iou_predictions"].astype(np.float32).reshape(-1)
        best_index = int(np.argmax(ious))
        low_res = low_res_masks[0, best_index]
        mask_logits = cv2.resize(low_res, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_LINEAR)
        mask = mask_logits > 0.0
        return {"mask": mask, "iou": float(ious[best_index]), "mask_index": best_index, "low_res_masks": low_res_masks}

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        """Run the complete segmentation pipeline.

        Args:
            image: Input image in BGR HWC format.

        Returns:
            Postprocessed segmentation result dictionary.
        """

        return self.post_process(self.forward(self.pre_process(image)))

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        """Alias for `predict`."""

        return self.predict(image)
