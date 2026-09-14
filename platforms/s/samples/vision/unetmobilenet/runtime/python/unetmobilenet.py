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

# flake8: noqa: E501
# flake8: noqa: E402

"""Provide a UnetMobileNet inference wrapper and pipeline utilities.

This module defines a lightweight UnetMobileNet runtime wrapper built on HBM runtime.
It includes configuration definitions and a complete semantic segmentation pipeline
(preprocess, forward, postprocess), producing a colorized blended result image.

Key Features:
    - UnetMobileNetConfig dataclass for configuring model parameters.
    - UnetMobileNet class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Direct-resize preprocessing (no letterbox) using INTER_AREA interpolation.
    - Per-pixel argmax postprocessing with color palette overlay.

Typical Usage:
    >>> from unetmobilenet import UnetMobileNet, UnetMobileNetConfig
    >>> cfg = UnetMobileNetConfig(model_path="/path/to/unet_mobilenet.hbm")
    >>> model = UnetMobileNet(cfg)
    >>> result_img = model(img)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - The model is trained on the Cityscapes dataset with 19 semantic classes.
"""

import os
import cv2
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess.py
#   utils/py_utils/postprocess.py
#   utils/py_utils/visualize.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils
import utils.py_utils.visualize as visualize


@dataclass
class UnetMobileNetConfig:
    """Configuration for initializing the UnetMobileNet model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the UnetMobileNet pipeline.

    Attributes:
        model_path: Path to the compiled UnetMobileNet `.hbm` model.
        classes_num: Number of semantic segmentation classes. Defaults to 19 (Cityscapes).
        resize_type: Image resize mode used during preprocessing.
            - 0: Direct resize (stretch to fit model input).
        alpha_f: Alpha blending factor for visualization.
            - 0.0: Show only the segmentation mask.
            - 1.0: Show only the original image.
    """
    model_path: str
    classes_num: int = 19
    resize_type: int = 0
    alpha_f: float = 0.75


class UnetMobileNet:
    """UnetMobileNet semantic segmentation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for UnetMobileNet, including
    input preprocessing, model execution, and postprocessing steps such as per-pixel
    argmax, color palette mapping, and alpha-blended visualization.

    Attributes:
        model: Loaded HBM runtime model instance.
        model_name: Name of the first loaded model.
        input_names: Input tensor name list.
        output_names: Output tensor name list.
        input_shapes: Input tensor shape dictionary.
        output_quants: Output quantization parameter dictionary.
        input_h: Model input height (pixels).
        input_w: Model input width (pixels).
        cfg: Model configuration object.

    Notes:
        The model output is an NHWC int32 tensor of raw logits over 19 Cityscapes classes.
        Argmax is applied directly on the quantized int32 values since relative order is
        preserved under quantization.
    """

    def __init__(self, config: UnetMobileNetConfig):
        """Initialize the UnetMobileNet model with the given configuration.

        Args:
            config: Configuration object containing model path and inference parameters.
                All field semantics are defined in `UnetMobileNetConfig`.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # Build BGR color palette from rdk_colors for fast numpy indexing
        self._palette = np.array(visualize.rdk_colors, dtype=np.uint8).reshape(-1, 3)

        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.

        Returns:
            None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                    img: np.ndarray,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is resized using direct resize (INTER_AREA) to the model
        input dimensions and converted from BGR to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            image_format: Input image format. Currently only `"BGR"` is supported.

        Returns:
            A nested input tensor dictionary: `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(
                img, self.input_w, self.input_h,
                self.cfg.resize_type, interpolation=cv2.INTER_AREA)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     ) -> np.ndarray:
        """Convert raw model outputs into a per-pixel semantic class ID mask.

        Steps:
        1) Argmax over the channel axis to obtain per-pixel class IDs.
        2) Resize the class ID map back to the original image resolution.

        Args:
            outputs: Raw output tensors from inference (as returned by `forward()`).
            ori_img_w: Width of the original input image (pixels).
            ori_img_h: Height of the original input image (pixels).

        Returns:
            np.ndarray: Per-pixel class ID mask with shape `(H, W)` and dtype
                `np.int32`, where each value is the predicted Cityscapes class
                index in the range `[0, classes_num - 1]`.
        """
        # Get raw int32 logits (argmax is order-preserving under quantization)
        logits = outputs[self.model_name][self.output_names[0]][0]  # (H, W, C)

        # Step 1: Argmax → per-pixel class IDs
        pred_class = np.argmax(logits, axis=-1).astype(np.int32)  # (H, W)

        # Step 2: Resize to original image size
        seg_mask = post_utils.recover_to_original_size(
            pred_class, ori_img_w, ori_img_h, self.cfg.resize_type)

        return seg_mask.astype(np.int32)

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                alpha_f: Optional[float] = None,
                ) -> np.ndarray:
        """Run the complete segmentation pipeline on a single image.

        This method internally performs preprocessing, inference, postprocessing,
        and visualization (colorize + alpha blend).

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently supports `"BGR"`.
            alpha_f: Alpha blending factor override.

        Returns:
            np.ndarray: Blended BGR result image of shape `(H, W, 3)` with the
                segmentation mask overlaid on the original image.
        """
        alpha_f = alpha_f if alpha_f is not None else self.cfg.alpha_f
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) Preprocess
        input_tensor = self.pre_process(img, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess: per-pixel class IDs at original image resolution
        seg_mask = self.post_process(outputs, ori_img_w, ori_img_h)

        # 4) Colorize class IDs to BGR and alpha-blend with original image
        seg_masked = np.where(seg_mask < self.cfg.classes_num, seg_mask, 0)
        parsing_img = self._palette[seg_masked]  # (H, W, 3) BGR
        blended_img = cv2.addWeighted(img, alpha_f, parsing_img, 1.0 - alpha_f, 0.0)

        return blended_img

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 alpha_f: Optional[float] = None,
                 ) -> np.ndarray:
        """Callable interface for the segmentation pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format.
            alpha_f: Alpha blending factor override.

        Returns:
            Same return value as `predict()`.
        """
        return self.predict(img, image_format, alpha_f)
