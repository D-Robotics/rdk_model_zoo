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

"""Provide a YOLO classification inference wrapper and pipeline utilities.

This module defines a lightweight YOLO classification runtime wrapper built
on HBM runtime. It supports YOLO classification models (v8-cls and v11-cls),
producing top-K class predictions with softmax probabilities.

Key Features:
    - YoloClsConfig dataclass for configuring model parameters.
    - YoloCls class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Top-K classification with softmax probability computation.

Typical Usage:
    >>> from yolo_cls import YoloCls, YoloClsConfig
    >>> cfg = YoloClsConfig(model_path="/path/to/yolo11n_cls.hbm")
    >>> model = YoloCls(cfg)
    >>> results = model(img)  # List of (class_id, probability)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - Classification models use stretch resize (resize_type=0) by default.
"""

import os
import sys
import hbm_runtime
import numpy as np
from scipy.special import softmax
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YoloClsConfig:
    """Configuration for initializing the YoloCls model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO classification
    pipeline. It applies to YOLO classification models (v8-cls and v11-cls).

    Attributes:
        model_path: Path to the compiled YOLO-Cls `.hbm` model.
        topk: Number of top classes to return. Defaults to 5.
        resize_type: Image resize strategy (0=stretch, 1=letterbox).
            Classification models typically use stretch resize.
    """
    model_path: str
    topk: int = 5
    resize_type: int = 0


class YoloCls:
    """YOLO classification wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLO classification
    models (v8-cls and v11-cls), including input preprocessing, model
    execution, and postprocessing with softmax and top-K selection.

    Attributes:
        model: Loaded HBM runtime model instance.
        model_name: Name of the first loaded model.
        input_names: Input tensor name list.
        output_names: Output tensor name list.
        input_shapes: Input tensor shape dictionary.
        input_h: Model input height (pixels).
        input_w: Model input width (pixels).
        cfg: Model configuration object.
    """

    def __init__(self, config: YoloClsConfig):
        """Initialize the YoloCls model with the given configuration.

        Args:
            config: Configuration object containing model path and all inference
                parameters. All field semantics are defined in `YoloClsConfig`.
        """
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # Model input resolution (H, W)
        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

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

        The input image is resized and converted from BGR to NV12
        (Y and UV planes). S100 models use the fixed dual-input Y/UV
        tensor protocol.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently only `"BGR"` is supported.

        Returns:
            A nested input tensor dictionary: `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        resized_img = pre_utils.resized_image(
            img, self.input_w, self.input_h, self.cfg.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized_img)

        if len(self.input_names) != 2:
            raise ValueError(
                f"YOLO classification expects fixed NV12 Y/UV inputs, got {len(self.input_names)} inputs.")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        return self.model.run(input_tensor)

    def post_process(self,
                     outputs: Dict,
                     topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Process raw logits to get Top-K classification results.

        The output tensor is reshaped and passed through softmax
        to obtain probabilities. The top-K classes are returned.

        Args:
            outputs: Raw output tensors from inference.
            topk: Number of top results to return. If `None`, uses config value.

        Returns:
            List of `(class_id, probability)` sorted by probability descending.
        """
        topk = topk if topk is not None else self.cfg.topk

        logits = outputs[self.model_name][self.output_names[0]].reshape(-1)

        probs = softmax(logits)
        top_indices = np.argsort(probs)[::-1][:topk]
        return [(int(idx), float(probs[idx])) for idx in top_indices]

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                topk: Optional[int] = None) -> List[Tuple[int, float]]:
        """Run the complete classification pipeline on a single image.

        This method internally performs preprocessing, inference, and
        postprocessing.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently supports `"BGR"`.
            topk: Number of top results to return.

        Returns:
            List of (class_id, probability) tuples.
        """
        inp = self.pre_process(img, image_format)
        out = self.forward(inp)
        return self.post_process(out, topk)

    def __call__(self,
                 img: np.ndarray,
                 **kwargs) -> List[Tuple[int, float]]:
        """Callable interface equivalent to predict().

        Args:
            img: Input image array.
            **kwargs: Additional keyword arguments passed to predict().

        Returns:
            List of (class_id, probability) tuples.
        """
        return self.predict(img, **kwargs)
