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

"""3D ResNet-18 video action classification runtime wrapper.

This module provides a BPU inference wrapper for the R3D-18 model, which
classifies short video clips into one of 400 Kinetics-400 action categories.

Key Features:
    - Accepts preprocessed float32 video clips of shape (1, 3, 16, 112, 112).
    - Returns Top-K action predictions as ``List[Tuple[class_id, probability]]``.
    - Integrates with shared ``utils/py_utils/visualize`` for Top-K decoding.

Typical Usage:
    >>> import numpy as np
    >>> from resnet3d import ResNet3DConfig, ResNet3D
    >>> config = ResNet3DConfig(model_path="../../model/s100/r3d_18.hbm")
    >>> model = ResNet3D(config)
    >>> clip = np.load("../../test_data/video0.npy")  # shape (1, 3, 16, 112, 112)
    >>> predictions = model.predict(clip, top_k=5)
    >>> # predictions: [(class_id, score), ...]

Notes:
    - Input must be a preprocessed clip: RGB, normalized, shape (1,3,16,112,112).
    - Run from the runtime/python/ directory so that sys.path resolves correctly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hbm_runtime
import numpy as np

import os
import sys

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.visualize as visualize


@dataclass
class ResNet3DConfig:
    """Configuration for the 3D ResNet-18 action classification model.

    Attributes:
        model_path: Path to the compiled HBM model.
    """

    model_path: str


class ResNet3D:
    """3D ResNet-18 video action classification wrapper.

    The compiled R3D-18 model has one float32 input tensor with shape
    ``(1, 3, 16, 112, 112)`` and one logits output tensor for Kinetics-400.
    """

    def __init__(self, config: ResNet3DConfig):
        """Load the HBM model and cache its fixed tensor names."""
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_name = self.model.input_names[self.model_name][0]
        self.output_name = self.model.output_names[self.model_name][0]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set hbm_runtime scheduling parameters for the model."""
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, clip: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare the 5D video clip tensor for model inference."""
        if clip.shape != (1, 3, 16, 112, 112):
            raise ValueError(
                f"Expected input shape (1, 3, 16, 112, 112), got {clip.shape}"
            )
        return {self.model_name: {self.input_name: clip.astype(np.float32)}}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run model inference with hbm_runtime."""
        return self.model.run(input_tensor)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Convert raw logits into Top-K class predictions."""
        logits = outputs[self.model_name][self.output_name]
        return visualize.get_topk_predictions(logits, topk=top_k)

    def predict(self, clip: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Run preprocess, inference, and postprocess for one video clip."""
        input_tensor = self.pre_process(clip)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, top_k=top_k)

    def __call__(self, clip: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Alias for ``predict``."""
        return self.predict(clip, top_k=top_k)
