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

"""PointNet part segmentation runtime wrapper for RDK S100.

This module provides a BPU inference wrapper for the PointNet model, which
performs part-level segmentation on 3D point clouds (e.g., ShapeNet chair category).

Key Features:
    - Accepts a normalized point cloud of shape (1, 3, N) as input.
    - Returns a per-point part label array of shape (N,) with dtype int32.
    - Integrates with hbm_runtime for BPU-accelerated inference on RDK S100.

Typical Usage:
    >>> import numpy as np
    >>> from pointnet import PointNetConfig, PointNet
    >>> config = PointNetConfig(model_path="../../model/s100/pointnet.hbm")
    >>> model = PointNet(config)
    >>> pts = np.load("../../test_data/chair.pts")  # shape (N, 3)
    >>> labels = model.predict(pts)  # np.ndarray (N,) int32, per-point part index

Notes:
    - Input point cloud is normalized by subtracting centroid and dividing by max radius.
    - Output is a per-point class index (int32), not a bounding box or probability vector.
    - Run from runtime/python/ so that sys.path resolves utils/ correctly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import hbm_runtime
import numpy as np


@dataclass
class PointNetConfig:
    """Configuration for the PointNet segmentation model.

    Attributes:
        model_path: Path to the compiled HBM model.
        num_parts: Number of chair part labels emitted by the model.
    """

    model_path: str
    num_parts: int = 4


class PointNet:
    """PointNet point cloud part segmentation wrapper.

    The compiled model has one input tensor with shape ``(1, 3, N)`` and one
    output tensor with per-point part logits shaped ``(1, N, 4)``.
    """

    def __init__(self, config: PointNetConfig):
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

    @staticmethod
    def load_point_cloud(pts_path: str) -> np.ndarray:
        """Load and normalize point cloud coordinates from a `.pts` file."""
        point_set = np.loadtxt(pts_path).astype(np.float32)
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        if dist == 0:
            raise ValueError("Point cloud normalization failed: zero radius.")
        return point_set / dist

    def pre_process(self, point_cloud: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Transpose the normalized point cloud into model input layout."""
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape (N, 3), got {point_cloud.shape}")
        point = np.expand_dims(point_cloud.transpose(1, 0), axis=0).astype(np.float32)
        return {self.model_name: {self.input_name: point}}

    def forward(
        self,
        input_tensor: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run model inference with hbm_runtime."""
        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Convert per-point logits into part label IDs."""
        pred = outputs[self.model_name][self.output_name]
        if pred.ndim != 3 or pred.shape[2] != self.cfg.num_parts:
            raise ValueError(f"Expected output shape (1, N, {self.cfg.num_parts}), got {pred.shape}")
        return np.argmax(pred[0], axis=1).astype(np.int32)

    def predict(self, point_cloud: np.ndarray) -> np.ndarray:
        """Run preprocess, inference, and postprocess for one point cloud."""
        input_tensor = self.pre_process(point_cloud)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs)

    def __call__(self, point_cloud: np.ndarray) -> np.ndarray:
        """Alias for ``predict``."""
        return self.predict(point_cloud)
