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

"""Provide YOLO26 OBB inference wrapper and pipeline utilities.

This module defines a YOLO26 oriented bounding box detection runtime
wrapper built on HBM runtime. It handles rotated box decoding, angle
calculation, and rotated NMS.

Model output layout (9 tensors):
    Stride 8:  [0] Cls(15), [1] Box(4), [2] Angle(1)
    Stride 16: [3] Cls(15), [4] Box(4), [5] Angle(1)
    Stride 32: [6] Cls(15), [7] Box(4), [8] Angle(1)
"""

import os
import sys
import math
import cv2
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YOLO26OBBConfig:
    """Configuration for the YOLO26 OBB model.

    Attributes:
        model_path: Path to the compiled `.hbm` model.
        score_thres: Confidence threshold for filtering.
        nms_thres: IoU threshold for rotated NMS.
        angle_sign: Multiplier for angle decoding.
        angle_offset: Offset in degrees to add to decoded angle.
        regularize: Whether to regularize boxes (w > h).
        resize_type: Image resize strategy (0=stretch, 1=letterbox).
        strides: Feature map strides.
    """
    model_path: str
    score_thres: float = 0.25
    nms_thres: float = 0.2
    angle_sign: float = 1.0
    angle_offset: float = 0.0
    regularize: bool = True
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class YOLO26OBB:
    """YOLO26 oriented bounding box wrapper based on HB_HBMRuntime."""

    def __init__(self, config: YOLO26OBBConfig):
        """Initialize the YOLO26 OBB model.

        Args:
            config: Configuration object.
        """
        self.cfg = config

        safe_thres = np.clip(self.cfg.score_thres, 1e-6, 1.0 - 1e-6)
        self.conf_raw = -np.log(1.0 / safe_thres - 1.0)
        self.angle_offset_rad = self.cfg.angle_offset * math.pi / 180.0

        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        input_shape = self.input_shapes[self.input_names[0]]
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]

        self.grids = {}
        for s in self.cfg.strides:
            grid_h, grid_w = self.input_h // s, self.input_w // s
            grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5

        self.map_idx = {8: (0, 1, 2), 16: (3, 4, 5), 32: (6, 7, 8)}

    def set_scheduling_params(self, priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters."""
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    image_format: str = "BGR") -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into NV12 Y/UV tensor format.

        Args:
            img: Input image array.
            image_format: Input image format. Only "BGR" is supported.

        Returns:
            Nested input tensor dictionary.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format != "BGR":
            raise ValueError(f"Unsupported image_format: {image_format}")

        resized_img = pre_utils.resized_image(img, self.input_w, self.input_h, self.cfg.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized_img)

        if len(self.input_names) != 2:
            raise ValueError(
                f"YOLO26 OBB expects fixed NV12 Y/UV inputs, got {len(self.input_names)} inputs.")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict) -> Dict:
        """Execute model inference."""
        return self.model.run(input_tensor)

    def post_process(self, outputs: Dict,
                     ori_w: int, ori_h: int) -> List[Dict]:
        """Decode and post-process OBB results.

        Args:
            outputs: Raw output tensors from inference.
            ori_w: Original image width.
            ori_h: Original image height.

        Returns:
            List of dicts with keys 'rrect', 'score', 'id'.
        """
        raw_outputs = outputs[self.model_name]

        all_rrects = []
        all_scores = []
        all_cids = []

        for stride in self.cfg.strides:
            if stride not in self.map_idx:
                continue

            ci, bi, ai = self.map_idx[stride]
            if max(bi, ci, ai) >= len(self.output_names):
                raise ValueError(
                    f"YOLO26 OBB expects 9 fixed outputs, got {len(self.output_names)} outputs.")

            box_feat = raw_outputs[self.output_names[bi]].reshape(-1, 4)
            cls_feat = raw_outputs[self.output_names[ci]].reshape(-1, raw_outputs[self.output_names[ci]].shape[-1])
            angle_feat = raw_outputs[self.output_names[ai]].reshape(-1, 1)

            max_scores = np.max(cls_feat, axis=1)
            mask = max_scores >= self.conf_raw
            if not np.any(mask):
                continue

            v_scores = post_utils.sigmoid(max_scores[mask])
            v_ids = np.argmax(cls_feat[mask], axis=1)
            v_box = np.abs(box_feat[mask])
            v_angle = angle_feat[mask]
            grid = self.grids[stride][mask]

            a_rad = (post_utils.sigmoid(v_angle[:, 0]) - 0.5) * math.pi * self.cfg.angle_sign + self.angle_offset_rad

            l, t, r, b = v_box.T
            xf, yf = (r - l) / 2.0, (b - t) / 2.0
            c_cos, s_sin = np.cos(a_rad), np.sin(a_rad)

            cx = (grid[:, 0] + xf * c_cos - yf * s_sin) * stride
            cy = (grid[:, 1] + xf * s_sin + yf * c_cos) * stride
            w = (l + r) * stride
            h = (t + b) * stride

            for _cx, _cy, _w, _h, _a, _s, _id in zip(cx, cy, w, h, a_rad, v_scores, v_ids):
                if self.cfg.regularize and _w < _h:
                    _w, _h, _a = _h, _w, _a + math.pi / 2
                all_rrects.append((_cx, _cy, _w, _h, _a))
                all_scores.append(float(_s))
                all_cids.append(int(_id))

        final_res = []
        if all_rrects:
            try:
                box_list = [((r[0], r[1]), (r[2], r[3]), math.degrees(r[4])) for r in all_rrects]
                indices = cv2.dnn.NMSBoxesRotated(
                    box_list, all_scores,
                    self.cfg.score_thres, self.cfg.nms_thres)

                if len(indices) > 0:
                    for i in indices.flatten():
                        cx, cy, w, h, a_rad = all_rrects[i]
                        scaled = post_utils.scale_coords_back_obb(
                            np.array([[cx, cy, w, h]]),
                            ori_w, ori_h, self.input_w, self.input_h, self.cfg.resize_type
                        )[0]
                        final_res.append({
                            'rrect': (scaled[0], scaled[1], scaled[2], scaled[3], a_rad),
                            'score': all_scores[i],
                            'id': all_cids[i]
                        })
            except AttributeError:
                pass

        return final_res

    def predict(self, img: np.ndarray, image_format: str = "BGR") -> List[Dict]:
        """Run the complete OBB pipeline.

        Args:
            img: Input image array.
            image_format: Input image format.

        Returns:
            List of dicts with OBB results.
        """
        ori_h, ori_w = img.shape[:2]
        inp = self.pre_process(img, image_format)
        out = self.forward(inp)
        return self.post_process(out, ori_w, ori_h)

    def __call__(self, img: np.ndarray, **kwargs) -> List[Dict]:
        """Callable interface equivalent to predict()."""
        return self.predict(img, **kwargs)
