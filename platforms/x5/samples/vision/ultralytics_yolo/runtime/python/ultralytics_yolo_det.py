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
Ultralytics YOLO Detection Runtime Wrapper.

This module implements the maintained Python detection wrapper for the
`ultralytics_yolo` sample on `RDK X5`. It is used by all supported detection
families in this sample:

    - YOLOv5u
    - YOLOv8
    - YOLOv9
    - YOLOv10
    - YOLO11
    - YOLO12
    - YOLO13

All detection models exported for this sample use the same fixed output
protocol on X5:

    - output[0]: stride-8 classification logits
    - output[1]: stride-8 DFL box tensor
    - output[2]: stride-16 classification logits
    - output[3]: stride-16 DFL box tensor
    - output[4]: stride-32 classification logits
    - output[5]: stride-32 DFL box tensor

The wrapper keeps the sample responsibilities clear:

    - preprocessing: convert one BGR image into packed NV12 input
    - forward: invoke `hbm_runtime`
    - postprocessing: decode boxes, apply class-wise NMS, and scale the
      results back to the original image
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hbm_runtime
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))

import utils.py_utils.postprocess as post_utils
import utils.py_utils.preprocess as pre_utils


logger = logging.getLogger("Ultralytics_YOLO")


@dataclass
class UltralyticsYOLODetectConfig:
    """Configuration used by the detection wrapper.

    Args:
        model_path: Path to the X5 `.bin` model.
        classes_num: Number of categories predicted by the model.
        score_thres: Confidence threshold used before NMS.
        nms_thres: IoU threshold used by class-wise NMS.
        reg: Number of DFL bins for one box edge.
        resize_type: Resize policy, `0` for direct resize and `1` for
            letterbox.
        strides: Feature strides used by the three detection heads.
    """

    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.70
    reg: int = 16
    resize_type: int = 1
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])


class UltralyticsYOLODetect:
    """Detection wrapper built on top of `hbm_runtime`.

    This wrapper owns one BPU model instance and exposes a consistent
    `predict()` interface used by `main.py`.
    """

    def __init__(self, config: UltralyticsYOLODetectConfig):
        """Initialize the runtime wrapper and decoding metadata.

        Args:
            config: Detection runtime configuration for the current model.
        """
        self.cfg = config
        self.conf_thres_raw = -np.log(1.0 / self.cfg.score_thres - 1.0)
        self.weights_static = np.arange(self.cfg.reg, dtype=np.float32)[None, None, :]

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
        """Set optional BPU scheduling parameters.

        Args:
            priority: Runtime priority used by `hbm_runtime`.
            bpu_cores: BPU core list used by the model scheduler.
        """
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
        """Convert one BGR image into packed NV12 model input.

        Args:
            img: Input BGR image loaded by OpenCV.
            resize_type: Optional resize policy override.

        Returns:
            The nested input tensor dictionary required by `HB_HBMRuntime.run`.
        """
        t0 = time.time()
        resize_type = self.cfg.resize_type if resize_type is None else resize_type

        resized = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)

        logger.info("\033[1;31mPre-process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return {self.model_name: {self.input_names[0]: packed_nv12}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run BPU inference for one input image.

        Args:
            input_tensor: Input tensor dictionary returned by `pre_process`.

        Returns:
            Raw output tensors indexed by model name and output name.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info("\033[1;31mForward time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        return outputs

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        ori_img_w: int,
        ori_img_h: int,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode detection tensors into final sample results.

        Args:
            outputs: Raw runtime outputs returned by `forward`.
            ori_img_w: Original image width.
            ori_img_h: Original image height.
            score_thres: Optional score threshold override.
            nms_thres: Optional NMS threshold override.

        Returns:
            A tuple containing:
                - bounding boxes in `(x1, y1, x2, y2)` format
                - confidence scores
                - class ids
        """
        t0 = time.time()
        score_thres = self.cfg.score_thres if score_thres is None else score_thres
        nms_thres = self.cfg.nms_thres if nms_thres is None else nms_thres
        conf_thres_raw = -np.log(1.0 / score_thres - 1.0)
        raw_outputs = outputs[self.model_name]

        detections = []
        for level_index, stride in enumerate(self.cfg.strides):
            base_idx = level_index * 2
            cls_output = raw_outputs[self.output_names[base_idx]].reshape(-1, self.cfg.classes_num)
            box_output = raw_outputs[self.output_names[base_idx + 1]]

            scores, cls_ids, valid_indices = post_utils.filter_classification(cls_output, conf_thres_raw)
            if valid_indices.size == 0:
                continue

            grid_size = self.input_h // stride
            boxes = post_utils.decode_boxes(
                box_output,
                valid_indices,
                grid_size,
                stride,
                self.weights_static,
            )
            detections.append(np.hstack([boxes, scores[:, None], cls_ids[:, None].astype(np.float32)]))

        if not detections:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        dets = np.concatenate(detections, axis=0).astype(np.float32)
        final_boxes = []
        final_scores = []
        final_cls_ids = []
        for cls_id in np.unique(dets[:, 5].astype(np.int32)):
            cls_dets = dets[dets[:, 5] == cls_id]
            keep = post_utils.NMS(cls_dets[:, :4], cls_dets[:, 4], cls_dets[:, 5], nms_thres)
            if not keep:
                continue

            kept = cls_dets[keep]
            boxes = post_utils.scale_coords_back(
                kept[:, :4].copy(),
                ori_img_w,
                ori_img_h,
                self.input_w,
                self.input_h,
                self.cfg.resize_type,
            )
            final_boxes.append(boxes.astype(np.float32))
            final_scores.append(kept[:, 4].astype(np.float32))
            final_cls_ids.append(kept[:, 5].astype(np.int32))

        logger.info("\033[1;31mPost Process time = %.2f ms\033[0m", 1000 * (time.time() - t0))
        if not final_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.concatenate(final_boxes, axis=0),
            np.concatenate(final_scores, axis=0),
            np.concatenate(final_cls_ids, axis=0),
        )

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the full detection pipeline on one image.

        Args:
            img: One BGR image in OpenCV format.

        Returns:
            Detection results in `(boxes, scores, cls_ids)` format.
        """
        ori_img_h, ori_img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, ori_img_w, ori_img_h)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Provide function-style inference for one image."""
        return self.predict(img)
