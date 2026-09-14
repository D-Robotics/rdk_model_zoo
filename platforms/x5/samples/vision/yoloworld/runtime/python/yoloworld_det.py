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

"""
YOLOWorld open-vocabulary detection inference module.

This module implements the RDK X5 runtime protocol for YOLOWorld. The deployed
model uses two FP32 inputs: one RGB image tensor and one 32-slot offline text
embedding tensor. Outputs are fixed as class scores and bounding boxes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.postprocess as post_utils


@dataclass
class YOLOWorldConfig:
    """
    Configuration for YOLOWorld open-vocabulary detection.

    Args:
        model_path: Path to the compiled RDK X5 `.bin` model.
        vocab_file: Path to the offline vocabulary embedding JSON file.
        score_thres: Confidence threshold used to filter predictions.
        nms_thres: IoU threshold used by class-wise NMS.
        input_size: Square model input resolution.
        text_slots: Number of text embedding slots expected by the model.
    """

    model_path: str
    vocab_file: str
    score_thres: float = 0.05
    nms_thres: float = 0.45
    input_size: int = 640
    text_slots: int = 32


class YOLOWorldDetect:
    """
    YOLOWorld detection wrapper based on `hbm_runtime`.

    The wrapper exposes the standard sample methods and keeps the exported
    YOLOWorld tensor protocol fixed: image input, text embedding input,
    class-score output, and bounding-box output.
    """

    def __init__(self, config: YOLOWorldConfig):
        """
        Initialize the runtime and load offline vocabulary embeddings.

        Args:
            config: Runtime configuration containing model and vocabulary
                paths plus post-processing thresholds.
        """

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        with open(config.vocab_file, "r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)
        self.class_names = list(self.vocabulary.keys())
        self._scale = 1.0
        self._selected_class_ids = np.zeros((self.cfg.text_slots,), dtype=np.int32)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """
        Set optional BPU runtime scheduling parameters.

        Args:
            priority: Runtime priority in the range `0~255`.
            bpu_cores: BPU core indexes used for model execution.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def _build_text_embeddings(self, prompts: List[str]) -> np.ndarray:
        """
        Build the fixed 32-slot text embedding tensor for selected prompts.

        Args:
            prompts: Prompt words that must exist in the offline vocabulary.

        Returns:
            Text embedding tensor with shape `(1, 32, 512, 1)`.
        """

        class_ids, embeddings = [], []
        for prompt in prompts:
            if prompt not in self.vocabulary:
                raise KeyError(f"Prompt '{prompt}' is not in offline vocabulary.")
            class_ids.append(self.class_names.index(prompt))
            embeddings.append(np.asarray(self.vocabulary[prompt], dtype=np.float32))

        while len(class_ids) < self.cfg.text_slots:
            class_ids.append(class_ids[-1])
            embeddings.append(embeddings[-1])

        self._selected_class_ids = np.asarray(class_ids[: self.cfg.text_slots], dtype=np.int32)
        return np.asarray(embeddings[: self.cfg.text_slots], dtype=np.float32).reshape(1, self.cfg.text_slots, -1, 1)

    def pre_process(self, image: np.ndarray, prompts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert one BGR image and prompt list into model inputs.

        Args:
            image: Input image in OpenCV BGR format.
            prompts: Prompt words used for open-vocabulary detection.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.
        """

        ori_h, ori_w = image.shape[:2]
        resize_scale = self.cfg.input_size / max(ori_h, ori_w)
        self._scale = max(ori_h, ori_w) / self.cfg.input_size
        resized = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        input_image = np.zeros((self.cfg.input_size, self.cfg.input_size, 3), dtype=np.float32)
        input_image[: resized.shape[0], : resized.shape[1], :] = resized.astype(np.float32)
        input_image = input_image[:, :, [2, 1, 0]][None].transpose(0, 3, 1, 2)
        text_embeddings = self._build_text_embeddings(prompts)
        return {
            self.model_name: {
                self.input_names[0]: input_image,
                self.input_names[1]: text_embeddings,
            }
        }

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Execute one BPU inference pass.

        Args:
            inputs: Prepared image and text embedding tensors.

        Returns:
            Raw runtime output dictionary returned by `hbm_runtime`.
        """

        return self.model.run(inputs)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        ori_img_w: int,
        ori_img_h: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert YOLOWorld raw outputs into final detection arrays.

        Args:
            outputs: Raw runtime outputs from `forward()`.
            ori_img_w: Original image width.
            ori_img_h: Original image height.

        Returns:
            A tuple `(boxes, scores, cls_ids)` in original image coordinates.
        """

        raw_outputs = outputs[self.model_name]
        class_scores = raw_outputs[self.output_names[0]].squeeze(-1)
        boxes = raw_outputs[self.output_names[1]].squeeze(-1)
        rows = class_scores.shape[1]

        selected_boxes, scores, class_slots = [], [], []
        for i in range(rows):
            slot = int(np.argmax(class_scores[0, i]))
            score = float(class_scores[0, i, slot])
            if score >= self.cfg.score_thres:
                selected_boxes.append(boxes[0, i])
                scores.append(score)
                class_slots.append(slot)

        if not selected_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        selected_boxes = np.asarray(selected_boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_slots = np.asarray(class_slots, dtype=np.int32)
        keep = post_utils.NMS(selected_boxes, scores, class_slots, self.cfg.nms_thres)
        if not keep:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        final_boxes = selected_boxes[keep] * self._scale
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, ori_img_w)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, ori_img_h)
        final_scores = scores[keep]
        final_cls_ids = self._selected_class_ids[class_slots[keep]]
        return final_boxes, final_scores, final_cls_ids

    def predict(self, image: np.ndarray, prompts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete YOLOWorld detection pipeline on one image.

        Args:
            image: Input image in OpenCV BGR format.
            prompts: Prompt words used for detection.

        Returns:
            Final detection arrays from `post_process()`.
        """

        ori_img_h, ori_img_w = image.shape[:2]
        start = time.perf_counter()
        inputs = self.pre_process(image, prompts)
        t_pre = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        outputs = self.forward(inputs)
        t_forward = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        results = self.post_process(outputs, ori_img_w, ori_img_h)
        t_post = (time.perf_counter() - start) * 1000

        print(f"\n[Log] Pre-process: {t_pre:.2f} ms | Inference: {t_forward:.2f} ms | Post-process: {t_post:.2f} ms")
        return results

    def __call__(self, image: np.ndarray, prompts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide functional-style access to `predict()`.

        Args:
            image: Input image in OpenCV BGR format.
            prompts: Prompt words used for detection.

        Returns:
            The same tuple returned by `predict()`.
        """

        return self.predict(image, prompts)
