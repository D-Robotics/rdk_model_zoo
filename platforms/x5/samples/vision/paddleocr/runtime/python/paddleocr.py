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
PaddleOCR Inference Module.

This module implements the PaddleOCR text detection and recognition pipeline
on BPU, including pre-processing, forward execution, and post-processing.

The pipeline uses two models:
  - A detection model (NV12 input) based on DB text detection.
  - A recognition model (float32 featuremap input) based on CTC decoding.

Key Features:
    - Two-stage OCR: DB detection followed by CTC recognition.
    - Supports NV12 input for detection and float32 NCHW RGB for recognition.
    - Provides a complete `predict()` pipeline for Python samples.

Typical Usage:
    >>> from paddleocr import PaddleOCRConfig, PaddleOCR
    >>> config = PaddleOCRConfig(det_model_path="path/to/det.bin", rec_model_path="path/to/rec.bin")
    >>> model = PaddleOCR(config)
    >>> boxes, texts = model.predict(image)
"""

from __future__ import annotations

import collections.abc
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np
import pyclipper

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.preprocess as pre_utils


logger = logging.getLogger("PaddleOCR")

# CTC alphabet for PP-OCRv3 English recognition
ALPHABET = r"""0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """


@dataclass
class PaddleOCRConfig:
    """Configuration for PaddleOCR inference.

    Args:
        det_model_path (str): Path to the detection model (.bin).
        rec_model_path (str): Path to the recognition model (.bin).
        det_threshold (float): Binarization threshold for detection output.
        det_ratio_prime (float): Dilation ratio for contour expansion.
        det_min_area (int): Minimum contour area to keep.
        rec_input_size (Tuple[int, int]): Recognition model input (H, W).
        rec_output_size (Tuple[int, int]): Recognition model output (T, C).
    """
    det_model_path: str = ""
    rec_model_path: str = ""
    det_threshold: float = 0.5
    det_ratio_prime: float = 2.7
    det_min_area: int = 100
    rec_input_size: Tuple[int, int] = (48, 320)
    rec_output_size: Tuple[int, int] = (40, 97)


class _CTCLabelConverter:
    """Convert between string and label indices for CTC decoding."""

    def __init__(self, alphabet: str, ignore_case: bool = False) -> None:
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + "-"
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def decode(self, t: np.ndarray, length: np.ndarray, raw: bool = False) -> str:
        """Decode a sequence of CTC indices into a string.

        Args:
            t (np.ndarray): Encoded indices.
            length (np.ndarray): Length of the sequence.
            raw (bool): If True, return raw decoding with blanks and repeats.

        Returns:
            str: Decoded text.
        """
        if len(length) == 1:
            length = length[0]
            assert len(t) == length, (
                f"text with length: {len(t)} does not match declared length: {length}"
            )
            if raw:
                return "".join([self.alphabet[i - 1] for i in t])
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return "".join(char_list)
        texts = []
        index = 0
        for i in range(length.size):
            l = length[i]
            texts.append(self.decode(t[index : index + l], np.array([l]), raw=raw))
            index += l
        return texts


class PaddleOCR:
    """PaddleOCR two-stage detection and recognition wrapper.

    This class follows the RDK Model Zoo coding standards for Python samples.
    It encapsulates both the DB text detection model and the CTC recognition
    model, providing a unified inference pipeline.
    """

    def __init__(self, config: PaddleOCRConfig) -> None:
        """Initialize both detection and recognition models.

        Args:
            config (PaddleOCRConfig): Configuration object containing model paths and params.
        """
        self.cfg = config

        # Detection model
        self.det_model = hbm_runtime.HB_HBMRuntime(config.det_model_path)
        self.det_model_name = self.det_model.model_names[0]
        self.det_input_name = self.det_model.input_names[self.det_model_name][0]
        self.det_input_shape = self.det_model.input_shapes[self.det_model_name][self.det_input_name]
        self.det_output_name = self.det_model.output_names[self.det_model_name][0]
        self.det_input_h = self.det_input_shape[2]
        self.det_input_w = self.det_input_shape[3]

        # Recognition model
        self.rec_model = hbm_runtime.HB_HBMRuntime(config.rec_model_path)
        self.rec_model_name = self.rec_model.model_names[0]
        self.rec_input_name = self.rec_model.input_names[self.rec_model_name][0]
        self.rec_output_name = self.rec_model.output_names[self.rec_model_name][0]

        # CTC converter
        self.converter = _CTCLabelConverter(ALPHABET)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set BPU scheduling parameters for both models.

        Args:
            priority (Optional[int]): Scheduling priority (0-255).
            bpu_cores (Optional[List[int]]): BPU core indexes to run inference.
        """
        det_kwargs = {}
        rec_kwargs = {}
        if priority is not None:
            det_kwargs["priority"] = {self.det_model_name: priority}
            rec_kwargs["priority"] = {self.rec_model_name: priority}
        if bpu_cores is not None:
            det_kwargs["bpu_cores"] = {self.det_model_name: bpu_cores}
            rec_kwargs["bpu_cores"] = {self.rec_model_name: bpu_cores}
        if det_kwargs:
            self.det_model.set_scheduling_params(**det_kwargs)
            self.rec_model.set_scheduling_params(**rec_kwargs)

    def pre_process(
        self, image: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert input image to NV12 for the detection model.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, 3).

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared input tensors for det model.

        Raises:
            ValueError: If input image is None.
        """
        if image is None:
            raise ValueError("Input image is None")

        resized = cv2.resize(image, (self.det_input_w, self.det_input_h))
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        nv12 = np.concatenate(
            (y.reshape(-1), uv.reshape(-1)), axis=0
        ).reshape((1, self.det_input_h * 3 // 2, self.det_input_w, 1))

        return {self.det_model_name: {self.det_input_name: nv12.astype(np.uint8)}}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute detection model inference on BPU.

        Args:
            input_tensor (Dict[str, Dict[str, np.ndarray]]): Prepared input tensors.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Raw output tensors from the runtime.
        """
        return self.det_model.run(input_tensor)

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        image: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run DB post-processing: binarize, find contours, dilate, compute boxes.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw output tensors from forward().
            image (Optional[np.ndarray]): Original image for coordinate mapping.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (dilated_polys, boxes_list)
        """
        raw = outputs[self.det_model_name][self.det_output_name]
        img_h, img_w = image.shape[:2] if image is not None else (self.det_input_h, self.det_input_w)

        preds = raw.reshape(1, self.det_input_h, self.det_input_w)
        preds = np.where(preds > self.cfg.det_threshold, 255, 0).astype(np.uint8).squeeze()
        preds = cv2.resize(preds, (img_w, img_h))

        contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_polys = self._dilate_contours(contours)
        boxes_list = self._get_bounding_boxes(dilated_polys, self.cfg.det_min_area)

        return dilated_polys, boxes_list

    def _dilate_contours(self, contours: list) -> List[np.ndarray]:
        """Dilate contours using the ratio_prime parameter."""
        dilated_polys = []
        for poly in contours:
            poly = poly[:, 0, :]
            arc_length = cv2.arcLength(poly, True)
            if arc_length == 0:
                continue
            d_prime = cv2.contourArea(poly) * self.cfg.det_ratio_prime / arc_length

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(pco.Execute(d_prime))

            if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
                continue
            dilated_polys.append(dilated_poly)
        return dilated_polys

    def _get_bounding_boxes(
        self, dilated_polys: List[np.ndarray], min_area: int
    ) -> List[np.ndarray]:
        """Compute minimum-area bounding rectangles from dilated polygons."""
        boxes_list = []
        for cnt in dilated_polys:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int_)
            boxes_list.append(box)
        return boxes_list

    def _rec_pre_process(self, cropped_img: np.ndarray) -> np.ndarray:
        """Pre-process a cropped image for the recognition model.

        Args:
            cropped_img (np.ndarray): Cropped text region in BGR.

        Returns:
            np.ndarray: Float32 NCHW RGB tensor ready for rec model.
        """
        h, w = self.cfg.rec_input_size
        resized = cv2.resize(cropped_img, (w, h))
        resized = (resized / 255.0).astype(np.float32)
        resized = resized[:, :, [2, 1, 0]]  # BGR -> RGB
        return resized[None].transpose(0, 3, 1, 2)  # NHWC -> NCHW

    def _rec_forward(self, input_tensor: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute recognition model inference on BPU.

        Args:
            input_tensor (np.ndarray): Float32 NCHW RGB tensor.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Raw output tensors from the runtime.
        """
        return self.rec_model.run({self.rec_model_name: {self.rec_input_name: input_tensor}})

    def _rec_post_process(
        self, outputs: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[str, str]:
        """Run CTC decoding on recognition model output.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw output tensors.

        Returns:
            Tuple[str, str]: (raw_prediction, simplified_prediction)
        """
        t, c = self.cfg.rec_output_size
        preds = outputs[self.rec_model_name][self.rec_output_name]
        preds = preds.reshape(1, t, c)
        preds = np.transpose(preds, (1, 0, 2))
        preds = np.argmax(preds, axis=2)
        preds = preds.transpose(1, 0).reshape(-1)
        preds_size = np.array([preds.size], dtype=np.int32)
        raw_pred = self.converter.decode(np.array(preds), preds_size, raw=True)
        sim_pred = self.converter.decode(np.array(preds), preds_size, raw=False)
        return raw_pred, sim_pred

    @staticmethod
    def _crop_and_rotate(img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Crop and perspective-rectify a text region from the image.

        Args:
            img (np.ndarray): Original image.
            box (np.ndarray): 4-point oriented bounding box.

        Returns:
            np.ndarray: Cropped and rotated text patch.
        """
        rect = cv2.minAreaRect(box)
        box_pts = cv2.boxPoints(rect).astype(np.intp)
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = rect[2]

        if width == 0 or height == 0:
            return img

        src_pts = box_pts.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        if angle >= 45:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped

    def predict(
        self,
        image: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Run the complete OCR pipeline: detect → crop → recognize.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            Tuple[List[np.ndarray], List[str]]: (boxes, recognized_texts)
        """
        # Step 1: Detection
        s1 = time.perf_counter()
        input_tensors = self.pre_process(image)
        det_outputs = self.forward(input_tensors)
        _, boxes = self.post_process(det_outputs, image)
        t1 = (time.perf_counter() - s1) * 1000

        # Step 2: Recognition for each box
        s2 = time.perf_counter()
        texts = []
        for box in boxes:
            cropped = self._crop_and_rotate(image, box)
            rec_input = self._rec_pre_process(cropped)
            rec_outputs = self._rec_forward(rec_input)
            _, sim_pred = self._rec_post_process(rec_outputs)
            texts.append(sim_pred)
        t2 = (time.perf_counter() - s2) * 1000

        logger.info(f"Detection: {t1:.2f} ms | Recognition ({len(boxes)} boxes): {t2:.2f} ms")
        return boxes, texts

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Provide functional-style calling capability, equivalent to predict().

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            Tuple[List[np.ndarray], List[str]]: (boxes, recognized_texts)
        """
        return self.predict(image)
