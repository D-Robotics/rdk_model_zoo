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

"""PaddleOCR text detection and recognition inference wrappers.

This module defines two runtime wrappers built on HBM runtime for a two-stage
OCR pipeline:

  1. ``PaddleOCRDet`` — DB-based text region detection using an NV12 input model.
  2. ``PaddleOCRRec`` — CRNN+CTC text recognition using a float32 RGB input model.

Each wrapper follows the standard Model Zoo pattern: configuration dataclass,
``pre_process``, ``forward``, ``post_process``, ``predict``, and ``__call__``.

Key Features:
    - NV12 YUV preprocessing for the detection model
    - Float32 RGB NCHW preprocessing for the recognition model
    - Pyclipper-based contour dilation for robust box extraction
    - Pure NumPy CTC greedy decode (no PaddlePaddle dependency)
    - Supports RDK S100 platform only

Typical Usage:
    >>> from paddle_ocr import PaddleOCRDet, PaddleOCRDetConfig
    >>> from paddle_ocr import PaddleOCRRec, PaddleOCRRecConfig
    >>> det = PaddleOCRDet(PaddleOCRDetConfig(model_path='...'))
    >>> img_boxes, crops, boxes = det.predict(img)
    >>> rec = PaddleOCRRec(PaddleOCRRecConfig(model_path='...'))
    >>> text = rec.predict(crops[0], char_list)

Notes:
    - This sample only supports RDK S100 platform.
    - RDK S600 users: the S100 model is not compatible with S600 BPU. Refer to
      README.md for platform compatibility details.
"""

import os
import cv2
import sys
import pyclipper
import hbm_runtime
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
#   utils/py_utils/preprocess.py
#   utils/py_utils/postprocess.py
#   utils/py_utils/visualize.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils
import utils.py_utils.visualize as vis_utils


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PaddleOCRDetConfig:
    """Configuration for the PaddleOCR text detection model.

    Attributes:
        model_path: Path to the compiled detection `.hbm` model (NV12 input).
        ratio_prime: Dilation ratio used when expanding detected text contours.
            Larger values produce wider bounding boxes.
        threshold: Binarization threshold applied to the float prediction map.
            Pixels whose value exceeds this threshold are treated as foreground.

    Notes:
        - This model only supports RDK S100 platform.
    """
    model_path: str = '/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm'
    ratio_prime: float = 2.7
    threshold: float = 0.5


@dataclass
class PaddleOCRRecConfig:
    """Configuration for the PaddleOCR text recognition model.

    Attributes:
        model_path: Path to the compiled recognition `.hbm` model (RGB F32 input).

    Notes:
        - This model only supports RDK S100 platform.
    """
    model_path: str = '/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm'


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def dilate_contours(contours: List[np.ndarray], ratio_prime: float) -> List[np.ndarray]:
    """Expand (dilate) contours using a data-driven offset distance.

    For each contour polygon, computes an expansion distance
    ``D' = area * ratio_prime / perimeter`` and performs polygon offsetting
    via Pyclipper. This is the standard DB post-processing step used in
    PaddleOCR text detection.

    Args:
        contours: List of contours from ``cv2.findContours``. Each element
            has shape ``(N, 1, 2)``.
        ratio_prime: Dilation scale factor. Larger values expand the polygons
            further from their original boundaries.

    Returns:
        List of dilated polygon arrays. Each element has shape ``(M, 1, 2)``
        with integer coordinates. Degenerate or multi-polygon results are
        silently skipped.
    """
    dilated_polys = []
    for poly in contours:
        poly = poly[:, 0, :]  # (N, 1, 2) -> (N, 2)

        arc_length = cv2.arcLength(poly, True)
        if arc_length == 0:
            continue

        D_prime = cv2.contourArea(poly) * ratio_prime / arc_length

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated_poly = np.array(pco.Execute(D_prime))

        if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
            continue

        dilated_polys.append(dilated_poly)

    return dilated_polys


def ctc_greedy_decode(logits: np.ndarray, char_list: List[str]) -> str:
    """Decode CTC logits to a string using greedy (best-path) decoding.

    Performs argmax at each timestep, collapses consecutive repeated tokens,
    and removes the CTC blank token (index 0). No external dependencies are
    required beyond NumPy.

    Args:
        logits: Model output logits with shape ``(1, T, V)`` or ``(T, V)``,
            where ``T`` is the sequence length and ``V`` is the vocabulary size
            (including the blank token at index 0).
        char_list: Token dictionary where ``char_list[i]`` is the string token
            for class index ``i``. ``char_list[0]`` is treated as the blank.

    Returns:
        The decoded text string. Returns an empty string if all timesteps
        predict blank or all tokens are repeats of a previous token.
    """
    # Flatten batch dimension if present
    if logits.ndim == 3:
        logits = logits[0]  # (T, V)

    result = []
    prev_idx = -1
    for t in range(logits.shape[0]):
        idx = int(np.argmax(logits[t]))
        if idx != 0 and idx != prev_idx:
            result.append(char_list[idx])
        prev_idx = idx

    return "".join(result)


# ---------------------------------------------------------------------------
# PaddleOCRDet
# ---------------------------------------------------------------------------

class PaddleOCRDet:
    """PaddleOCR text detection wrapper based on HB_HBMRuntime.

    This class provides a complete inference pipeline for the DB-based
    text detection model. It accepts a BGR image, preprocesses it into
    NV12 format, runs BPU inference, and postprocesses the binary
    prediction map into polygon bounding boxes and cropped text regions.

    Args:
        config: Configuration object. All parameters and their semantics
            are documented in ``PaddleOCRDetConfig``.

    Attributes:
        model: Loaded HBM runtime model handle.
        model_name: Name of the first model in the pack.
        input_names: List of input tensor names for the model.
        output_names: List of output tensor names for the model.
        input_shapes: Dictionary mapping input tensor names to shapes.
        output_quants: Output quantization parameters.
        input_H: Model input height (pixels).
        input_W: Model input width (pixels).
        threshold: Binarization threshold (float 0–1).
        ratio_prime: Contour dilation ratio.

    Notes:
        - This model only supports RDK S100 platform.
        - The detection model takes two NV12 inputs: Y plane and UV plane.
          Input shape layout is ``(1, H, W, 1)`` for Y and ``(1, H/2, W/2, 2)``
          for UV; height/width are read from indices ``[1]`` and ``[2]``.
    """

    def __init__(self, config: PaddleOCRDetConfig) -> None:
        """Initialize the detection model and extract metadata.

        Args:
            config: Configuration object containing model path, threshold,
                and ratio_prime. See ``PaddleOCRDetConfig`` for details.
        """
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # NV12 layout: shape is (1, H, W, 1) — H at index 1, W at index 2
        self.input_H = self.input_shapes[self.input_names[0]][1]
        self.input_W = self.input_shapes[self.input_names[0]][2]

        self.threshold = config.threshold
        self.ratio_prime = config.ratio_prime

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[List[int]] = None) -> None:
        """Configure BPU inference scheduling parameters.

        Args:
            priority: Inference priority in the range ``[0, 255]``.
                Higher values indicate higher scheduling priority.
            bpu_cores: List of BPU core indices to use for inference.

        Returns:
            None
        """
        kwargs: Dict = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess a BGR image into NV12 input tensors.

        Resizes the image to the model input resolution using direct resize
        (``INTER_AREA``), converts it to NV12 format, and formats the result
        as a nested tensor dictionary ready for ``forward()``.

        Args:
            img: Input image in BGR format with shape ``(H, W, 3)``.

        Returns:
            A nested input tensor dictionary:
            ``{model_name: {y_input_name: y_plane, uv_input_name: uv_plane}}``.
        """
        resized = pre_utils.resized_image(img, self.input_W, self.input_H, 0, cv2.INTER_AREA)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv,
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                ``pre_process()``, in the form
                ``{model_name: {input_name: tensor}}``.

        Returns:
            A dictionary containing raw output tensors returned by the runtime,
            in the form ``{model_name: {output_name: tensor}}``.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     img: np.ndarray,
                     img_w: int,
                     img_h: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Convert raw model outputs into text boxes and cropped regions.

        Steps:
        1. Extract the prediction map and binarize using ``self.threshold``.
        2. Resize the binary mask to the original image dimensions.
        3. Find external contours on the binary mask.
        4. Dilate contours using ``self.ratio_prime`` via Pyclipper.
        5. Extract minimum-area bounding boxes (min area = 100 px²).
        6. Draw polygon boxes on a copy of the original image.
        7. Perspective-crop and rectify each detected text region.

        Args:
            outputs: Raw output tensor dictionary from ``forward()``, in the
                form ``{model_name: {output_name: tensor}}``.
            img: Original input image in BGR format.
            img_w: Original image width in pixels.
            img_h: Original image height in pixels.

        Returns:
            A tuple containing:
                - img_boxes: Copy of ``img`` with polygon boxes drawn on it.
                - cropped_images: List of perspective-rectified text region
                  crops, one per detected box.
                - boxes_list: List of ``(4, 2)`` integer polygon vertex arrays.
        """
        preds = outputs[self.model_name][self.output_names[0]]
        preds = np.where(preds > self.threshold, 255, 0).astype(np.uint8).squeeze()

        preds = cv2.resize(preds, (img_w, img_h))

        contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dilated_polys = dilate_contours(contours, self.ratio_prime)

        boxes_list = post_utils.get_bounding_boxes(dilated_polys, 100)

        img_boxes = vis_utils.draw_polygon_boxes(img, boxes_list)

        cropped_images = []
        for box in boxes_list:
            cropped_img = post_utils.crop_and_rotate_image(img, box)
            cropped_images.append(cropped_img)

        return img_boxes, cropped_images, boxes_list

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Run the complete text detection pipeline on a single image.

        Internally calls ``pre_process``, ``forward``, and ``post_process``.

        Args:
            img: Input image in BGR format.

        Returns:
            Same as ``post_process()``: ``(img_boxes, cropped_images, boxes_list)``.
        """
        img_h, img_w = img.shape[:2]
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, img, img_w, img_h)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Callable interface for the text detection pipeline.

        Equivalent to calling ``predict()``.

        Args:
            img: Input image in BGR format.

        Returns:
            Same as ``predict()``.
        """
        return self.predict(img)


# ---------------------------------------------------------------------------
# PaddleOCRRec
# ---------------------------------------------------------------------------

class PaddleOCRRec:
    """PaddleOCR text recognition wrapper based on HB_HBMRuntime.

    This class provides a complete inference pipeline for the CRNN-based
    text recognition model. It accepts a cropped BGR word image, preprocesses
    it to float32 RGB NCHW format, runs BPU inference, and decodes the CTC
    logits into a text string.

    Args:
        config: Configuration object. All parameters and their semantics
            are documented in ``PaddleOCRRecConfig``.

    Attributes:
        model: Loaded HBM runtime model handle.
        model_name: Name of the first model in the pack.
        input_names: List of input tensor names.
        output_names: List of output tensor names.
        input_shapes: Dictionary mapping input tensor names to shapes.
        output_quants: Output quantization parameters.
        input_H: Model input height (pixels).
        input_W: Model input width (pixels).
        seq_len: Output sequence length T.
        num_classes: Output vocabulary size V (including blank).

    Notes:
        - This model only supports RDK S100 platform.
        - Input tensor layout is NCHW; height and width are at indices 2 and 3.
        - Output tensor shape is ``[1, T, V]`` (float32 logits).
    """

    def __init__(self, config: PaddleOCRRecConfig) -> None:
        """Initialize the recognition model and extract metadata.

        Args:
            config: Configuration object containing the model path.
                See ``PaddleOCRRecConfig`` for details.
        """
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # NCHW layout: shape is (N, C, H, W)
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        # Output shape: (1, T, V)
        output_shape = self.model.output_shapes[self.model_name][self.output_names[0]]
        self.seq_len = output_shape[1]
        self.num_classes = output_shape[2]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[List[int]] = None) -> None:
        """Configure BPU inference scheduling parameters.

        Args:
            priority: Inference priority in the range ``[0, 255]``.
            bpu_cores: List of BPU core indices to use for inference.

        Returns:
            None
        """
        kwargs: Dict = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess a cropped word image for recognition.

        Resizes to ``(input_W, input_H)``, normalizes to ``[0, 1]``,
        converts BGR to RGB, and reorders from HWC to NCHW.

        Args:
            img: Cropped text region in BGR format with shape ``(H, W, 3)``.

        Returns:
            A nested input tensor dictionary:
            ``{model_name: {input_name: float32_nchw_tensor}}``.
        """
        resized = cv2.resize(img, (self.input_W, self.input_H))
        resized = (resized / 255.0).astype(np.float32)

        # BGR -> RGB by swapping the channel axis
        rgb = resized[:, :, [2, 1, 0]]

        # HWC -> NCHW
        nchw = rgb[np.newaxis].transpose(0, 3, 1, 2)

        return {
            self.model_name: {
                self.input_names[0]: nchw,
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                ``pre_process()``, in the form
                ``{model_name: {input_name: tensor}}``.

        Returns:
            Raw output tensor dictionary in the form
            ``{model_name: {output_name: tensor}}``.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     char_list: List[str]) -> str:
        """Decode CTC logits into a text string.

        Args:
            outputs: Raw output tensor dictionary from ``forward()``, in the
                form ``{model_name: {output_name: tensor}}``.
            char_list: Token dictionary where ``char_list[0]`` is the blank
                token and ``char_list[i]`` (i > 0) maps to a character.

        Returns:
            The decoded text string.
        """
        logits = outputs[self.model_name][self.output_names[0]]
        return ctc_greedy_decode(logits, char_list)

    def predict(self, img: np.ndarray, char_list: List[str]) -> str:
        """Run the complete text recognition pipeline on a single crop.

        Internally calls ``pre_process``, ``forward``, and ``post_process``.

        Args:
            img: Cropped text region in BGR format.
            char_list: Token dictionary (blank at index 0).

        Returns:
            The recognized text string.
        """
        input_tensor = self.pre_process(img)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs, char_list)

    def __call__(self, img: np.ndarray, char_list: List[str]) -> str:
        """Callable interface for the text recognition pipeline.

        Equivalent to calling ``predict()``.

        Args:
            img: Cropped text region in BGR format.
            char_list: Token dictionary (blank at index 0).

        Returns:
            The recognized text string.
        """
        return self.predict(img, char_list)
