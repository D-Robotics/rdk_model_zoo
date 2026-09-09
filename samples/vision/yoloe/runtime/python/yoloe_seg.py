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
YOLOE Segmentation Inference Module.

This module implements the YOLOE-11 instance segmentation (Prompt-Free)
pipeline on BPU, including pre-processing, forward execution, DFL box
decoding, mask assembly, and NMS.

The BPU model outputs 10 tensors per forward pass:
  - 3 classification heads  (stride 8, 16, 32)
  - 3 bounding-box heads    (raw DFL distributions, 16 bins * 4 sides)
  - 3 mask-coefficient heads
  - 1 prototype tensor

Key Features:
    - 4585-class open-vocabulary instance segmentation.
    - DFL box decoding at runtime (softmax + weighted expectation).
    - Letterbox-aware coordinate scaling back to original image space.

Typical Usage:
    >>> from yoloe_seg import YOLOESegConfig, YOLOESeg
    >>> config = YOLOESegConfig(model_path="path/to/yoloe_seg.bin")
    >>> model = YOLOESeg(config)
    >>> xyxy, score, cls, masks = model.predict(image)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


logger = logging.getLogger("YOLOE_Seg")


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Crop instance masks using their corresponding bounding boxes.

    Args:
        masks: Predicted masks with shape ``(N, H, W)``.
        boxes: Bounding boxes with shape ``(N, 4)`` in ``xyxy`` format.

    Returns:
        Cropped masks constrained to their bounding boxes.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    r = np.arange(w, dtype=np.float32)[None, None, :]
    c = np.arange(h, dtype=np.float32)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos: np.ndarray,
                 masks_in: np.ndarray,
                 bboxes: np.ndarray,
                 shape: Tuple[int, int],
                 upsample: bool = False,
                 input_shape: Tuple[int, int] = (640, 640),
                 resize_type: int = 1) -> np.ndarray:
    """Build instance masks from prototype features and mask coefficients.

    Args:
        protos: Prototype feature map with shape ``(C, H, W)``.
        masks_in: Mask coefficients for each detection.
        bboxes: Bounding boxes in model input coordinates.
        shape: Target ``(height, width)`` for output masks.
        upsample: Whether to resize masks to the target image shape.
        input_shape: Model input height and width.
        resize_type: Match the runtime resize (0) or letterbox (1).

    Returns:
        Binary instance masks aligned to the target image shape.
    """
    c, mh, mw = protos.shape
    ih, iw = shape

    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    masks = post_utils.sigmoid(masks)
    input_h, input_w = input_shape
    downsampled_bboxes = bboxes * np.array([mw / input_w, mh / input_h,
                                           mw / input_w, mh / input_h])
    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        resized_masks = []
        for m in masks:
            m = cv2.resize(m, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            if resize_type == 1:
                scale = min(input_h / ih, input_w / iw)
                new_w, new_h = int(iw * scale), int(ih * scale)
                left, top = (input_w - new_w) // 2, (input_h - new_h) // 2
                m = m[top:top + new_h, left:left + new_w]
            elif resize_type != 0:
                raise ValueError("resize_type must be 0 or 1")
            m_res = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(m_res)
        masks = np.array(resized_masks)

    return masks > 0.5


def decode_seg_layer_dfl(box_feat: np.ndarray,
                         cls_feat: np.ndarray,
                         mc_feat: np.ndarray,
                         stride: int,
                         score_thres: float,
                         classes_num: int,
                         reg: int,
                         weights_static: np.ndarray) -> np.ndarray:
    """Decode one segmentation head output with DFL box regression.

    Unlike ``decode_seg_layer`` (which assumes pre-decoded 4-value bbox),
    this function applies DFL expectation (softmax + weighted sum) on the
    raw 16-bin distributions before computing anchor-based box coordinates.

    Args:
        box_feat: Bounding box branch output, shape ``(1, H, W, reg*4)``.
        cls_feat: Classification branch output, shape ``(1, H, W, C)``.
        mc_feat: Mask coefficient branch output, shape ``(1, H, W, mc)``.
        stride: Feature stride for the current head.
        score_thres: Confidence threshold for filtering.
        classes_num: Number of classes.
        reg: DFL bin count (typically 16).
        weights_static: DFL expectation weights, shape ``(1, 1, reg)``.

    Returns:
        Decoded proposals in ``(x1, y1, x2, y2, score, cls_id, mc...)`` format.
    """
    if box_feat.shape[0] == 1:
        box_feat = box_feat[0]
    if cls_feat.shape[0] == 1:
        cls_feat = cls_feat[0]
    if mc_feat.shape[0] == 1:
        mc_feat = mc_feat[0]

    grid_size = box_feat.shape[0]

    # Classification filtering in logit space
    conf_raw = -np.log(1.0 / np.clip(score_thres, 1e-6, 1.0 - 1e-6) - 1.0)
    scores, cls_ids, valid_indices = post_utils.filter_classification(
        cls_feat.reshape(-1, classes_num), conf_raw
    )

    if valid_indices.size == 0:
        return np.empty((0, 6 + mc_feat.shape[-1]), dtype=np.float32)

    # DFL box decoding
    xyxy = post_utils.decode_boxes(
        box_feat.reshape(-1, reg * 4), valid_indices, grid_size, stride, weights_static
    )

    # Mask coefficients
    mc = post_utils.filter_mces(mc_feat.reshape(-1, mc_feat.shape[-1]), valid_indices)

    out = np.hstack([
        xyxy,
        scores[:, None],
        cls_ids[:, None].astype(np.float32),
        mc,
    ])
    return out


@dataclass
class YOLOESegConfig:
    """Configuration for YOLOE segmentation inference.

    Args:
        model_path: Path to the compiled BIN model file.
        classes_num: Number of segmentation classes (4585 for Prompt-Free).
        score_thres: Confidence threshold.
        nms_thres: IoU threshold for NMS.
        resize_type: Resize strategy (0: direct, 1: letterbox).
        strides: Detection head strides.
        reg: DFL bin count per side.
        mc: Mask coefficient dimension.
    """
    model_path: str
    classes_num: int = 4585
    score_thres: float = 0.25
    nms_thres: float = 0.70
    resize_type: int = 1
    strides: np.ndarray = field(
        default_factory=lambda: np.array([8, 16, 32], dtype=np.int32)
    )
    reg: int = 16
    mc: int = 32


class YOLOESeg:
    """YOLOE instance segmentation wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    It encapsulates the YOLOE-11 Seg Prompt-Free model, providing a unified
    inference pipeline with DFL box decoding and prototype mask generation.
    """

    def __init__(self, config: YOLOESegConfig) -> None:
        """Initialize the model and extract runtime metadata.

        Args:
            config: Configuration object containing model path and params.
        """
        t0 = time.time()
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31mLoad Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        if len(self.input_names) != 1 or len(self.output_names) != 10:
            raise ValueError("YOLOE-11 PF requires one image input and ten NHWC outputs.")
        input_shape = self.input_shapes[self.input_names[0]]
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError(f"Unsupported input metadata: {input_shape}")
        if input_shape[1] == 3:
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        elif input_shape[3] == 3:
            self.input_h = input_shape[1]
            self.input_w = input_shape[2]
        else:
            raise ValueError(f"Expected RGB-shaped NV12 input metadata, got {input_shape}")
        if (self.input_h, self.input_w) != (640, 640):
            raise ValueError("This YOLOE-11 PF runtime currently supports 640x640 models only.")
        if (self.cfg.classes_num, self.cfg.reg, self.cfg.mc) != (4585, 16, 32):
            raise ValueError("YOLOE-11 PF requires classes=4585, DFL=16, mask channels=32.")
        if list(self.cfg.strides) != [8, 16, 32]:
            raise ValueError("YOLOE-11 PF requires strides 8,16,32.")
        self._resize_type = self.cfg.resize_type

        self.weights_static = np.arange(self.cfg.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set BPU scheduling parameters like priority and core affinity.

        Args:
            priority: Scheduling priority (0-255).
            bpu_cores: BPU core indexes to run inference.
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
        image_format: Optional[str] = "BGR",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert a BGR image to packed NV12 input for hbm_runtime.

        Args:
            img: Input image in BGR format.
            resize_type: Override default resize strategy.
            image_format: Input image format.

        Returns:
            Prepared input tensors for hbm_runtime.run().

        Raises:
            ValueError: If ``image_format`` is not ``"BGR"``.
        """
        t0 = time.time()
        if resize_type is None:
            resize_type = self.cfg.resize_type
        self._resize_type = resize_type

        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        logger.info(f"\033[1;31mPre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)
        return {self.model_name: {self.input_names[0]: packed_nv12}}

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute inference on BPU using hbm_runtime.

        Args:
            input_tensor: Prepared input tensors.

        Returns:
            Raw output tensors from the runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info(f"\033[1;31mForward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(
        self,
        outputs: Dict[str, Dict[str, np.ndarray]],
        ori_img_w: int,
        ori_img_h: int,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs to final segmentation results.

        Args:
            outputs: Raw model outputs.
            ori_img_w: Original image width.
            ori_img_h: Original image height.
            score_thres: Override confidence threshold.
            nms_thres: Override NMS threshold.

        Returns:
            Tuple of ``(xyxy, score, cls, masks)`` where masks are binary
            arrays aligned to the original image dimensions.
        """
        t0 = time.time()
        if score_thres is None:
            score_thres = self.cfg.score_thres
        if nms_thres is None:
            nms_thres = self.cfg.nms_thres

        raw_outputs = outputs[self.model_name]
        expected = []
        for size in (80, 40, 20):
            expected.extend([(1, size, size, c) for c in (4585, 64, 32)])
        expected.append((1, 160, 160, 32))
        for name, shape in zip(self.output_names, expected):
            tensor = raw_outputs[name]
            if tensor.shape != shape or not np.issubdtype(tensor.dtype, np.floating):
                raise ValueError(f"Output {name}: expected dequantized float {shape}, "
                                 f"got {tensor.shape} {tensor.dtype}")
        decoded = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_feat = raw_outputs[self.output_names[base_idx]]
            box_feat = raw_outputs[self.output_names[base_idx + 1]]
            mc_feat = raw_outputs[self.output_names[base_idx + 2]]

            layer_pred = decode_seg_layer_dfl(
                box_feat, cls_feat, mc_feat,
                stride, score_thres, self.cfg.classes_num,
                self.cfg.reg, self.weights_static,
            )
            decoded.append(layer_pred)

        proto_tensor = raw_outputs[self.output_names[9]]
        if proto_tensor.shape[0] == 1:
            proto_tensor = proto_tensor[0]
        proto_tensor = np.transpose(proto_tensor, (2, 0, 1))

        if not decoded:
            return np.array([]), np.array([]), np.array([]), np.array([])

        pred = np.concatenate(decoded, axis=0)
        if pred.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        mask_coefs = pred[:, 6:]

        keep = post_utils.NMS(xyxy, score, cls, nms_thres)
        if not keep:
            return np.array([]), np.array([]), np.array([]), np.array([])

        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        mask_coefs = mask_coefs[keep]

        masks = process_mask(
            proto_tensor, mask_coefs, xyxy,
            (ori_img_h, ori_img_w), upsample=True,
            input_shape=(self.input_h, self.input_w), resize_type=self._resize_type,
        )

        xyxy = post_utils.scale_coords_back(
            xyxy, ori_img_w, ori_img_h,
            self.input_w, self.input_h, self._resize_type,
        )

        logger.info(f"\033[1;31mPost Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return xyxy, score, cls.astype(int), masks

    def predict(
        self,
        img: np.ndarray,
        image_format: str = "BGR",
        resize_type: Optional[int] = None,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete segmentation pipeline on a single image.

        Args:
            img: Input image in BGR format.
            image_format: Input image format.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: NMS threshold override.

        Returns:
            Tuple of ``(xyxy, score, cls, masks)``.
        """
        ori_img_h, ori_img_w = img.shape[:2]
        inp = self.pre_process(img, resize_type, image_format)
        out = self.forward(inp)
        return self.post_process(out, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(
        self,
        img: np.ndarray,
        image_format: str = "BGR",
        resize_type: Optional[int] = None,
        score_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Provide functional-style calling capability, equivalent to ``predict()``.

        Args:
            img: Input image in BGR format.
            image_format: Input image format.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: NMS threshold override.

        Returns:
            Tuple of ``(xyxy, score, cls, masks)``.
        """
        return self.predict(img, image_format, resize_type, score_thres, nms_thres)
