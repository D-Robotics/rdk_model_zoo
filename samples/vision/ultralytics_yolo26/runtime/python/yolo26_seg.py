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

"""
YOLO26 Segmentation Inference Module.

This module implements the YOLO26 instance segmentation pipeline on BPU,
including pre-processing, forward execution, box decoding, and mask assembly.

Key Features:
    - Optimized for RDK X5 single-input NV12 models.
    - Supports YOLO26 segmentation head decoding and prototype masks.
    - Generates binary instance masks aligned to the original image size.

Typical Usage:
    >>> from yolo26_seg import YOLO26SegConfig, YOLO26Seg
    >>> config = YOLO26SegConfig(model_path="path/to/yolo26_seg.bin")
    >>> model = YOLO26Seg(config)
    >>> xyxy, score, cls, masks = model.predict(image)
"""

import os
import cv2
import time
import sys
import logging
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# Add project root to sys.path to import shared utilities.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_Seg")


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Crop instance masks using their corresponding bounding boxes.

    Args:
        masks (np.ndarray): Predicted masks with shape `(N, H, W)`.
        boxes (np.ndarray): Bounding boxes with shape `(N, 4)` in `xyxy` format.

    Returns:
        np.ndarray: Cropped masks constrained to their bounding boxes.
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
                 upsample: bool = False) -> np.ndarray:
    """
    Build instance masks from prototype features and mask coefficients.

    Args:
        protos (np.ndarray): Prototype feature map with shape `(C, H, W)`.
        masks_in (np.ndarray): Mask coefficients for each detection.
        bboxes (np.ndarray): Bounding boxes in model input coordinates.
        shape (Tuple[int, int]): Target `(height, width)` for output masks.
        upsample (bool): Whether to resize masks to the target image shape.

    Returns:
        np.ndarray: Binary instance masks aligned to the target image shape.
    """
    c, mh, mw = protos.shape
    ih, iw = shape

    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    masks = post_utils.sigmoid(masks)
    downsampled_bboxes = bboxes * (mh / 640.0) 
    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        resized_masks = []
        for m in masks:
            m_res = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(m_res)
        masks = np.array(resized_masks)

    return masks > 0.5


def decode_seg_layer(box_feat: np.ndarray,
                     cls_feat: np.ndarray,
                     mc_feat: np.ndarray,
                     stride: int,
                     score_thres: float,
                     classes_num: int = 80) -> np.ndarray:
    """
    Decode one segmentation head output.

    Args:
        box_feat (np.ndarray): Bounding box branch output tensor.
        cls_feat (np.ndarray): Classification branch output tensor.
        mc_feat (np.ndarray): Mask coefficient branch output tensor.
        stride (int): Feature stride for the current segmentation head.
        score_thres (float): Confidence threshold used for filtering.
        classes_num (int): Number of segmentation classes.

    Returns:
        np.ndarray: Decoded proposals in
        `(x1, y1, x2, y2, score, class_id, mask_coeffs...)` format.
    """
    if box_feat.shape[0] == 1: box_feat = box_feat[0]
    if cls_feat.shape[0] == 1: cls_feat = cls_feat[0]
    if mc_feat.shape[0] == 1: mc_feat = mc_feat[0]

    h, w, _ = box_feat.shape
    safe_thres = np.clip(score_thres, 1e-6, 1.0 - 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)
    max_logits = np.max(cls_feat, axis=-1)
    mask = max_logits >= logit_thres

    if not np.any(mask):
        return np.empty((0, 6 + 32), dtype=np.float32)

    grid_y, grid_x = np.indices((h, w))
    valid_grid_x = grid_x[mask]
    valid_grid_y = grid_y[mask]
    valid_box = box_feat[mask]
    valid_mc = mc_feat[mask]
    valid_cls_logits = cls_feat[mask]
    valid_cls_scores = post_utils.sigmoid(valid_cls_logits)
    valid_score = np.max(valid_cls_scores, axis=-1)
    valid_cls_id = np.argmax(valid_cls_scores, axis=-1)
    grid_center_x = valid_grid_x.astype(np.float32) + 0.5
    grid_center_y = valid_grid_y.astype(np.float32) + 0.5
    x1 = (grid_center_x - valid_box[:, 0]) * stride
    y1 = (grid_center_y - valid_box[:, 1]) * stride
    x2 = (grid_center_x + valid_box[:, 2]) * stride
    y2 = (grid_center_y + valid_box[:, 3]) * stride

    out = np.stack([x1, y1, x2, y2, valid_score, valid_cls_id], axis=-1)
    out = np.concatenate([out, valid_mc], axis=-1)
    return out


@dataclass
class YOLO26SegConfig:
    """
    Configuration for YOLO26 segmentation inference.

    Args:
        model_path (str): Path to the compiled BIN model file.
        classes_num (int): Number of segmentation classes.
        score_thres (float): Confidence threshold.
        nms_thres (float): IoU threshold for NMS.
        resize_type (int): Resize strategy (0: direct, 1: letterbox).
        strides (np.ndarray): Detection head strides.
    """
    model_path: str
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: np.ndarray = field(
        default_factory=lambda: np.array([8, 16, 32], dtype=np.int32)
    )


class YOLO26Seg:
    """
    YOLO26 segmentation wrapper based on hbm_runtime.

    This class follows the RDK Model Zoo coding standards for Python samples.
    """

    def __init__(self, config: YOLO26SegConfig):
        """
        Initialize the model and extract runtime metadata.

        Args:
            config (YOLO26SegConfig): Configuration object containing model path and params.
        """
        t0 = time.time()
        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31m[Seg] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

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

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        Set BPU scheduling parameters like priority and core affinity.

        Args:
            priority (Optional[int]): Scheduling priority (0-255).
            bpu_cores (Optional[list]): BPU core indexes to run inference.
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert a BGR image to packed NV12 input for hbm_runtime.

        Args:
            img (np.ndarray): Input image in BGR format.
            resize_type (Optional[int]): Override default resize strategy.
            image_format (Optional[str]): Input image format.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Prepared input tensors for hbm_runtime.run().
        """
        t0 = time.time()
        if resize_type is None:
            resize_type = self.cfg.resize_type

        if image_format == "BGR":
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        logger.info(f"\033[1;31m[Seg] Pre-process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        
        packed_nv12 = np.concatenate([y.reshape(-1), uv.reshape(-1)]).astype(np.uint8)
        return {
            self.model_name: {
                self.input_names[0]: packed_nv12
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Execute inference on BPU using hbm_runtime.

        Args:
            input_tensor (Dict[str, Dict[str, np.ndarray]]): Prepared input tensors.

        Returns:
            Dict[str, np.ndarray]: Raw output tensors from the runtime.
        """
        t0 = time.time()
        outputs = self.model.run(input_tensor)
        logger.info(f"\033[1;31m[Seg] Forward time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert raw model outputs to final segmentation results.

        Args:
            outputs (Dict[str, Dict[str, np.ndarray]]): Raw model outputs.
            ori_img_w (int): Original image width.
            ori_img_h (int): Original image height.
            score_thres (Optional[float]): Override confidence threshold.
            nms_thres (Optional[float]): Override NMS threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - xyxy: Bounding boxes.
                - score: Confidence scores.
                - cls: Class indices.
                - masks: Binary segmentation masks.
        """
        t0 = time.time()
        score_thres = score_thres or self.cfg.score_thres
        nms_thres = nms_thres or self.cfg.nms_thres
        raw_outputs = outputs[self.model_name]
        decoded = []

        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            cls_feat = raw_outputs[self.output_names[base_idx]]
            box_feat = raw_outputs[self.output_names[base_idx + 1]]
            mc_feat = raw_outputs[self.output_names[base_idx + 2]]
            
            layer_pred = decode_seg_layer(box_feat, cls_feat, mc_feat, 
                                          stride, score_thres, self.cfg.classes_num)
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

        masks = process_mask(proto_tensor, mask_coefs, xyxy, 
                             (ori_img_h, ori_img_w), upsample=True)

        xyxy = post_utils.scale_coords_back(xyxy, ori_img_w, ori_img_h,
                                            self.input_w, self.input_h, self.cfg.resize_type)

        logger.info(f"\033[1;31m[Seg] Post Process time = {1000 * (time.time() - t0):.2f} ms\033[0m")
        
        return xyxy, score, cls.astype(int), masks

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete segmentation pipeline on a single image.

        Args:
            img (np.ndarray): Input image.
            image_format (str): Input image format.
            resize_type (Optional[int]): Resize strategy.
            score_thres (Optional[float]): Confidence threshold.
            nms_thres (Optional[float]): NMS threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Segmentation results.
        """
        ori_img_h, ori_img_w = img.shape[:2]
        inp = self.pre_process(img, resize_type, image_format)
        out = self.forward(inp)
        return self.post_process(out, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide functional-style calling capability.

        Args:
            img (np.ndarray): Input image in BGR format.
            image_format (str): Input image format.
            resize_type (Optional[int]): Resize strategy.
            score_thres (Optional[float]): Confidence threshold.
            nms_thres (Optional[float]): NMS threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Segmentation results from `predict()`.
        """
        return self.predict(img, image_format, resize_type, score_thres, nms_thres)
