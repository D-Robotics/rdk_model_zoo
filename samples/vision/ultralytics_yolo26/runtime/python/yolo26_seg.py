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

"""Provide a YOLO26 Segmentation inference wrapper and pipeline utilities.

This module defines a lightweight YOLO26 Instance Segmentation runtime wrapper 
built on HBM runtime. It handles Box, Class, Mask Coefficients decoding, 
and Prototype Mask assembly.

Model Structure Assumption (based on provided logs):
    Inputs:
        0: images_y (1, 640, 640, 1)
        1: images_uv (1, 320, 320, 2)
    Outputs (10 tensors):
        Stride 8:  [0] Box(4), [1] Cls(80), [2] MaskCoef(32)
        Stride 16: [3] Box(4), [4] Cls(80), [5] MaskCoef(32)
        Stride 32: [6] Box(4), [7] Cls(80), [8] MaskCoef(32)
        Proto:     [9] ProtoMaps (1, 160, 160, 32)
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

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils

logger = logging.getLogger("YOLO26_Seg")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function.

    Applies the sigmoid function element-wise to the input NumPy array.

    Args:
        x: Input NumPy array.

    Returns:
        A NumPy array with the sigmoid function applied element-wise.
    """
    return 1.0 / (1.0 + np.exp(-x))


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Crop masks to bounding boxes to remove background noise.

    This function masks out any pixels in the generated masks that fall 
    outside the corresponding bounding box coordinates. It ensures that 
    segmentation masks are localized strictly within detected objects.

    Args:
        masks: Generated masks with shape `(N, H, W)`.
        boxes: Bounding boxes with shape `(N, 4)` in format `[x1, y1, x2, y2]`.

    Returns:
        Cropped masks with shape `(N, H, W)`, where values outside the 
        bounding boxes are set to 0.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    
    # Create coordinate grids
    r = np.arange(w, dtype=np.float32)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=np.float32)[None, :, None]  # cols shape(1,h,1)

    # Boolean mask: Valid if within box coordinates
    # Logic: (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos: np.ndarray, 
                 masks_in: np.ndarray, 
                 bboxes: np.ndarray, 
                 shape: Tuple[int, int], 
                 upsample: bool = False) -> np.ndarray:
    """Process masks from prototypes and coefficients.

    Performs matrix multiplication between Mask Coefficients (per object) 
    and Proto Maps (global features), applies sigmoid activation, crops 
    masks to the bounding box, and optionally upsamples to the original 
    image resolution.

    Args:
        protos: Prototype masks tensor with shape `(32, 160, 160)`.
        masks_in: Mask coefficients with shape `(N, 32)`.
        bboxes: Bounding boxes with shape `(N, 4)` scaled to input size (640x640).
        shape: Original image shape `(h, w)`.
        upsample: If True, upsample masks to original image resolution.

    Returns:
        Processed binary masks with shape `(N, img_h, img_w)`.
    """
    c, mh, mw = protos.shape  # CHW (32, 160, 160)
    ih, iw = shape

    # Step 1: Matrix Multiplication
    # (N, 32) @ (32, 25600) -> (N, 25600) -> (N, 160, 160)
    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    
    # Step 2: Sigmoid Activation
    masks = sigmoid(masks)

    # Step 3: Scale boxes to mask resolution (160x160) for cropping
    # Note: Model input is 640, Mask is 160, so scale factor is 0.25
    downsampled_bboxes = bboxes * (mh / 640.0) 
    
    # Step 4: Crop masks (remove artifacts outside the box)
    masks = crop_mask(masks, downsampled_bboxes)

    # Step 5: Resize to Original Image Size
    if upsample:
        # Optimized resize loop
        resized_masks = []
        for m in masks:
            m_res = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(m_res)
        masks = np.array(resized_masks)
    
    # Step 6: Binary Threshold (0.5)
    return masks > 0.5


def decode_seg_layer(box_feat: np.ndarray,
                     cls_feat: np.ndarray,
                     mc_feat: np.ndarray,
                     stride: int,
                     score_thres: float,
                     classes_num: int = 80) -> np.ndarray:
    """Decode a single feature layer for Segmentation.

    This function decodes the raw output tensors (Box, Class, Mask Coefficients) 
    of one detection layer using Anchor-Free logic. It includes an optimization 
    to filter out background anchors using raw logits.

    Args:
        box_feat: Raw bounding box output tensor `(1, H, W, 4)`.
            Contains distal distances [l, t, r, b].
        cls_feat: Raw classification output tensor `(1, H, W, 80)`.
            Contains class logits.
        mc_feat: Raw mask coefficient output tensor `(1, H, W, 32)`.
            Contains coefficients for linear combination with proto maps.
        stride: Stride of the feature layer relative to input image.
        score_thres: Confidence threshold for pre-filtering.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 38)` containing decoded predictions, 
        formatted as `[x1, y1, x2, y2, score, cls, mc0...mc31]`.
    """
    # Remove batch dimension if present
    if box_feat.shape[0] == 1: box_feat = box_feat[0]
    if cls_feat.shape[0] == 1: cls_feat = cls_feat[0]
    if mc_feat.shape[0] == 1: mc_feat = mc_feat[0]

    h, w, _ = box_feat.shape

    # -----------------------------------------------------------
    # Optimization: Filter using Raw Logits
    # -----------------------------------------------------------
    
    # Calculate logit threshold
    safe_thres = np.clip(score_thres, 1e-6, 1.0 - 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)
    
    max_logits = np.max(cls_feat, axis=-1)
    mask = max_logits >= logit_thres
    
    if not np.any(mask):
        return np.empty((0, 6 + 32), dtype=np.float32)

    # -----------------------------------------------------------
    # Decode Valid Candidates
    # -----------------------------------------------------------

    # Create coordinate grid
    grid_y, grid_x = np.indices((h, w))
    valid_grid_x = grid_x[mask]
    valid_grid_y = grid_y[mask]
    
    valid_box = box_feat[mask]
    valid_mc = mc_feat[mask]
    valid_cls_logits = cls_feat[mask]

    # Compute Scores and Class IDs
    valid_cls_scores = sigmoid(valid_cls_logits)
    valid_score = np.max(valid_cls_scores, axis=-1)
    valid_cls_id = np.argmax(valid_cls_scores, axis=-1)

    # Decode Anchor-Free Box (Distal-to-Center)
    # box_feat contains [l, t, r, b] distances from the grid center
    grid_center_x = valid_grid_x.astype(np.float32) + 0.5
    grid_center_y = valid_grid_y.astype(np.float32) + 0.5
    
    x1 = (grid_center_x - valid_box[:, 0]) * stride
    y1 = (grid_center_y - valid_box[:, 1]) * stride
    x2 = (grid_center_x + valid_box[:, 2]) * stride
    y2 = (grid_center_y + valid_box[:, 3]) * stride

    # Stack results: [x1, y1, x2, y2, score, cls, mc...]
    out = np.stack([x1, y1, x2, y2, valid_score, valid_cls_id], axis=-1)
    out = np.concatenate([out, valid_mc], axis=-1)
    
    return out


@dataclass
class YOLO26SegConfig:
    """Configuration for initializing the YOLO26 Segmentation model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO26 pipeline.

    Attributes:
        model_path: Path to the compiled YOLO26 `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding (Letterbox).
        strides: Feature map strides for each detection scale.
            Defaults to `[8, 16, 32]`.
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
    """YOLO26 Instance Segmentation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLO26 Segmentation,
    handling Input (NV12), BPU Inference, and Postprocessing (including 
    Box detection and Mask generation).
    """

    def __init__(self, config: YOLO26SegConfig):
        """Initialize the YOLO26 Segmentation model with the given configuration.

        Args:
            config: Configuration object containing model path, preprocessing
                parameters, and postprocessing parameters.
        """
        t0 = time.time()
        self.cfg = config
        
        # Load Model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        logger.info(f"\033[1;31m[Seg] Load Model time = {1000 * (time.time() - t0):.2f} ms\033[0m")

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        
        # Infer Input Size (Assuming NHWC layout)
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

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

    def pre_process(self, img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is resized according to the specified resize strategy
        and converted from BGR format to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            resize_type: Resize strategy override.
            image_format: Input image format. Currently, only `"BGR"` is
                supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
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
        
        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
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
        """Convert raw model outputs into final segmentation results.

        This step includes decoding predictions, applying Non-Maximum Suppression (NMS)
        on bounding boxes, and generating instance masks via matrix multiplication
        with prototype maps.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold for NMS.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes `(N, 4)` in original image coordinates.
                - score: Confidence scores `(N,)`.
                - cls: Class indices `(N,)`.
                - masks: Binary segmentation masks `(N, H, W)`.
        """
        t0 = time.time()
        score_thres = score_thres or self.cfg.score_thres
        nms_thres = nms_thres or self.cfg.nms_thres
        
        raw_outputs = outputs[self.model_name]
        decoded = []

        # Step 1: Decode Detection Layers (Strides 8, 16, 32)
        # Assuming fixed order from model logs:
        # Stride 8: [0]Box, [1]Cls, [2]MaskCoef
        # Stride 16:[3]Box, [4]Cls, [5]MaskCoef
        # Stride 32:[6]Box, [7]Cls, [8]MaskCoef
        
        for i, stride in enumerate(self.cfg.strides):
            base_idx = i * 3
            box_feat = raw_outputs[self.output_names[base_idx]]
            cls_feat = raw_outputs[self.output_names[base_idx + 1]]
            mc_feat = raw_outputs[self.output_names[base_idx + 2]]
            
            layer_pred = decode_seg_layer(box_feat, cls_feat, mc_feat, 
                                          stride, score_thres, self.cfg.classes_num)
            decoded.append(layer_pred)

        # Step 2: Extract Proto Layer (Index 9)
        # Shape: (1, 160, 160, 32) -> Transpose to (32, 160, 160) for MatMul
        proto_tensor = raw_outputs[self.output_names[9]]
        if proto_tensor.shape[0] == 1: 
            proto_tensor = proto_tensor[0]
        # HWC (160,160,32) -> CHW (32,160,160)
        proto_tensor = np.transpose(proto_tensor, (2, 0, 1))

        if not decoded:
             return np.array([]), np.array([]), np.array([]), np.array([])
        
        pred = np.concatenate(decoded, axis=0)
        if pred.shape[0] == 0:
             return np.array([]), np.array([]), np.array([]), np.array([])

        # Unpack predictions: [x1, y1, x2, y2, score, cls, mc0...mc31]
        xyxy = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5]
        mask_coefs = pred[:, 6:]

        # Step 3: Non-Maximum Suppression (NMS) on Boxes
        keep = post_utils.NMS(xyxy, score, cls, nms_thres)

        if not keep:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        xyxy = xyxy[keep]
        score = score[keep]
        cls = cls[keep]
        mask_coefs = mask_coefs[keep]

        # Step 4: Generate Masks from Coefficients and Protos
        # Note: We need boxes in model input scale (640x640) for mask cropping
        masks = process_mask(proto_tensor, mask_coefs, xyxy, 
                             (ori_img_h, ori_img_w), upsample=True)

        # Step 5: Rescale Boxes to Original Image Dimensions
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
        """Run the complete segmentation pipeline on a single image.

        This method internally performs preprocessing, inference, and
        postprocessing.

        Args:
            img: Input image array.
            image_format: Input image format. Currently supports `"BGR"`.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes `(N, 4)`.
                - score: Confidence scores `(N,)`.
                - cls: Class indices `(N,)`.
                - masks: Segmentation masks `(N, H, W)`.
        """
        # Original image size
        ori_img_h, ori_img_w = img.shape[:2]
        
        # 1) Preprocess
        inp = self.pre_process(img, resize_type, image_format)
        
        # 2) Inference
        out = self.forward(inp)
        
        # 3) Postprocess
        return self.post_process(out, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the segmentation pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array.
            image_format: Input image format.
            resize_type: Resize strategy override.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format, resize_type, score_thres, nms_thres)