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

"""Provide a YOLOe11-Seg open-vocabulary instance segmentation inference wrapper and pipeline utilities.

This module defines a lightweight YOLOe11-Seg runtime wrapper built on HBM runtime.
It includes configuration definitions and a complete instance segmentation pipeline
(preprocess, forward, postprocess), producing bounding boxes, class IDs, scores,
and per-instance binary masks for an open-vocabulary setting with 4585 classes.

Key Features:
    - YoloE11SegConfig dataclass for configuring model parameters.
    - YoloE11Seg class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Anchor-free DFL box decoding and MCES mask coefficient decoding.
    - Prototype-based mask generation with optional morphological post-processing.

Typical Usage:
    >>> from yoloe11seg import YoloE11Seg, YoloE11SegConfig
    >>> cfg = YoloE11SegConfig(model_path="/path/to/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm")
    >>> model = YoloE11Seg(cfg)
    >>> boxes, scores, cls_ids, masks = model(img)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - The model outputs 10 tensors: 3 scales × (cls, box, mces) + protos.
    - classes_num is 4585 for the open-vocabulary model.
    - S600 platform is NOT supported; a runtime check is performed in main.py.
"""

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess.py
#   utils/py_utils/postprocess.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YoloE11SegConfig:
    """Configuration for initializing the YoloE11Seg open-vocabulary model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLOe11-Seg pipeline.

    Attributes:
        model_path: Path to the compiled YOLOe11-Seg `.hbm` model.
        classes_num: Number of detection classes. Defaults to 4585 (open-vocabulary).
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        mces_num: Dimension of the MCES (mask coefficient) vector. Defaults to 32.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
        do_morph: Whether to apply morphological opening to clean mask edges.
            Defaults to False for the open-vocabulary model to preserve fine details.
    """
    model_path: str
    classes_num: int = 4585
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.7
    reg: int = 16
    mces_num: int = 32
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = field(default_factory=lambda: [80, 40, 20])
    do_morph: bool = False


class YoloE11Seg:
    """YOLOe11 open-vocabulary instance segmentation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLOe11-Seg, including
    input preprocessing, model execution, and postprocessing steps such as
    anchor-free DFL box decoding, MCES mask coefficient extraction,
    Non-Maximum Suppression (NMS), prototype-based mask generation, and
    mask resizing to bounding box coordinates.

    Attributes:
        model: Loaded HBM runtime model instance.
        model_name: Name of the first loaded model.
        input_names: Input tensor name list.
        output_names: Output tensor name list.
        input_shapes: Input tensor shape dictionary.
        output_quants: Output quantization parameter dictionary.
        input_h: Model input height (pixels).
        input_w: Model input width (pixels).
        weights_static: DFL bin weights for box expectation computation.
        cfg: Model configuration object.

    Notes:
        The model outputs 10 tensors: for 3 detection scales (stride 8/16/32),
        each has [cls, box, mces] outputs, plus a final protos tensor at index 9.
        Box regression uses DFL (16-bin distributions per side).
        Mask generation uses linear combination of 32 prototype features with
        per-instance MCES coefficients.
        classes_num is 4585 for this open-vocabulary model.
        S600 platform is NOT supported.
    """

    def __init__(self, config: YoloE11SegConfig):
        """Initialize the YoloE11Seg model with the given configuration.

        Args:
            config: Configuration object containing model path and all inference
                parameters. All field semantics are defined in `YoloE11SegConfig`.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution (H, W)
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # DFL bin weights: shape (1, 1, reg) for expectation computation
        self.weights_static = np.arange(config.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

        self.cfg = config

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

    def pre_process(self,
                    img: np.ndarray,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is letterbox-resized and converted from BGR to NV12
        (Y and UV planes).

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently only `"BGR"` is supported.

        Returns:
            A nested input tensor dictionary: `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(
                img, self.input_w, self.input_h, self.cfg.resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by `pre_process()`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Convert raw model outputs into instance segmentation results.

        Steps:
        1) Dequantize all output tensors.
        2) For each detection scale, filter by logit threshold and decode DFL
           boxes and MCES mask coefficients.
        3) Concatenate results, apply NMS.
        4) Decode per-instance binary masks from prototype features and MCES.
        5) Rescale boxes to original image dimensions.
        6) Resize masks to their bounding boxes.

        Args:
            outputs: Raw output tensors from inference (as returned by `forward()`).
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override. If `None`, uses config value.
            nms_thres: IoU threshold for NMS override. If `None`, uses config value.

        Returns:
            A tuple containing:
                - boxes: Bounding boxes with shape `(N, 4)`, format `[x1, y1, x2, y2]`.
                - scores: Confidence scores with shape `(N,)`.
                - cls_ids: Class indices with shape `(N,)`.
                - masks: List of N binary mask arrays, each sized to its bounding box.
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        # Inverse sigmoid threshold for raw logit filtering
        conf_thres_raw = -np.log(1.0 / score_thres - 1.0)

        # Step 1: Dequantize all outputs
        fp32_outputs = post_utils.dequantize_outputs(
            outputs[self.model_name], self.output_quants)

        # Step 2: Extract prototype features (index 9), shape (160, 160, 32)
        protos_float32 = fp32_outputs[self.output_names[9]][0]
        mask_h, mask_w = protos_float32.shape[:2]

        all_boxes = []
        all_scores = []
        all_ids = []
        all_mces = []

        # Step 3: Decode each detection head (cls + box + mces)
        for i, (stride, anchor_size) in enumerate(zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key  = self.output_names[3 * i]
            box_key  = self.output_names[3 * i + 1]
            mces_key = self.output_names[3 * i + 2]

            # Filter by raw logit threshold, decode DFL boxes
            scores, ids, valid_indices = post_utils.filter_classification(
                fp32_outputs[cls_key], conf_thres_raw)
            dbboxes = post_utils.decode_boxes(
                fp32_outputs[box_key], valid_indices,
                anchor_size, stride, self.weights_static)
            mces = post_utils.filter_mces(fp32_outputs[mces_key], valid_indices)

            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_mces.append(mces)

        # Concatenate across all scales
        boxes  = np.concatenate(all_boxes,  axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids    = np.concatenate(all_ids,    axis=0)
        mces   = np.concatenate(all_mces,   axis=0)

        # Step 4: NMS
        keep = post_utils.NMS(boxes, scores, ids, nms_thres)

        # Step 5: Decode instance masks from protos and MCES coefficients
        masks = post_utils.decode_masks(
            mces[keep], boxes[keep], protos_float32,
            self.input_w, self.input_h, mask_w, mask_h,
            mask_thresh=0.5
        )

        # Step 6: Rescale boxes to original image
        xyxy = post_utils.scale_coords_back(
            boxes[keep], ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)

        # Step 7: Resize each mask to its box in original image coordinates
        resized_masks = post_utils.resize_masks_to_boxes(
            masks, xyxy, ori_img_w, ori_img_h, do_morph=self.cfg.do_morph)

        return xyxy, scores[keep], ids[keep], resized_masks

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Run the complete instance segmentation pipeline on a single image.

        This method internally performs preprocessing, inference, and postprocessing.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format. Currently supports `"BGR"`.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            A tuple containing:
                - boxes: Bounding boxes with shape `(N, 4)`.
                - scores: Confidence scores with shape `(N,)`.
                - cls_ids: Class indices with shape `(N,)`.
                - masks: List of N per-instance binary mask arrays.
        """
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) Preprocess
        input_tensor = self.pre_process(img, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        return self.post_process(outputs, ori_img_w, ori_img_h, score_thres, nms_thres)

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Callable interface for the instance segmentation pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array in BGR format.
            image_format: Input image format.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format, score_thres, nms_thres)
