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

"""Provide a YOLO DFL-based detection inference wrapper and pipeline utilities.

This module defines a lightweight YOLO detection runtime wrapper built on
HBM runtime. It supports DFL-based (Distribution Focal Loss) detection models
including YOLOv5u, YOLOv8, YOLO11, and YOLO12, which share the same
anchor-free box decoding logic.

Key Features:
    - YoloDetectConfig dataclass for configuring model parameters.
    - YoloDetect class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Anchor-free box decoding via DFL regression outputs.
    - Class-wise Non-Maximum Suppression (NMS).

Typical Usage:
    >>> from yolo_detect import YoloDetect, YoloDetectConfig
    >>> cfg = YoloDetectConfig(model_path="/path/to/yolo11n_detect.hbm")
    >>> model = YoloDetect(cfg)
    >>> boxes, scores, cls_ids = model(img)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - The detection head uses anchor-free, DFL regression. Each detection
      scale emits a paired classification output and a box distribution
      output (4 * reg values per prediction).
"""

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YoloDetectConfig:
    """Configuration for initializing the YoloDetect model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO detection
    pipeline. It applies to all DFL-based YOLO detection models (v5u, v8,
    v11, and v12).

    Attributes:
        model_path: Path to the compiled YOLO detection `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80 (COCO).
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes (in pixels) for each detection scale.
    """
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = field(default_factory=lambda: [80, 40, 20])


class YoloDetect:
    """YOLO DFL-based detection wrapper for HB_HBMRuntime.

    This class provides a unified inference pipeline for DFL-based YOLO
    detection models (v5u, v8, v11, and v12), including input preprocessing,
    model execution, and postprocessing steps such as anchor-free DFL box
    decoding, confidence filtering, and Non-Maximum Suppression (NMS).

    Attributes:
        model: Loaded HBM runtime model instance.
        model_name: Name of the first loaded model.
        input_names: Input tensor name list.
        output_names: Output tensor name list.
        input_shapes: Input tensor shape dictionary.
        input_h: Model input height (pixels).
        input_w: Model input width (pixels).
        weights_static: DFL discrete location weights for box expectation.
        cfg: Model configuration object.

    Notes:
        All supported YOLO detection variants (v5u, v8, v11, and v12) share the
        same anchor-free DFL head structure. Each detection scale emits a
        paired classification output and a box distribution output.
    """

    def __init__(self, config: YoloDetectConfig):
        """Initialize the YoloDetect model with the given configuration.

        Args:
            config: Configuration object containing model path, preprocessing
                parameters, and postprocessing parameters. All field semantics
                and constraints are defined in the `YoloDetectConfig` dataclass.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # DFL weights: shape (1, 1, reg), used to compute expected box offsets
        self.weights_static = np.arange(config.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

        # Store configuration
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

        The input image is resized according to the configured resize strategy
        and converted from BGR format to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            image_format: Input image format. Currently, only `"BGR"` is
                supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`.

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
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`.

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
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into final detection results.

        This step includes anchor-free DFL box decoding,
        confidence filtering, Non-Maximum Suppression (NMS), and coordinate
        scaling back to the original image resolution.

        Args:
            outputs: Raw output tensors from inference (as returned by `forward()`).
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override. If `None`, the value
                from the configuration is used.
            nms_thres: IoU threshold for NMS override. If `None`, the value
                from the configuration is used.

        Returns:
            A tuple containing:
                - boxes: Bounding boxes with shape `(N, 4)` in original image
                  coordinates, formatted as `[x1, y1, x2, y2]`.
                - scores: Confidence scores with shape `(N,)`.
                - cls_ids: Class indices with shape `(N,)`.
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        # Compute inverse-sigmoid threshold for raw logit filtering
        conf_thres_raw = -np.log(1.0 / score_thres - 1)

        # Step 1: Decode each detection scale's paired classification and box outputs
        model_outputs = outputs[self.model_name]
        all_boxes = []
        all_scores = []
        all_ids = []
        for i, (stride, anchor_size) in enumerate(
                zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key = self.output_names[2 * i]      # Classification logits output
            box_key = self.output_names[2 * i + 1]  # DFL box distribution output

            # Filter by raw logit threshold before sigmoid
            scores, ids, valid_indices = post_utils.filter_classification(
                model_outputs[cls_key], conf_thres_raw)

            # Decode DFL bounding boxes for valid predictions
            dbboxes = post_utils.decode_boxes(
                model_outputs[box_key], valid_indices,
                anchor_size, stride, self.weights_static)

            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        # Step 2: Concatenate results across all detection scales
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_ids, axis=0)

        # Step 3: Non-Maximum Suppression
        keep = post_utils.NMS(boxes, scores, cls_ids, nms_thres)

        # Step 4: Rescale boxes to original image dimensions
        xyxy = post_utils.scale_coords_back(
            boxes[keep], ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)

        return xyxy, scores[keep], cls_ids[keep]

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete detection pipeline on a single image.

        This method internally performs preprocessing, inference, and
        postprocessing.

        Args:
            img: Input image array.
            image_format: Input image format. Currently supports `"BGR"`.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            A tuple containing:
                - boxes: Bounding boxes with shape `(N, 4)`.
                - scores: Confidence scores with shape `(N,)`.
                - cls_ids: Class indices with shape `(N,)`.
        """
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) Preprocess
        input_tensor = self.pre_process(img, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        boxes, scores, cls_ids = self.post_process(
            outputs, ori_img_w, ori_img_h, score_thres, nms_thres)

        return boxes, scores, cls_ids

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the detection pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            img: Input image array.
            image_format: Input image format.
            score_thres: Confidence threshold override.
            nms_thres: IoU threshold override for NMS.

        Returns:
            Same return values as `predict()`.
        """
        return self.predict(img, image_format, score_thres, nms_thres)
