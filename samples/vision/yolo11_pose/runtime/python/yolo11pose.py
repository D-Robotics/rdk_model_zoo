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

"""Provide a YOLO11 pose estimation inference wrapper and pipeline utilities.

This module defines a lightweight YOLO11-Pose runtime wrapper built on HBM runtime.
It includes configuration definitions and a complete pose estimation pipeline
(preprocess, forward, postprocess), producing bounding boxes, class IDs, scores,
keypoint coordinates and keypoint scores.

Key Features:
    - YoloV11PoseConfig dataclass for configuring model parameters.
    - YoloV11Pose class providing pre_process, forward, post_process, predict,
      and __call__ methods.
    - Anchor-free DFL box decoding and keypoint coordinate decoding.
    - Class-wise NMS with keypoints kept aligned.

Typical Usage:
    >>> from yolo11pose import YoloV11Pose, YoloV11PoseConfig
    >>> cfg = YoloV11PoseConfig(model_path="/path/to/yolo11n_pose.hbm")
    >>> model = YoloV11Pose(cfg)
    >>> boxes, scores, cls_ids, kpts_xy, kpts_score = model(img)

Notes:
    - Requires hbm_runtime to be installed in the deployment environment.
    - Input images are expected in BGR format by default.
    - The model outputs 9 tensors: 3 scales × (cls, box, kpts).
    - Keypoint scores are returned as raw logits (not sigmoid-activated);
      visualize.draw_keypoints handles the threshold comparison internally.
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
class YoloV11PoseConfig:
    """Configuration for initializing the YoloV11Pose model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO11-Pose pipeline.

    Attributes:
        model_path: Path to the compiled YOLO11-Pose `.hbm` model.
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
    """
    model_path: str
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.7
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = field(default_factory=lambda: [80, 40, 20])


class YoloV11Pose:
    """YOLO11 pose estimation wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLO11-Pose, including
    input preprocessing, model execution, and postprocessing steps such as
    anchor-free DFL box decoding, keypoint coordinate decoding,
    Non-Maximum Suppression (NMS), and coordinate rescaling to original image space.

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
        The model outputs 9 tensors: for 3 detection scales (stride 8/16/32),
        each has [cls, box, kpts] outputs.
        Box regression uses DFL (16-bin distributions per side).
        Keypoints use 17 COCO keypoints, each with (x, y, score).
    """

    def __init__(self, config: YoloV11PoseConfig):
        """Initialize the YoloV11Pose model with the given configuration.

        Args:
            config: Configuration object containing model path and all inference
                parameters. All field semantics are defined in `YoloV11PoseConfig`.
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
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into pose estimation results.

        Steps:
        1) Dequantize all output tensors.
        2) For each detection scale, filter by logit threshold and decode DFL
           boxes and keypoint coordinates.
        3) Concatenate results, apply NMS.
        4) Rescale boxes and keypoints to original image dimensions.

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
                - kpts_xy: Keypoint coordinates with shape `(N, 17, 2)`.
                - kpts_score: Keypoint scores (raw logits) with shape `(N, 17, 1)`.
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        # Inverse sigmoid threshold for raw logit filtering
        conf_thres_raw = -np.log(1.0 / score_thres - 1.0)

        # Step 1: Dequantize all outputs
        fp32_outputs = post_utils.dequantize_outputs(
            outputs[self.model_name], self.output_quants)

        all_boxes = []
        all_scores = []
        all_ids = []
        all_kpts_xy = []
        all_kpts_score = []

        # Step 2: Decode each detection head (cls + box + kpts)
        for i, (stride, anchor_size) in enumerate(zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key  = self.output_names[3 * i]
            box_key  = self.output_names[3 * i + 1]
            kpts_key = self.output_names[3 * i + 2]

            # Filter by raw logit threshold, decode DFL boxes
            scores, ids, valid_indices = post_utils.filter_classification(
                fp32_outputs[cls_key], conf_thres_raw)
            dbboxes = post_utils.decode_boxes(
                fp32_outputs[box_key], valid_indices,
                anchor_size, stride, self.weights_static)

            # Decode keypoints (x, y, score) for valid detections
            anchor = post_utils.gen_anchor(anchor_size)[valid_indices]
            kpts_xy, kpts_score = post_utils.decode_kpts(
                fp32_outputs[kpts_key], valid_indices,
                anchor_size, stride, anchor)

            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_kpts_xy.append(kpts_xy)
            all_kpts_score.append(kpts_score)

        # Concatenate across all scales
        boxes      = np.concatenate(all_boxes,      axis=0)
        scores     = np.concatenate(all_scores,     axis=0)
        ids        = np.concatenate(all_ids,        axis=0)
        kpts_xy    = np.concatenate(all_kpts_xy,    axis=0)
        kpts_score = np.concatenate(all_kpts_score, axis=0)

        # Step 3: NMS
        keep = post_utils.NMS(boxes, scores, ids, nms_thres)

        # Step 4: Rescale boxes to original image
        xyxy = post_utils.scale_coords_back(
            boxes[keep], ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)

        # Step 5: Rescale keypoints to original image
        scaled_kpts_xy, scaled_kpts_score = post_utils.scale_keypoints_to_original_image(
            kpts_xy[keep], kpts_score[keep],
            ori_img_w, ori_img_h,
            self.input_w, self.input_h,
            self.cfg.resize_type)

        return xyxy, scores[keep], ids[keep], scaled_kpts_xy, scaled_kpts_score

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete pose estimation pipeline on a single image.

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
                - kpts_xy: Keypoint coordinates with shape `(N, 17, 2)`.
                - kpts_score: Keypoint scores (raw logits) with shape `(N, 17, 1)`.
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
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the pose estimation pipeline.

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
