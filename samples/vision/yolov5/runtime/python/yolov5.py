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

"""Provide a YOLOv5X inference wrapper and pipeline utilities.

This module defines a lightweight YOLOv5X runtime wrapper built on HBM runtime.
It includes configuration definitions and a complete inference pipeline
(preprocess, forward, postprocess), along with minimal decoding helpers.
"""

import os
import cv2
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Literal

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess_utils.py
#   utils/py_utils/postprocess_utils.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function.

    Applies the sigmoid function element-wise to the input NumPy array.

    Args:
        x: Input NumPy array.

    Returns:
        A NumPy array with the sigmoid function applied element-wise.
    """
    return 1.0 / (1.0 + cv2.exp(-x))


def decode_layer(feat: np.ndarray,
                 stride: int,
                 anchor: np.ndarray,
                 classes_num: int = 80) -> np.ndarray:
    """Decode a single feature layer from the detection head.

    This function decodes the raw output tensor of one detection layer
    (corresponding to a specific stride) into bounding box predictions
    in the original image scale.

    The decoded output includes:
        - Bounding box center coordinates (x, y)
        - Bounding box width and height (w, h)
        - Objectness score
        - Per-class confidence scores

    Args:
        feat: Raw model output tensor with shape
            `(1, na, h, w, 5 + num_classes)`, where `na` is the number of anchors.
        stride: Stride of the feature layer relative to the input image.
        anchor: Anchor sizes for this feature layer with shape `(na, 2)`,
            formatted as `(width, height)`.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 5 + num_classes)` containing decoded
        predictions, where `N = na * h * w`.
    """
    _, _, h, w, _ = feat.shape  #  h/w: feature map size

    # Create coordinate grid of shape (1, 1, h, w, 2)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid = np.stack((grid_x, grid_y), axis=-1)[None, None]

    # batch sigmoid
    feat_sig = sigmoid(feat[..., :5 + classes_num])

    # Decode center offsets (dx, dy) and size (dw, dh)
    dxdy = feat_sig[..., :2]
    dwdh = feat_sig[..., 2:4]
    obj  = feat_sig[..., 4:5]
    cls  = feat_sig[..., 5:]

    # Compute center coordinates in original image scale
    xy = (dxdy * 2. - 0.5 + grid) * stride

    # Compute width/height from anchor sizes
    wh = (dwdh * 2.) ** 2 * anchor[:, None, None, :]

    # Construct final output tensor (xywh + obj + class scores)
    out = np.empty((*xy.shape[:-1], 5 + classes_num), dtype=np.float32)
    out[..., 0:2] = xy
    out[..., 2:4] = wh
    out[..., 4:5] = obj
    out[..., 5:]  = cls

    return out.reshape(-1, 5 + classes_num)


def decode_outputs(output_names: list[str],
                   fp32_outputs: dict[str, np.ndarray],
                   strides: list[int],
                   anchors: list[np.ndarray],
                   classes_num: int = 80) -> np.ndarray:
    """Decode all feature maps from the model output.

    This function iterates over all detection heads, reshapes and reorders
    the raw output tensors, decodes each feature map using its corresponding
    stride and anchor configuration, and concatenates the results into a
    single prediction tensor.

    Args:
        output_names: List of output tensor names corresponding to detection heads.
        fp32_outputs: Dictionary mapping output tensor names to FP32 NumPy arrays
            produced by the model.
        strides: List of stride values for each detection head.
        anchors: List of anchor arrays for each detection head, where each element
            has shape `(na, 2)`.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 5 + classes_num)` containing all decoded
        predictions, where `N` is the total number of predictions across
        all detection heads.
    """
    decoded = []
    for i, key in enumerate(output_names):
        out = fp32_outputs[key]
        h, w = out.shape[1:3]
        # Reshape and transpose to (1, na, h, w, c)
        feat = out.reshape(1, h, w, 3, 5 + classes_num).transpose(0, 3, 1, 2, 4)
        decoded.append(decode_layer(feat, strides[i], anchors[i], classes_num))
    return np.concatenate(decoded, axis=0)


@dataclass
class YOLOv5Config:
    """Configuration for initializing the YoloV5X model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLOv5 pipeline.

    Attributes:
        model_path: Path to the compiled YOLOv5 `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80.
        resize_type: Image resize mode used during preprocessing.
            - 0: Stretch resize.
            - 1: Keep aspect ratio with padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        strides: Feature map strides for each detection scale.
            Defaults to `[8, 16, 32]`.
        anchors: Anchor box definitions for each detection scale with shape
            `(3, 3, 2)`.
    """
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    # Feature map downsampling strides
    strides: np.ndarray = field(
        default_factory=lambda: np.array([8, 16, 32], dtype=np.int32)
    )
    # Anchors for each scale
    anchors: np.ndarray = field(
        default_factory=lambda: np.array([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ], dtype=np.float32).reshape(3, 3, 2)
    )


class YoloV5X:
    """YOLOv5X object detection wrapper based on HB_HBMRuntime.

    This class provides a unified inference pipeline for YOLOv5X, including
    input preprocessing, model execution, and postprocessing steps such as
    decoding, confidence filtering, and Non-Maximum Suppression (NMS).
    """

    def __init__(self, config: YOLOv5Config):
        """Initialize the YOLOv5X model with the given configuration.

        Args:
            config: Configuration object containing model path, preprocessing
                parameters, and postprocessing parameters. All field semantics
                and constraints are defined in the `YOLOv5Config` dataclass.
        """

        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution (H, W) inferred from model input tensor
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # Detection and preprocessing configuration
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

    def pre_process(self, img: np.ndarray,
                    resize_type: Optional[int] = None,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an input image into model-required tensor format.

        The input image is resized according to the specified resize strategy
        and converted from BGR format to NV12 (Y and UV planes).

        Args:
            img: Input image array.
            resize_type: Resize strategy override. If `None`, the value from
                the configuration is used. If provided, the configuration
                value will be updated.
            image_format: Input image format. Currently, only `"BGR"` is
                supported.

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`.

        Raises:
            ValueError: If an unsupported image format is provided.
        """
        if resize_type == None:
            resize_type = self.cfg.resize_type
        else:
            self.cfg.resize_type = resize_type

        # Resize and convert to NV12
        if(image_format == "BGR"):
            resize_img = pre_utils.resized_image(img, self.input_w, self.input_h, resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else :
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

        This step includes dequantization, decoding, confidence filtering,
        Non-Maximum Suppression (NMS), and coordinate scaling back to the
        original image resolution.

        Args:
            outputs: Raw output tensors from inference.
            ori_img_w: Width of the original input image.
            ori_img_h: Height of the original input image.
            score_thres: Confidence threshold override. If `None`, the value
                from the configuration is used.
            nms_thres: IoU threshold for NMS. If `None`, the value from the
                configuration is used.

        Returns:
            A tuple containing:
                - xyxy: Bounding boxes with shape `(N, 4)` in original image
                  coordinates.
                - score: Confidence scores with shape `(N,)`.
                - cls: Class indices with shape `(N,)`.
        """
        score_thres = score_thres or self.cfg.score_thres
        nms_thres = nms_thres or self.cfg.nms_thres

        # Step 1: Convert quantized outputs to float32
        fp32_outputs = post_utils.dequantize_outputs(outputs[self.model_name], self.output_quants)

        # Step 2: Decode YOLO outputs into unified predictions
        pred = decode_outputs(self.output_names, fp32_outputs,
                              self.cfg.strides, self.cfg.anchors, self.cfg.classes_num)

        # Step 3: Filter predictions by confidence threshold
        xyxy_boxes, score, cls = post_utils.filter_predictions(pred, score_thres)

        # Step 4: Non-Maximum Suppression (NMS)
        keep = post_utils.NMS(xyxy_boxes, score, cls, nms_thres)

        # Step 5: Rescale boxes to original image dimensions
        xyxy = post_utils.scale_coords_back(xyxy_boxes[keep], ori_img_w, ori_img_h,
                                            self.input_w, self.input_h, self.cfg.resize_type)

        return xyxy, score[keep], cls[keep]

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                resize_type: Optional[int] = None,
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the complete detection pipeline on a single image.

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
                - xyxy: Bounding boxes with shape `(N, 4)`.
                - score: Confidence scores with shape `(N,)`.
                - cls: Class indices with shape `(N,)`.
        """
        # Original image size
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) Preprocess
        input_tensor = self.pre_process(img, resize_type, image_format)

        # 2) Inference
        outputs = self.forward(input_tensor)

        # 3) Postprocess
        xyxy, score, cls = self.post_process(outputs, ori_img_w, ori_img_h,
                                             score_thres, nms_thres)

        return xyxy, score, cls

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 resize_type: Optional[int] = None,
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for the detection pipeline.

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
