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
    """
    @brief Compute the sigmoid activation function.
    @param x Input NumPy array.
    @return NumPy array after applying sigmoid function element-wise.
    """
    return 1.0 / (1.0 + cv2.exp(-x))


def decode_layer(feat: np.ndarray,
                 stride: int,
                 anchor: np.ndarray,
                 classes_num: int = 80) -> np.ndarray:
    """
    @brief Decode a single feature layer from detection head.
    @param feat Raw model output tensor of shape (1, na, h, w, c).
    @param stride Stride of the feature layer.
    @param anchor Anchor sizes for this layer (na, 2).
    @param classes_num Number of output classes.
    @return Decoded prediction array of shape (N, 5 + num_classes).
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
    """
    @brief Decode all feature maps from model output.
    @param output_names List of output tensor names.
    @param fp32_outputs Dict of decoded tensors from model.
    @param strides Stride values for each output head.
    @param anchors Anchor arrays for each head.
    @param classes_num Number of output classes.
    @return Concatenated prediction tensor of shape (N, 5 + classes).
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
    """
    @brief Configuration for initializing YoloV5X model.

    Contains model path and all runtime parameters required for preprocessing,
    inference, and postprocessing.

    Fields:
        model_path (str):
            Path to the compiled YOLOv5 .hbm model.

        classes_num (int):
            Number of detection categories. Default: 80.

        resize_type (int):
            Image resize mode used during preprocessing.
            0 = stretch resize
            1 = keep aspect ratio with padding

        score_thres (float):
            Minimum confidence threshold for filtering candidates.

        nms_thres (float):
            IoU threshold used for Non-Maximum Suppression.

        strides (np.ndarray):
            Feature map strides, default [8,16,32].

        anchors (np.ndarray):
            YOLO anchor definitions with shape (3, 3, 2).
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
    """
    @brief YOLOv5X object detection wrapper using HB_HBMRuntime.

    Provides a unified inference pipeline including input preprocessing,
    model execution, and postprocessing (decode, confidence filtering,
    and Non-Maximum Suppression).
    """

    def __init__(self, config: YOLOv5Config):
        """
        @brief Initialize the YOLOv5X model with a provided configuration.

        @param config (YOLOv5Config)
            Configuration object containing model path, preprocessing,
            and postprocessing parameters. All field descriptions and
            constraints are defined within the YOLOv5Config dataclass.
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
        """
        @brief Configure inference scheduling parameters.

        @param priority (int, optional)
            Inference priority in range [0, 255].
        @param bpu_cores (list[int], optional)
            BPU core indices used for inference.
        @return None
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
        @brief Preprocess input image into model-required tensor format.

        The input image is resized according to resize_type and converted
        from BGR to NV12 (Y + UV planes).

        @param img (np.ndarray)
            Input image array.
        @param resize_type (int, optional)
            Resize strategy. If None, uses the value from configuration.
            If provided, the configuration value will be updated.
        @param image_format (str, optional)
            Input image format. Currently only "BGR" is supported.
        @return dict
            Nested input tensor dictionary:
            {model_name: {input_name: tensor}}
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
        """
        @brief Execute model inference.

        @param input_tensor (dict)
            Preprocessed input tensor dictionary produced by pre_process().
        @return dict
            Raw output tensors returned by the runtime.
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
        """
        @brief Convert raw model outputs into final detection results.

        This includes dequantization, decoding, confidence filtering,
        Non-Maximum Suppression (NMS), and coordinate scaling.

        @param outputs (dict)
            Raw output tensors from inference.
        @param ori_img_w (int)
            Width of the original input image.
        @param ori_img_h (int)
            Height of the original input image.
        @param score_thres (float, optional)
            Confidence threshold. If None, uses configuration value.
        @param nms_thres (float, optional)
            NMS IoU threshold. If None, uses configuration value.
        @return Tuple:
            - xyxy (np.ndarray): Bounding boxes (N, 4) in original image coordinates.
            - cls (np.ndarray): Class indices (N).
            - score (np.ndarray): Confidence scores (N).
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
        """
        @brief Run a complete detection pipeline on a single image.

        This method internally performs preprocessing, inference,
        and postprocessing.

        @param img (np.ndarray)
            Input image array.
        @param image_format (str)
            Input image format. Currently supports "BGR".
        @param resize_type (int, optional)
            Resize strategy override.
        @param score_thres (float, optional)
            Confidence threshold override.
        @param nms_thres (float, optional)
            NMS IoU threshold override.
        @return Tuple:
            - xyxy (np.ndarray): Bounding boxes (N, 4).
            - cls (np.ndarray): Class indices (N).
            - score (np.ndarray): Confidence scores (N).
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
        """
        @brief Callable interface for the detection pipeline.

        Equivalent to calling predict().

        @param img (np.ndarray)
            Input image.
        @param image_format (str)
            Input image format.
        @return Same as predict().
        """
        return self.predict(img, image_format, resize_type, score_thres, nms_thres)
