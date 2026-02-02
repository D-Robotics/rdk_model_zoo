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

"""
postprocess: Postprocessing utilities for vision model outputs.

This module provides reusable postprocessing helpers to convert raw model
outputs into task-level results, including coordinate scaling, quantization
recovery, prediction filtering, and optional decoding for different output
types. It is designed to be shared across multiple samples and runtimes.

Key Features:
    - Recover/rescale results back to the original image space.
    - Dequantize and decode raw outputs into usable representations.
    - Apply common filtering and suppression strategies (e.g., NMS).
    - Provide optional utilities for masks, keypoints, and geometric handling.

Notes:
    - The module focuses on generic postprocessing building blocks; task- or
      model-specific policies should be implemented at the sample level.
    - Output formats and helper coverage may evolve as new models are added.
"""


import cv2
import numpy as np
from hbm_runtime import QuantParams
from scipy.special import softmax


def recover_to_original_size(img: np.ndarray,
                             orig_w: int,
                             orig_h: int,
                             resize_type: int = 1) -> np.ndarray:
    """Restore a resized image back to its original size.

    This function reverses the resizing operation applied during preprocessing.
    It supports both direct resizing and letterbox-based resizing with padding
    removal.

    Args:
        img: Input image array with shape `(H, W, C)` after preprocessing.
        orig_w: Original image width.
        orig_h: Original image height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding.

    Returns:
        The image resized back to shape `(orig_h, orig_w, C)`.

    Raises:
        ValueError: If an invalid `resize_type` is provided.
    """
    h, w = img.shape[:2]  # current size after preprocess

    if resize_type == 0:
        # Resize directly to original dimensions
        img_resized = cv2.resize(img, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
    elif resize_type == 1:
        # Remove padding and resize back from letterbox
        scale = min(h / orig_h, w / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        pad_w = w - new_w
        pad_h = h - new_h
        left = pad_w // 2
        top = pad_h // 2

        # Crop out the letterbox padding
        cropped = img[top:top + new_h, left:left + new_w]

        # Resize cropped region to original size
        img_resized = cv2.resize(cropped, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

    return img_resized


def dequantize_tensor(q_tensor: np.ndarray, quant_info: QuantParams) -> np.ndarray:
    """Dequantize a quantized tensor to floating-point values.

    This function converts a quantized tensor (e.g., int8 or uint8) into
    floating-point values using the provided quantization parameters.
    Both per-tensor and per-channel dequantization are supported.

    Args:
        q_tensor: Quantized input tensor.
        quant_info: Quantization parameters including scale, zero point,
            quantization axis, and quantization type.

    Returns:
        A float32 NumPy array containing the dequantized tensor values.
    """
    if quant_info.quant_type != 1:  # 1 indicates linear scale quantization
        return q_tensor

    if quant_info.scale.ndim == 0 or q_tensor.ndim == 1 or quant_info.scale.size == 1:
        # Per-tensor dequantization
        return (q_tensor.astype(np.float32) - quant_info.zero_point.astype(np.float32)) * quant_info.scale
    else:
        # Per-channel dequantization
        shape = [1] * q_tensor.ndim
        shape[quant_info.axis] = -1
        scale = quant_info.scale.reshape(shape)
        zero_point = quant_info.zero_point.reshape(shape)
        return (q_tensor.astype(np.float32) - zero_point.astype(np.float32)) * scale


def dequantize_outputs(outputs: dict, quan_infos: dict) -> dict:
    """Dequantize a dictionary of quantized model outputs.

    This function applies tensor dequantization to each model output using
    its corresponding quantization parameters and returns the results as
    floating-point tensors.

    Args:
        outputs: Dictionary mapping output tensor names to quantized tensors.
        quan_infos: Dictionary mapping output tensor names to their
            corresponding quantization parameters.

    Returns:
        A dictionary mapping output tensor names to dequantized float32
        NumPy arrays.
    """
    fp32_outputs = {}
    for name, output in outputs.items():
        quant_info = quan_infos[name]
        fp32_outputs[name] = dequantize_tensor(output, quant_info)
    return fp32_outputs


def scale_coords_back(xyxy: np.ndarray,
                      img_w: int,
                      img_h: int,
                      input_w: int,
                      input_h: int,
                      resize_type: int = 1) -> np.ndarray:
    """Map bounding box coordinates back to the original image scale.

    This function converts bounding box coordinates from the resized
    (model input) image space back to the original image resolution.
    Both direct resize and letterbox resize strategies are supported.

    Args:
        xyxy: Bounding boxes with shape `(N, 4)` in the resized image space,
            formatted as `(xmin, ymin, xmax, ymax)`.
        img_w: Original image width.
        img_h: Original image height.
        input_w: Network input width.
        input_h: Network input height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding.

    Returns:
        Bounding boxes rescaled to the original image dimensions with
        shape `(N, 4)`.

    Raises:
        ValueError: If an invalid `resize_type` is provided.
    """
    if resize_type == 0:
        # Direct resize
        scale_x = img_w / input_w
        scale_y = img_h / input_h
        xyxy[:, [0, 2]] *= scale_x
        xyxy[:, [1, 3]] *= scale_y
    elif resize_type == 1:
        # Letterbox resize
        scale = min(input_w / img_w, input_h / img_h)
        pad_w = (input_w - img_w * scale) / 2
        pad_h = (input_h - img_h * scale) / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / scale
    else:
        raise ValueError("resize_type must be 0 (resize) or 1 (letterbox)")

    # Clamp coordinates within valid image bounds
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, img_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, img_h)

    return xyxy


def NMS(xyxy: np.ndarray,
        score: np.ndarray,
        cls: np.ndarray,
        iou_thresh: float = 0.45) -> list:
    """Perform class-wise Non-Maximum Suppression (NMS).

    This function applies Non-Maximum Suppression independently for each
    class. For each class, bounding boxes are sorted by confidence score,
    and boxes with an Intersection over Union (IoU) greater than the given
    threshold are suppressed.

    Args:
        xyxy: Bounding boxes with shape `(N, 4)`, formatted as
            `(xmin, ymin, xmax, ymax)`.
        score: Confidence scores for each bounding box with shape `(N,)`.
        cls: Class IDs for each bounding box with shape `(N,)`.
        iou_thresh: IoU threshold used to suppress overlapping boxes.

    Returns:
        A list of indices corresponding to the bounding boxes that are kept
        after Non-Maximum Suppression.
    """
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        x1, y1, x2, y2 = xyxy[idx].T
        area = (x2 - x1) * (y2 - y1)
        order = score[idx].argsort()[::-1]  # Sort by descending score

        while order.size > 0:
            i = order[0]
            keep.append(idx[i])
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)

            # Keep boxes with IoU below threshold
            order = order[1:][iou < iou_thresh]

    return keep


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center format to corner format.

    This function converts bounding boxes from
    `(center_x, center_y, width, height)` format to
    `(x1, y1, x2, y2)` format, where `(x1, y1)` is the top-left corner
    and `(x2, y2)` is the bottom-right corner.

    Args:
        xywh: Bounding boxes with shape `(N, 4)` in
            `(center_x, center_y, width, height)` format.

    Returns:
        Bounding boxes with shape `(N, 4)` in `(x1, y1, x2, y2)` format.
    """
    x1y1 = xywh[:, :2] - xywh[:, 2:] / 2
    x2y2 = xywh[:, :2] + xywh[:, 2:] / 2
    return np.hstack([x1y1, x2y2])


def filter_classification(cls_output: np.ndarray, conf_thres_raw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter classification outputs using a raw confidence threshold.

    This function selects classification predictions whose maximum logit
    value exceeds a given threshold, then computes sigmoid confidence
    scores for the selected predictions.

    Args:
        cls_output: Classification logits with shape `(N, C)`, where `N`
            is the number of predictions and `C` is the number of classes.
        conf_thres_raw: Threshold applied to the maximum logit value
            (before sigmoid).

    Returns:
        A tuple containing:
            - scores: Sigmoid confidence scores of the selected predictions.
            - ids: Class indices of the selected predictions.
            - valid_indices: Indices of the selected predictions in the
              original input array.
    """
    cls_output = cls_output.reshape(-1, cls_output.shape[-1])
    max_scores = np.max(cls_output, axis=1)
    valid_indices = np.flatnonzero(max_scores >= conf_thres_raw)
    ids = np.argmax(cls_output[valid_indices], axis=1)
    # Apply sigmoid
    scores = 1 / (1 + np.exp(-max_scores[valid_indices]))
    return scores, ids, valid_indices


def filter_mces(mces_output: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    """Extract MCES features for selected predictions.

    This function selects MCES feature vectors corresponding to the given
    valid prediction indices.

    Args:
        mces_output: MCES output tensor, where the last dimension represents
            the feature dimension.
        valid_indices: Indices of valid predictions to be selected.

    Returns:
        A NumPy array of shape `(K, D)` containing the filtered MCES features,
        where `K = len(valid_indices)` and `D` is the feature dimension.
    """
    mces_output = mces_output.reshape(-1, mces_output.shape[-1])
    mces = mces_output[valid_indices, :]
    return mces


def filter_predictions(pred: np.ndarray, score_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter detection predictions by confidence threshold.

    This function combines objectness scores and class probabilities to
    compute final confidence scores, then filters out predictions whose
    confidence is below the given threshold.

    Args:
        pred: Prediction tensor with shape `(N, 5 + C)`, formatted as
            `[x, y, w, h, objectness, class_probs...]`.
        score_thres: Confidence threshold applied to
            `(objectness * class_probability)`.

    Returns:
        A tuple containing:
            - xyxy: Filtered bounding boxes with shape `(Nf, 4)` in
              `(x1, y1, x2, y2)` format.
            - score: Confidence scores of the filtered predictions with
              shape `(Nf,)`.
            - cls: Class indices of the filtered predictions with
              shape `(Nf,)`.
    """
    xywh = pred[:, :4]

    # Combine objectness and class scores
    conf_all = pred[:, 4:5] * pred[:, 5:]
    cls = conf_all.argmax(axis=1)
    score = conf_all[np.arange(len(pred)), cls]
    mask = score > score_thres
    xyxy = xywh_to_xyxy(xywh[mask])
    return xyxy, score[mask], cls[mask]


def gen_anchor(grid_size: int) -> np.ndarray:
    """Generate anchor center positions on a square grid.

    This function generates evenly spaced anchor center coordinates for a
    square grid of size `grid_size x grid_size`. The centers are placed at
    half-integer positions (e.g., 0.5, 1.5, ...) along both axes.

    Args:
        grid_size: Size of the square grid (e.g., `80` for an `80 x 80` grid).

    Returns:
        A NumPy array of shape `(N, 2)` containing anchor center coordinates
        in `(x, y)` format, where `N = grid_size * grid_size`.
    """
    x = np.tile(np.linspace(0.5, grid_size - 0.5, grid_size), reps=grid_size)
    y = np.repeat(np.linspace(0.5, grid_size - 0.5, grid_size), grid_size)
    return np.stack([x, y], axis=1)


def decode_boxes(boxes_output: np.ndarray,
                 valid_indices: np.ndarray,
                 grid_size: int,
                 stride: int,
                 weights_static: np.ndarray) -> np.ndarray:
    """Decode bounding boxes from distributional predictions.

    This function decodes bounding box coordinates from distribution-based
    regression outputs (e.g., 16-bin discrete distributions per side).
    It applies softmax to each distribution, computes the expected offsets,
    and converts them into bounding boxes in `(x1, y1, x2, y2)` format.

    Args:
        boxes_output: Bounding box output tensor with shape `(N, 4 * 16)`,
            representing discrete distributions for left, top, right, and
            bottom offsets.
        valid_indices: Indices of valid predictions to be decoded.
        grid_size: Feature map grid size (e.g., width or height of the grid).
        stride: Downsampling factor used to map grid coordinates to the
            original image scale.
        weights_static: Discrete location weights used to compute expectations
            (e.g., values from `0` to `15`).

    Returns:
        A NumPy array of shape `(M, 4)` containing decoded bounding boxes in
        `(x1, y1, x2, y2)` format, where `M = len(valid_indices)`.
    """
    bboxes = boxes_output.reshape(-1, boxes_output.shape[-1])
    bboxes_float32 = bboxes[valid_indices]
    # Softmax over 16 bins per LTRB side and apply expectation
    ltrb = np.sum(softmax(bboxes_float32.reshape(-1, 4, 16), axis=2) *
                  weights_static, axis=2)
    anchor = gen_anchor(grid_size)[valid_indices]
    x1y1 = anchor - ltrb[:, 0:2]
    x2y2 = anchor + ltrb[:, 2:4]
    return np.hstack([x1y1, x2y2]) * stride


def decode_masks(mces: np.ndarray,
                 boxes: np.ndarray,
                 protos: np.ndarray,
                 input_w: int,
                 input_h: int,
                 mask_w: int,
                 mask_h: int,
                 mask_thresh: float = 0.5) -> list[np.ndarray]:
    """Decode instance segmentation masks.

    This function generates instance-level binary masks by linearly combining
    mask prototype features with per-instance mask coefficients (MCES).
    The resulting masks are cropped according to the corresponding bounding
    boxes and binarized using a threshold.

    Args:
        mces: Mask coefficients for each detection with shape `(M, C)`.
        boxes: Bounding boxes with shape `(M, 4)` in `(x1, y1, x2, y2)` format.
        protos: Mask prototype feature map with shape `(H, W, C)`.
        input_w: Width of the input image.
        input_h: Height of the input image.
        mask_w: Width of the mask prototype feature map.
        mask_h: Height of the mask prototype feature map.
        mask_thresh: Threshold used to binarize the decoded masks.

    Returns:
        A list of binary mask arrays, where each element has shape `(H_i, W_i)`
        corresponding to one detected instance.
    """
    masks = []
    x_scale = mask_w / input_w
    y_scale = mask_h / input_h

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Crop proto features using scaled coordinates
        x1_corp = int(x1 * x_scale)
        y1_corp = int(y1 * y_scale)
        x2_corp = int(x2 * x_scale)
        y2_corp = int(y2 * y_scale)

        proto_crop = protos[y1_corp:y2_corp, x1_corp:x2_corp, :]  # (H, W, C)
        mc = mces[i]
        # Linear combination and thresholding
        mask = (np.sum(proto_crop * mc[np.newaxis, np.newaxis, :], axis=2)
                > mask_thresh).astype(np.uint8)
        masks.append(mask)

    return masks


def decode_kpts(kpts_output: np.ndarray,
                valid_indices: np.ndarray,
                grid_size: int,
                stride: int,
                anchor: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode keypoint coordinates from model output.

    This function decodes keypoint predictions from the model output tensor
    into pixel coordinates using anchor points and the given stride. Each
    keypoint consists of `(x, y, score)` values.

    Args:
        kpts_output: Keypoint output tensor with shape `(N, 17 * 3)`, where
            each keypoint is represented by `(x, y, score)`.
        valid_indices: Indices of valid predictions to be decoded.
        grid_size: Size of the feature map grid.
        stride: Downsampling factor used to map grid coordinates to the
            original image scale (e.g., 8, 16, 32).
        anchor: Optional anchor center coordinates with shape `(M, 2)`.
            If `None`, anchor points are generated automatically based on
            `grid_size`.

    Returns:
        A tuple containing:
            - kpts_xy: Keypoint pixel coordinates with shape `(M, 17, 2)`.
            - kpts_score: Keypoint confidence scores with shape `(M, 17, 1)`.
    """
    kpts_output = kpts_output.reshape(-1, kpts_output.shape[-1])
    kpts = kpts_output[valid_indices].reshape(-1, 17, 3)  # (M, 17, 3)

    if anchor is None:
        anchor = gen_anchor(grid_size)[valid_indices]  # (M, 2)

    # Decode x, y using anchor and stride
    kpts_xy = (kpts[:, :, :2] * 2.0 + (anchor[:, None, :] - 0.5)) * stride

    # Extract score without activation (or apply sigmoid optionally)
    kpts_score = kpts[:, :, 2:3]

    return kpts_xy, kpts_score


def decode_layer(feat: np.ndarray,
                 stride: int,
                 anchor: np.ndarray,
                 classes_num: int = 80) -> np.ndarray:
    """Decode a single feature layer from the detection head.

    This function decodes the raw output tensor of one detection layer into
    bounding box predictions in the original image scale. The decoded output
    includes bounding box center coordinates, width and height, objectness
    score, and per-class confidence scores.

    Args:
        feat: Raw model output tensor with shape
            `(1, na, h, w, 5 + classes_num)`, where `na` is the number of anchors.
        stride: Stride of the feature layer relative to the input image.
        anchor: Anchor sizes for this feature layer with shape `(na, 2)`,
            formatted as `(width, height)`.
        classes_num: Number of object classes.

    Returns:
        A NumPy array of shape `(N, 5 + classes_num)` containing decoded
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


def get_bounding_boxes(dilated_polys: list[np.ndarray], min_area: float) -> list[np.ndarray]:
    """Extract minimum-area bounding boxes from polygon contours.

    This function computes the minimum-area bounding rectangle for each
    polygon contour and filters out boxes whose area is smaller than the
    specified threshold.

    Args:
        dilated_polys: List of polygon contours. Each element is a NumPy array
            with shape `(N, 1, 2)`.
        min_area: Minimum area threshold used to filter out small bounding boxes.

    Returns:
        A list of bounding boxes. Each bounding box is a NumPy array with
        shape `(4, 2)` and integer coordinates.
    """
    boxes_list = []
    for cnt in dilated_polys:
        if cv2.contourArea(cnt) < min_area:
            continue  # Skip small contours
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int_)
        boxes_list.append(box)
    return boxes_list


def resize_masks_to_boxes(masks: list[np.ndarray],
                          boxes: list[tuple[float, float, float, float]],
                          img_w: int, img_h: int,
                          interpolation: int = cv2.INTER_LANCZOS4,
                          do_morph: bool = True) -> list[np.ndarray]:
    """Resize binary masks to fit inside their corresponding bounding boxes.

    This function resizes each binary mask to match the size of its
    corresponding bounding box. Optionally, a morphological open operation
    can be applied to smooth the resized masks.

    Args:
        masks: List of binary mask arrays with shape `(H, W)` and dtype
            `uint8`.
        boxes: List of bounding boxes in `(x1, y1, x2, y2)` format.
        img_w: Width of the original image.
        img_h: Height of the original image.
        interpolation: OpenCV interpolation method used for resizing.
        do_morph: Whether to apply a morphological open operation to smooth
            the resized masks.

    Returns:
        A list of resized binary masks, each cropped to the size of its
        corresponding bounding box.
    """
    resized_masks = []
    for mask, (x1, y1, x2, y2) in zip(masks, boxes):
        # Clamp coordinates to image bounds
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), img_w), min(int(y2), img_h)

        target_w = max(x2 - x1, 1)
        target_h = max(y2 - y1, 1)

        resized = cv2.resize(mask, (target_w, target_h), interpolation=interpolation)

        if do_morph:
            # Apply morphological filtering
            resized = cv2.morphologyEx(resized, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        resized_masks.append(resized)

    return resized_masks


def scale_keypoints_to_original_image(kpts_xy: np.ndarray,
                                      kpts_score: np.ndarray,
                                      img_w: int, img_h: int,
                                      input_w: int, input_h: int,
                                      resize_type: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Scale keypoints back to the original image coordinates.

    This function maps keypoint coordinates from the model input space back
    to the original image resolution. Both direct resize and letterbox resize
    strategies are supported.

    Args:
        kpts_xy: Keypoint coordinates with shape `(M, 17, 2)` in the model
            input space.
        kpts_score: Keypoint confidence scores with shape `(M, 17, 1)`.
        img_w: Width of the original image.
        img_h: Height of the original image.
        input_w: Width of the model input.
        input_h: Height of the model input.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding.

    Returns:
        A tuple containing:
            - scaled_kpts: Keypoints scaled to the original image coordinates
              with shape `(M, 17, 2)`.
            - kpts_score: Keypoint confidence scores with shape `(M, 17, 1)`.
    """
    scaled_kpts = kpts_xy.copy()

    if resize_type == 0:
        scale_x = img_w / input_w
        scale_y = img_h / input_h
        scaled_kpts[..., 0] *= scale_x
        scaled_kpts[..., 1] *= scale_y

    elif resize_type == 1:
        scale = min(input_w / img_w, input_h / img_h)
        pad_w = (input_w - img_w * scale) / 2
        pad_h = (input_h - img_h * scale) / 2
        scaled_kpts[..., 0] = (scaled_kpts[..., 0] - pad_w) / scale
        scaled_kpts[..., 1] = (scaled_kpts[..., 1] - pad_h) / scale

    else:
        raise ValueError("resize_type must be 0 or 1")

    # Clip to image bounds
    scaled_kpts[..., 0] = np.clip(scaled_kpts[..., 0], 0, img_w)
    scaled_kpts[..., 1] = np.clip(scaled_kpts[..., 1], 0, img_h)

    return scaled_kpts, kpts_score


def crop_and_rotate_image(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop and rotate a region from an image using a rotated bounding box.

    This function extracts a rotated rectangular region defined by a
    four-point bounding box. It applies a perspective transformation to
    obtain an upright cropped image and optionally rotates the result
    based on the bounding box angle.

    Args:
        img: Input image array with shape `(H, W, C)` and dtype `uint8`.
        box: Rotated bounding box represented as a four-point array with
            shape `(4, 2)`.

    Returns:
        A NumPy array representing the cropped and rotated image region.
    """
    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect).astype(np.intp)
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    # Rotate if angle is large
    if angle >= 45:
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = warped

    print("width:", rotated.shape[1], "height:", rotated.shape[0])
    return rotated
