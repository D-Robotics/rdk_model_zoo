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

import cv2
import numpy as np
from hbm_runtime import QuantParams
from scipy.special import softmax


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    @brief Compute the sigmoid activation function.
    @param x Input NumPy array.
    @return NumPy array after applying sigmoid function element-wise.
    """
    return 1.0 / (1.0 + cv2.exp(-x))


def recover_to_original_size(img: np.ndarray,
                             orig_w: int,
                             orig_h: int,
                             resize_type: int = 1) -> np.ndarray:
    """
    @brief Restore resized image back to original size.
    @details Supports direct resize or reverse letterbox removal.
    @param img Input image of shape (H, W, C).
    @param orig_w Original image width.
    @param orig_h Original image height.
    @param resize_type Resize type used before: 0 (direct) or 1 (letterbox).
    @return Resized image of shape (orig_h, orig_w, C).
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


# def print_topk_predictions(output: np.ndarray,
#                            idx2label: dict,
#                            topk: int = 5) -> None:
#     """
#     @brief Print top-k classification predictions.
#     @details Uses softmax to compute probability and selects top-k.
#     @param output Raw logits as NumPy array (shape: [num_classes]).
#     @param idx2label Dictionary mapping class indices to labels.
#     @param topk Number of top predictions to display.
#     @return None
#     """
#     # Softmax with stability adjustment
#     exp_logits = np.exp(output - np.max(output))
#     probabilities = exp_logits / np.sum(exp_logits)

#     # Top-k indices
#     topk_idx = np.argsort(probabilities)[-topk:][::-1]
#     topk_prob = probabilities[topk_idx]

#     print(f"Top-{topk} Predictions:")
#     for i in range(topk):
#         idx = topk_idx[i]
#         prob = topk_prob[i]
#         label = idx2label[idx] if idx2label and idx in idx2label else f"Class {idx}"
#         print(f"{label}: {prob:.4f}")

def print_topk_predictions(output: np.ndarray,
                           idx2label: dict,
                           topk: int = 5) -> None:
    """
    @brief Print top-k classification predictions.
    @details Uses softmax to compute probability and selects top-k.
    @param output Raw logits as NumPy array (shape: [num_classes]).
    @param idx2label Dictionary mapping class indices to labels.
    @param topk Number of top predictions to display.
    @return None
    """
    # Softmax with stability adjustment
    exp_logits = np.exp(output - np.max(output))
    probabilities = exp_logits / np.sum(exp_logits)

    # 确保 probabilities 是1维数组
    probabilities = probabilities.flatten()

    # Top-k indices
    topk_idx = np.argsort(probabilities)[-topk:][::-1]
    topk_prob = probabilities[topk_idx]

    print(f"Top-{topk} Predictions:")
    for i in range(topk):
        idx = topk_idx[i]
        prob = topk_prob[i]
        label = idx2label[idx] if idx2label and idx in idx2label else f"Class {idx}"
        print(f"{label}: {prob:.4f}")


def dequantize_tensor(q_tensor: np.ndarray, quant_info: QuantParams) -> np.ndarray:
    """
    @brief Dequantize a quantized tensor to floating-point values.
    @details Supports both per-tensor and per-channel dequantization based on quant_info.
    @param q_tensor Quantized tensor (e.g., int8 or uint8).
    @param quant_info Quantization parameters (scale, zero_point, axis, type).
    @return Dequantized tensor (float32).
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
    """
    @brief Dequantize a dictionary of quantized model outputs.
    @param outputs Dictionary of quantized output tensors.
    @param quan_infos Dictionary of quantization parameters per output.
    @return Dictionary of dequantized float32 outputs.
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
    """
    @brief Map coordinates from resized image back to original image scale.
    @param xyxy Bounding boxes (N, 4) in resized image.
    @param img_w Original image width.
    @param img_h Original image height.
    @param input_w Network input width.
    @param input_h Network input height.
    @param resize_type Resize strategy: 0 (resize), 1 (letterbox).
    @return Bounding boxes rescaled to original image dimensions.
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
    """
    @brief Perform class-wise Non-Maximum Suppression (NMS).
    @details Keeps boxes with highest scores and removes overlaps above IoU threshold.
    @param xyxy Bounding boxes (N, 4).
    @param score Confidence scores (N,).
    @param cls Class IDs for each box (N,).
    @param iou_thresh IoU threshold for suppression.
    @return List of indices to keep.
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
    """
    @brief Convert bounding boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2).
    @param xywh (N, 4) array in [center_x, center_y, width, height] format.
    @return (N, 4) array in [x1, y1, x2, y2] format.
    """
    x1y1 = xywh[:, :2] - xywh[:, 2:] / 2
    x2y2 = xywh[:, :2] + xywh[:, 2:] / 2
    return np.hstack([x1y1, x2y2])


def filter_classification(cls_output: np.ndarray, conf_thres_raw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Filter classification outputs using raw confidence threshold.
    @param cls_output Classification logits of shape (N, C).
    @param conf_thres_raw Threshold applied to max logit (before sigmoid).
    @return Tuple of:
        - scores: Sigmoid confidence scores of selected predictions
        - ids: Class indices of selected predictions
        - valid_indices: Original indices of selected predictions
    """
    cls_output = cls_output.reshape(-1, cls_output.shape[-1])
    max_scores = np.max(cls_output, axis=1)
    valid_indices = np.flatnonzero(max_scores >= conf_thres_raw)
    ids = np.argmax(cls_output[valid_indices], axis=1)
    # Apply sigmoid
    scores = 1 / (1 + np.exp(-max_scores[valid_indices]))
    return scores, ids, valid_indices


def filter_mces(mces_output: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    """
    @brief Extract MCES features from selected predictions.
    @param mces_output.
    @param valid_indices Indices of valid predictions.
    @return Filtered MCES tensor of shape (K, D), K = len(valid_indices).
    """
    mces_output = mces_output.reshape(-1, mces_output.shape[-1])
    mces = mces_output[valid_indices, :]
    return mces


def filter_predictions(pred: np.ndarray, score_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Filter detection predictions by confidence threshold.
    @param pred Tensor of shape (N, 5 + C): [x, y, w, h, obj_conf, class_probs...].
    @param score_thres Threshold on (obj_conf * class_conf).
    @return Tuple of:
        - xyxy: Filtered bounding boxes (Nf, 4)
        - score: Filtered scores (Nf,)
        - cls: Class indices (Nf,)
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
    """
    @brief Generate anchor center positions on a square grid.
    @param grid_size Size of the square grid (e.g., 80 for 80x80).
    @return (N, 2) array of anchor coordinates [x, y].
    """
    x = np.tile(np.linspace(0.5, grid_size - 0.5, grid_size), reps=grid_size)
    y = np.repeat(np.linspace(0.5, grid_size - 0.5, grid_size), grid_size)
    return np.stack([x, y], axis=1)


def decode_boxes(boxes_output: np.ndarray,
                 valid_indices: np.ndarray,
                 grid_size: int,
                 stride: int,
                 weights_static: np.ndarray) -> np.ndarray:
    """
    @brief Decode bounding boxes from distributional predictions.
    @param boxes_output Tensor of shape (N, 4 * 16).
    @param valid_indices Indices of valid predictions.
    @param grid_size Feature map grid size.
    @param stride Downsampling factor.
    @param weights_static Discrete location weights (e.g., 0~15).
    @return Decoded bounding boxes in xyxy format (M, 4).
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
    """
    @brief Decode instance segmentation masks.
    @param mces Mask coefficients for each detection (M, C).
    @param boxes Bounding boxes (M, 4).
    @param protos Mask prototype feature map (H, W, C).
    @param input_w Width of the input image.
    @param input_h Height of the input image.
    @param mask_w Width of the mask proto.
    @param mask_h Height of the mask proto.
    @param mask_thresh Threshold to binarize masks.
    @return List of (H, W) binary mask arrays.
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
    """
    @brief Decode keypoint coordinates from model output.
    @param kpts_output Keypoint tensor of shape (N, 17*3).
    @param valid_indices Indices of valid predictions.
    @param grid_size Size of feature map grid.
    @param stride Downsampling factor (e.g., 8, 16, 32).
    @param anchor Optional anchor points. If None, generated automatically.
    @return Tuple:
            - kpts_xy: (M, 17, 2) pixel coordinates of keypoints.
            - kpts_score: (M, 17, 1) keypoint confidence scores.
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


def get_bounding_boxes(dilated_polys: list[np.ndarray], min_area: float) -> list[np.ndarray]:
    """
    @brief Extract minimum area bounding boxes from polygon contours.
    @param dilated_polys List of polygon contours. Each element is a NumPy array of shape (N, 1, 2).
    @param min_area Minimum area threshold to filter small boxes.
    @return List of bounding boxes. Each is a NumPy array of shape (4, 2), type int.
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
    """
    @brief Resize binary masks to fit inside their corresponding bounding boxes.
    @param masks List of binary mask arrays of shape (H, W), dtype=uint8.
    @param boxes List of bounding boxes in (x1, y1, x2, y2) format.
    @param img_w Width of the original image.
    @param img_h Height of the original image.
    @param interpolation OpenCV interpolation method used for resizing.
    @param do_morph Whether to apply morphological open to smooth the mask.
    @return List of resized binary masks cropped to box size.
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
                                      boxes: list[tuple[float, float, float, float]],
                                      img_w: int, img_h: int,
                                      input_w: int, input_h: int,
                                      resize_type: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Scale keypoints back to original image coordinates.
    @param kpts_xy Keypoint coordinates of shape (M, 17, 2), float32.
    @param kpts_score Keypoint scores of shape (M, 17, 1), float32.
    @param boxes List of bounding boxes, not used here.
    @param img_w Width of the original image.
    @param img_h Height of the original image.
    @param input_w Width of model input.
    @param input_h Height of model input.
    @param resize_type 0 = direct resize, 1 = letterbox resize.
    @return Tuple of (scaled keypoints, scores), both NumPy arrays.
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
    """
    @brief Crop and rotate a region from the image using a rotated bounding box.
    @param img Input image array of shape (H, W, C), dtype=uint8.
    @param box Bounding box as 4-point array of shape (4, 2).
    @return Cropped and rotated region image as a NumPy array.
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
