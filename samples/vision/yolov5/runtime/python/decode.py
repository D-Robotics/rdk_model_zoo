# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOv5 source equations; X5 and S retain their different NMS/order semantics."""
import cv2
import numpy as np

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


def classwise_nms(xyxy: np.ndarray,
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



def decode_x5(outputs, names, size, anchors, score_thres, nms_thres):
    """Preserve X5 source candidate order, OpenCV XYXY-call convention and score gate.

    The source feeds XYXY to OpenCV NMSBoxes (whose Rect interface is XYWH).
    This legacy convention is intentionally unchanged for migration comparison;
    it is not equivalent to S class-wise XYXY NMS. See the evaluator limits.
    """
    all_boxes=[];all_scores=[];all_ids=[]
    for i,(name,stride) in enumerate(zip(names,(8,16,32))):
        h=size//stride
        grid=np.stack([np.tile(np.linspace(.5,h-.5,h),reps=h),np.repeat(np.arange(.5,h+.5,1),h)],axis=0).T
        grid=np.hstack([grid,grid,grid]).reshape(-1,2).astype(np.float32)
        tiled=np.tile(anchors[i],(h*h,1)).astype(np.float32)
        pred=outputs[name].reshape(-1,85)
        cls_max=np.max(pred[:,5:],axis=1)
        # X5 source uses NumPy exp; S decoder uses OpenCV exp.
        scores=(1.0/(1.0+np.exp(-pred[:,4])))*(1.0/(1.0+np.exp(-cls_max)))
        valid=np.flatnonzero(scores>=score_thres)
        if not valid.size:continue
        delta=1.0/(1.0+np.exp(-pred[valid,:4]))
        xy=(delta[:,:2]*2+grid[valid]-1)*stride
        wh=(delta[:,2:]*2)**2*tiled[valid]
        all_boxes.append(np.hstack([xy-wh*.5,xy+wh*.5]));all_scores.append(scores[valid]);all_ids.append(np.argmax(pred[valid,5:],axis=1))
    if not all_boxes:return np.empty((0,4),np.float32),np.empty(0,np.float32),np.empty(0,np.int32)
    boxes=np.concatenate(all_boxes).astype(np.float32);scores=np.concatenate(all_scores).astype(np.float32);ids=np.concatenate(all_ids).astype(np.int32)
    keep=np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),score_thres,nms_thres),dtype=int).reshape(-1)
    return boxes[keep],scores[keep],ids[keep]
