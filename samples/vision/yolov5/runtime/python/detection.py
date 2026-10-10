# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOv5's pure image → native tensors → owned detections pipeline.

``detection.py`` owns the model end to end: the published tensor contract
(including the S-series quantization snapshots), the lazy SDK runner, the
source decode math (X5 raw-F32 and S dequant anchor decoding with their
different NMS/order semantics) and the :class:`YOLOv5Task` stages
(``preprocess`` → ``infer`` → ``postprocess`` → ``predict``). Published
asset identities/selection and rendering live in ``cli.py``.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
from utils.py_utils.assets import list_assets
from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.platforms import resolve_target
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples.vision.yolov5.runtime.python.cli import (
    ANCHORS,
    STRIDES,
    ModelSelection,
    resolve_selection,
)

# ======================================================================
# Published tensor contract: exact identities, quantization snapshots and
# strict native tensor validation.
# ======================================================================


@dataclass(frozen=True)
class Quantization:
    quant_type: str
    scale: object
    zero_point: object
    axis: int


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    model_name: str
    input_names: tuple
    input_shapes: object
    output_names: tuple
    output_shapes: object
    output_dtypes: object
    output_quants: object
    input_size: int
    output_transform: str
    resize_type: int
    activation: str = 'sigmoid_logits'


def _quant_snapshot(info, shape, dtype):
    import numpy as np
    if info is None:raise MetadataMismatchError('S output requires its source quantization descriptor.')
    q=getattr(info,'quant_type',info);kind=str(getattr(q,'name',q))
    if kind not in ('SCALE','1','NONE','0'):raise MetadataMismatchError('Unsupported quantization descriptor kind.')
    if dtype!='float32' and kind not in ('SCALE','1'):raise MetadataMismatchError('Integer output requires SCALE quantization.')
    scale=tuple(float(x) for x in np.asarray(getattr(info,'scale',())).reshape(-1))
    zero=tuple(float(x) for x in np.asarray(getattr(info,'zero_point',())).reshape(-1))
    axis=int(getattr(info,'axis',0))
    if kind in ('SCALE','1'):
        if not scale or not all(math.isfinite(x) and x>0 for x in scale) or not all(math.isfinite(x) for x in zero):raise MetadataMismatchError('Invalid quantization scale/zero point.')
        if len(scale)>1 and (not -len(shape)<=axis<len(shape) or len(scale)!=shape[axis] or len(zero) not in (0,1,len(scale))):raise MetadataMismatchError('Quantization channels/axis mismatch.')
    def frozen_array(value):
        a=np.asarray(value)
        return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    return Quantization(kind,frozen_array(getattr(info,'scale',())),frozen_array(getattr(info,'zero_point',())),axis)


def bind_model(selection, metadata):
    """Validate actual native shapes/dtypes; never infer an HBM layout from a name."""
    check=resolve_selection(selection.target,variant=selection.variant,asset_id=selection.asset.reference,model_path=selection.model_path,consumer=selection.consumer)
    if check.asset!=selection.asset:raise ValueError('Manifest publication facts changed.')
    m=metadata if isinstance(metadata,RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if m.model_names!=(m.model_name,):raise MetadataMismatchError('YOLOv5 artifact must contain exactly one model.')
    size=640 if selection.target=='x5' else 672
    expected=((1,3,size,size),) if selection.target=='x5' else ((1,size,size,1),(1,size//2,size//2,2))
    if len(m.input_names)!=len(expected) or tuple(m.input_shapes.get(n) for n in m.input_names)!=expected:raise MetadataMismatchError('YOLOv5 logical input shapes/order mismatch.')
    if any(m.input_dtypes.get(n) not in ('uint8','nv12') for n in m.input_names):raise MetadataMismatchError('YOLOv5 inputs require uint8/NV12 metadata.')
    if len(set(m.output_names))!=3:raise MetadataMismatchError('YOLOv5 requires three distinct detection heads.')
    ordered=[];quants={};dtypes={}
    for stride in STRIDES:
        h=size//stride
        matches=[n for n in m.output_names if m.output_shapes.get(n) in ((1,h,h,255),(1,h,h,3,85))]
        if len(matches)!=1:raise MetadataMismatchError(f'No unique stride-{stride} 3-anchor/80-class head.')
        name=matches[0];ordered.append(name);dtype=m.output_dtypes.get(name)
        allowed=('float32',) if selection.target=='x5' else ('float32','int8','uint8','int16','int32')
        if dtype not in allowed:raise MetadataMismatchError(f'Unsupported native output dtype {dtype!r}.')
        dtypes[name]=dtype
        if selection.target!='x5':quants[name]=_quant_snapshot(m.output_quants.get(name),m.output_shapes[name],dtype)
    freeze=lambda x:MappingProxyType(dict(x))
    return ModelBinding(selection,m.model_name,m.input_names,freeze(m.input_shapes),tuple(ordered),freeze(m.output_shapes),freeze(dtypes),freeze(quants),size,'raw_f32' if selection.target=='x5' else 'dequant',0 if selection.target=='x5' else 1)


def validate_tensors(binding,tensors,*,outputs=False):
    """Validate flat tensors without casting, reshaping or replacing array values."""
    import numpy as np
    from collections.abc import Mapping
    names=binding.output_names if outputs else binding.input_names
    if not isinstance(tensors,Mapping) or set(tensors)!=set(names):raise MetadataMismatchError('Tensor names do not match binding.')
    for n in names:
        value=tensors[n]
        shape=binding.output_shapes[n] if outputs else ((binding.input_size**2*3//2,) if binding.selection.target=='x5' else binding.input_shapes[n])
        dtype=binding.output_dtypes[n] if outputs else 'uint8'
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.dtype(dtype):raise MetadataMismatchError(f'{n}: expected native {shape}/{dtype}.')
        if not np.isfinite(value).all():raise MetadataMismatchError(f'{n}: non-finite tensor.')
    return {n:tensors[n] for n in names}


# ======================================================================
# Lazy SDK adapter; both fixed source platforms use named containers.
# ======================================================================

class RuntimeModelRunner:
    """Bind once; validate input/output without decoding or numerical conversion.

    A supplied runtime_factory is a host-fixture seam. Normal execution verifies
    hardware identity and artifact bytes before importing the SDK. Not thread safe.
    """
    def __init__(self,selection,*,runtime_factory=None):
        self.selection=selection;self._factory=runtime_factory;self.runtime=None;self.binding=None

    def load(self):
        """Return the exact binding; mismatch fails before any inference."""
        if self.binding is not None:return self.binding
        if self._factory is None:
            from utils.py_utils.platforms import require_execution_target
            from utils.py_utils.assets import verify_asset_file
            require_execution_target(self.selection.target)
            verify_asset_file(self.selection.asset,self.selection.model_path)
            from utils.py_utils.model_runner import _default_runtime_factory
            factory=_default_runtime_factory()
        else:factory=self._factory
        runtime=factory(str(self.selection.model_path))
        binding=bind_model(self.selection,RuntimeMetadata.from_runtime(runtime))
        self.runtime=runtime;self.binding=binding
        return binding

    def __call__(self,tensors):
        """Return flat native outputs; both sources call run({model_name: inputs})."""
        binding=self.load();inputs=validate_tensors(binding,tensors)
        result=self.runtime.run({binding.model_name:inputs})
        if not isinstance(result,dict) or set(result)!={binding.model_name}:raise MetadataMismatchError('Unexpected runtime model container.')
        return validate_tensors(binding,result[binding.model_name],outputs=True)

    def set_scheduling_params(self,*,priority=0,bpu_cores=None):
        """Pass native named-model priority/core values; no installation or fallback."""
        if type(priority) is not int or not 0<=priority<=255:raise ValueError('priority must be integer 0..255.')
        if bpu_cores is not None and (not isinstance(bpu_cores,(tuple,list)) or not bpu_cores or any(type(v) is not int or v<0 for v in bpu_cores)):raise ValueError('bpu_cores must be nonempty nonnegative integer indexes.')
        binding=self.load();kwargs={'priority':{binding.model_name:priority}}
        if bpu_cores is not None:kwargs['bpu_cores']={binding.model_name:list(bpu_cores)}
        self.runtime.set_scheduling_params(**kwargs)


# ======================================================================
# Source decode equations; X5 and S retain their different NMS/order
# semantics.
# ======================================================================

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


# ======================================================================
# The readable task stages.
# ======================================================================


@dataclass(frozen=True)
class DetectionResult:
    """Owned detections in source order: F32 XYXY (N,4), F32 scores (N,), I32 IDs.

    Coordinates are clipped to original image bounds. X5 source coordinates are
    integer-truncated (stored here as F32); S preserves subpixel coordinates.
    Scores lie in [0,1]; class IDs in [0,79]. Empty outputs retain these shapes.
    """
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray


def _threshold(value, name):
    if isinstance(value,bool) or not np.isfinite(value) or not 0<=value<=1:raise ValueError(f'{name} must be finite in [0,1].')
    return float(value)


def resized_image(img: np.ndarray, input_W: int, input_H: int,
                  resize_type: int = 1,
                  interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """Resize an image using direct resize or letterbox strategy.

    This function resizes the input image to the target resolution required
    by the model. It supports either direct resizing or letterbox resizing
    with padding to preserve the aspect ratio.

    Args:
        img: Input image array with shape `(H, W, 3)`.
        input_W: Target image width.
        input_H: Target image height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding to preserve aspect ratio.
        interpolation: OpenCV interpolation method used for resizing.

    Returns:
        The resized image with shape `(input_H, input_W, 3)`.

    Raises:
        ValueError: If an invalid `resize_type` is provided.
    """
    img_h, img_w = img.shape[:2]

    if resize_type == 0:  # Direct resize
        resized = cv2.resize(img, (input_W, input_H), interpolation=interpolation)
    elif resize_type == 1:  # Letterbox resize (preserve aspect ratio)
        scale = min(input_H / img_h, input_W / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = input_W - new_w
        pad_h = input_H - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2

        # Pad image with gray (127,127,127)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(127, 127, 127))
    else:
        raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

    return resized


@dataclass(frozen=True)
class DetectionContext:
    """Original H/W, bound network size and source inverse-resize mode, per call."""
    original_size: tuple[int, int]
    model_size: int
    resize_type: int

    def __post_init__(self):
        if not isinstance(self.original_size,tuple) or len(self.original_size)!=2 or any(type(v) is not int or v<1 for v in self.original_size):
            raise ValueError('original_size must be a positive (height, width) tuple.')
        if type(self.model_size) is not int or self.model_size<2 or self.model_size%2:
            raise ValueError('model_size must be a positive even integer.')
        if type(self.resize_type) is not int or self.resize_type not in (0,1):
            raise ValueError('resize_type must be 0 or 1.')


@dataclass(frozen=True)
class PreparedInput:
    """Owned uint8 flat packed NV12 or Y/UV tensor map plus immutable geometry."""
    tensors: dict[str, np.ndarray]
    context: DetectionContext


def prepare_image(image, binding, resize_type=None):
    """Prepare positive HWC uint8 BGR input; mode 0 nearest, mode 1 source letterbox.

    X5 returns flat uint8 H*W*3/2; S returns uint8 Y (1,H,W,1) and UV
    (1,H/2,W/2,2). Letterbox uses linear resize, integer floor dimensions,
    symmetric 127 padding; inverse coordinates retain source ideal scale.
    Raises ValueError for invalid pixels/mode or degenerate resized geometry.
    """
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:
        raise ValueError('Expected nonempty HWC uint8 BGR image.')
    resize=binding.resize_type if resize_type is None else resize_type
    if type(resize) is not int or resize not in (0,1):raise ValueError('resize_type must be 0 or 1.')
    size=binding.input_size
    if resize==1 and min(int(n*min(size/image.shape[0],size/image.shape[1])) for n in image.shape[:2])<1:
        raise ValueError('Image aspect ratio produces an empty letterbox dimension.')
    pixels=resized_image(image,size,size,resize)
    y,uv=bgr_to_nv12_planes(pixels)
    names=binding.input_names
    tensors=({names[0]:np.concatenate((y.reshape(-1),uv.reshape(-1)))} if binding.selection.target=='x5' else {names[0]:y,names[1]:uv})
    return PreparedInput(tensors,DetectionContext(tuple(image.shape[:2]),size,resize))


class YOLOv5Task:
    """Run one published detector from its manifest selection.

    Construction with a ModelSelection loads the artifact through the shared
    SDK adapter: board identity and artifact bytes are verified before the
    SDK is imported, then the tensor contract is bound. Host fixtures inject
    a runner (plus its binding) or a runtime_factory instead of a selection.
    Instances carry fixed thresholds/anchors only; context is returned by
    pre and consumed by post, so no per-call state is cached. No concurrent
    SDK safety is promised.
    """
    def __init__(self, selection=None, *, runner=None, binding=None, runtime_factory=None,
                 score_thres=.25, nms_thres=.45, anchors=ANCHORS):
        if runner is None:
            if not isinstance(selection, ModelSelection):
                raise TypeError('Pass a ModelSelection from resolve_selection, or inject runner=/binding=.')
            runner=RuntimeModelRunner(selection,runtime_factory=runtime_factory)
            binding=runner.load()
        elif binding is None:
            binding=getattr(runner,'binding',None)
            if binding is None:raise ValueError('An injected runner must supply or be given its binding.')
        self.runner=runner;self.binding=binding
        self.score_thres=_threshold(score_thres,'score_thres');self.nms_thres=_threshold(nms_thres,'nms_thres')
        anchors=np.asarray(anchors,dtype=np.float32)
        if anchors.size!=18 or not np.isfinite(anchors).all() or np.any(anchors<=0):raise ValueError('anchors must contain 18 finite positive numbers.')
        self.anchors=np.frombuffer(anchors.tobytes(),dtype=np.float32).reshape(3,3,2)

    def set_scheduling_params(self,*,priority=0,bpu_cores=None):
        """Apply explicit scheduling values to the loaded board runtime."""
        self.runner.set_scheduling_params(priority=priority,bpu_cores=bpu_cores)

    def preprocess(self,image,*,resize_type=None):
        """Return owned packed/split uint8 inputs + frozen context for HWC U8 BGR."""
        return prepare_image(image,self.binding,resize_type)

    def infer(self,tensors):
        """Return metadata-validated native output arrays without numeric changes."""
        inputs=validate_tensors(self.binding,tensors)
        return validate_tensors(self.binding,self.runner(inputs),outputs=True)

    def postprocess(self,outputs,context,*,score_thres=None,nms_thres=None):
        """Decode bound native heads using matching per-call geometry.

        X5 raw F32 and S declared dequant are followed by source sigmoid anchor
        decoding. S uses class-wise NMS; X5 preserves its legacy OpenCV call.
        Invalid metadata/context/thresholds raise ValueError. Results own data.
        """
        if not isinstance(context,DetectionContext) or context.model_size!=self.binding.input_size:raise ValueError('Context does not match this detector geometry.')
        score=self.score_thres if score_thres is None else _threshold(score_thres,'score_thres')
        nms=self.nms_thres if nms_thres is None else _threshold(nms_thres,'nms_thres')
        values=apply_output_transform(self.binding.output_transform,validate_tensors(self.binding,outputs,outputs=True),self.binding.output_quants)
        if self.binding.selection.target=='x5':
            boxes,scores,ids=decode_x5(values,self.binding.output_names,self.binding.input_size,self.anchors,score,nms)
        else:
            pred=decode_outputs(self.binding.output_names,values,STRIDES,self.anchors)
            boxes,scores,ids=filter_predictions(pred,score)
            keep=np.asarray(classwise_nms(boxes,scores,ids,nms),dtype=int)
            boxes,scores,ids=boxes[keep],scores[keep],ids[keep]
        h,w=context.original_size
        boxes=scale_coords_back(boxes.copy(),w,h,context.model_size,context.model_size,context.resize_type)
        if self.binding.selection.target=='x5':boxes=boxes.astype(int)
        return DetectionResult(np.array(boxes,dtype=np.float32,copy=True),np.array(scores,dtype=np.float32,copy=True),np.array(ids,dtype=np.int32,copy=True))

    def predict(self,image,*,resize_type=None,score_thres=None,nms_thres=None):
        """Compose the same three public stages; explicit zero thresholds are valid."""
        prepared=self.preprocess(image,resize_type=resize_type)
        outputs=self.infer(prepared.tensors)
        return self.postprocess(outputs,prepared.context,score_thres=score_thres,nms_thres=nms_thres)

    def pre_process(self,image,*,resize_type=None):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image,resize_type=resize_type)

    def forward(self,tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self,outputs,context,*,score_thres=None,nms_thres=None):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs,context,score_thres=score_thres,nms_thres=nms_thres)
