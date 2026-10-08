# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOv5's pure image → native tensors → owned detections pipeline."""
from dataclasses import dataclass
import cv2
import numpy as np
from utils.py_utils.image import bgr_to_nv12_planes
from utils.py_utils.quantization import apply_output_transform
from .model_binding import ANCHORS, STRIDES, validate_tensors
from . import decode


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
            from .model_binding import ModelSelection
            from .model_runner import RuntimeModelRunner
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
            boxes,scores,ids=decode.decode_x5(values,self.binding.output_names,self.binding.input_size,self.anchors,score,nms)
        else:
            pred=decode.decode_outputs(self.binding.output_names,values,STRIDES,self.anchors)
            boxes,scores,ids=decode.filter_predictions(pred,score)
            keep=np.asarray(decode.classwise_nms(boxes,scores,ids,nms),dtype=int)
            boxes,scores,ids=boxes[keep],scores[keep],ids[keep]
        h,w=context.original_size
        boxes=decode.scale_coords_back(boxes.copy(),w,h,context.model_size,context.model_size,context.resize_type)
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
