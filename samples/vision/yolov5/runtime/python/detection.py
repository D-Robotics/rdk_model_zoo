# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOv5's pure image → native tensors → owned detections pipeline."""
from dataclasses import dataclass
import numpy as np
from samples._shared.quantization import apply_output_transform
from .model_binding import ANCHORS, STRIDES, validate_tensors
from .tensor_io import DetectionContext, prepare_image
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


class YOLOv5Task:
    """Run one bound detector with explicit per-call context; no SDK/file IO here.

    Instances carry fixed thresholds/anchors only. Context is returned by pre;
    post needs the matching context. No concurrent SDK safety is promised.
    """
    def __init__(self, runner, binding, *, score_thres=.25, nms_thres=.45, anchors=ANCHORS):
        self.runner=runner;self.binding=binding
        self.score_thres=_threshold(score_thres,'score_thres');self.nms_thres=_threshold(nms_thres,'nms_thres')
        anchors=np.asarray(anchors,dtype=np.float32)
        if anchors.size!=18 or not np.isfinite(anchors).all() or np.any(anchors<=0):raise ValueError('anchors must contain 18 finite positive numbers.')
        self.anchors=np.frombuffer(anchors.tobytes(),dtype=np.float32).reshape(3,3,2)

    def pre_process(self,image,*,resize_type=None):
        """Return owned packed/split uint8 inputs + frozen context for HWC U8 BGR."""
        return prepare_image(image,self.binding,resize_type)

    def forward(self,tensors):
        """Return metadata-validated native output arrays without numeric changes."""
        inputs=validate_tensors(self.binding,tensors)
        return validate_tensors(self.binding,self.runner(inputs),outputs=True)

    def post_process(self,outputs,context,*,score_thres=None,nms_thres=None):
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
        prepared=self.pre_process(image,resize_type=resize_type)
        outputs=self.forward(prepared.tensors)
        return self.post_process(outputs,prepared.context,score_thres=score_thres,nms_thres=nms_thres)
