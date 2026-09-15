from pathlib import Path
import sys, runpy
_HERE=Path(__file__).resolve()
_ROOT=next((p for p in _HERE.parents if (p/'samples/vision/ultralytics_yolo/runtime/python/yolo_dispatch.py').is_file()),None)
if _ROOT is None:raise ImportError('Complete repository checkout with the shared Ultralytics sample is required.')
_SAMPLE=_ROOT/'samples/vision/ultralytics_yolo'
sys.path.insert(0,str(_SAMPLE/'runtime/python'))
def _has(args,key):return any(a==key or a.startswith(key+'=') for a in args)

import importlib.util
from dataclasses import dataclass, field
from typing import *
import numpy as np
from yolo_platform import resolve_platform
_name='_shared_yolo26_seg'
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,_SAMPLE/'runtime/python/yolo26_seg.py')
    _module=importlib.util.module_from_spec(_spec)
    sys.modules[_name]=_module
    _spec.loader.exec_module(_module)
_Base=sys.modules[_name].YOLO26Seg
_Config=sys.modules[_name].YOLO26SegConfig
@dataclass
class YOLO26SegConfig(_Config):
    classes_num: int = 80
    score_thres: float = 0.25
    nms_thres: float = 0.65
    resize_type: int = 1
    strides: np.ndarray = field(default_factory=lambda: np.array([8, 16, 32], dtype=np.int32))
    platform: object = field(default_factory=lambda: resolve_platform("x5"))

YOLO26SegConfig=YOLO26SegConfig
class YOLO26Seg(_Base):
    """Retain the old import and return contract."""
    def post_process(self,outputs,ori_img_w,ori_img_h,*args,**kwargs):
        boxes,scores,ids,crops=super().post_process(outputs,ori_img_w,ori_img_h,*args,**kwargs)
        masks=np.zeros((len(boxes),ori_img_h,ori_img_w),dtype=bool)
        for i,(box,crop) in enumerate(zip(boxes,crops)):
            x1,y1=max(0,int(box[0])),max(0,int(box[1]))
            x2,y2=min(ori_img_w,int(box[2])),min(ori_img_h,int(box[3]))
            if x2>x1 and y2>y1:masks[i,y1:y2,x1:x2]=crop
        return boxes,scores,ids,masks
