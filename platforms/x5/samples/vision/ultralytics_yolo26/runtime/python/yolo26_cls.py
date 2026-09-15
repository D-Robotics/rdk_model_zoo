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
_name='_shared_yolo26_cls'
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,_SAMPLE/'runtime/python/yolo26_cls.py')
    _module=importlib.util.module_from_spec(_spec)
    sys.modules[_name]=_module
    _spec.loader.exec_module(_module)
_Base=sys.modules[_name].YOLO26Cls
_Config=sys.modules[_name].YOLO26ClsConfig
@dataclass
class YOLO26ClsConfig(_Config):
    topk: int = 5
    resize_type: int = 0
    platform: object = field(default_factory=lambda: resolve_platform("x5"))

YOLO26ClsConfig=YOLO26ClsConfig
class YOLO26Cls(_Base):
    """Retain the old import and return contract."""
    pass
