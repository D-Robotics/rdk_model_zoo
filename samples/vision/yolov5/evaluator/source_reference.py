"""Load fixed source code for migration comparison; not a runtime dependency."""
from pathlib import Path
import importlib.util,sys,types
ROOT=Path(__file__).resolve().parents[4]


def source_paths(target):
    group='x5' if target=='x5' else 's'
    base=ROOT/'platforms'/group
    return (base/'samples/vision/yolov5/runtime/python'/('yolov5_det.py' if group=='x5' else 'yolov5.py'),
            base/'utils/py_utils/preprocess.py',base/'utils/py_utils/postprocess.py',base/'utils/py_utils/nn_math.py')


def load_legacy(selection,factory,*,resize_type,score_thres,nms_thres,anchors):
    """Use the exact source implementation with an injected recording SDK factory.

    Only the hbm_runtime module key is temporarily replaced for host fixtures;
    unrelated imported extension modules remain cached. Source math is unchanged.
    """
    group='x5' if selection.target=='x5' else 's'
    paths=source_paths(selection.target)
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
        spec.loader.exec_module(mod);return mod
    old=sys.modules.get('hbm_runtime');path_before=list(sys.path)
    if old is None:sys.modules['hbm_runtime']=types.SimpleNamespace(QuantParams=object,HB_HBMRuntime=factory)
    try:
        module=load(paths[0],'yolov5_source_'+group)
        # Source owns its imported SDK reference; do not mutate the real SDK.
        module.hbm_runtime=types.SimpleNamespace(HB_HBMRuntime=factory)
        module.pre_utils=load(paths[1],f'platforms.{group}.utils.py_utils.preprocess')
        module.post_utils=load(paths[2],f'platforms.{group}.utils.py_utils.postprocess')
    finally:
        if old is None:sys.modules.pop('hbm_runtime',None)
        else:sys.modules['hbm_runtime']=old
        sys.path[:]=path_before
    import numpy as np
    cfg=module.YOLOv5Config(str(selection.model_path),resize_type=resize_type,score_thres=score_thres,nms_thres=nms_thres,
        anchors=list(anchors) if group=='x5' else np.array(anchors,dtype=np.float32).reshape(3,3,2))
    return (module.YOLOv5Detect if group=='x5' else module.YoloV5X)(cfg)
