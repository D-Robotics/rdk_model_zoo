# Compatibility entry; implementation is maintained in the shared sample.
from pathlib import Path
import sys, runpy
_HERE=Path(__file__).resolve()
_ROOT=next((p for p in _HERE.parents if (p/'samples/vision/ultralytics_yolo/runtime/python/yolo_dispatch.py').is_file()),None)
if _ROOT is None:raise ImportError('Complete repository checkout with the shared Ultralytics sample is required.')
_SAMPLE=_ROOT/'samples/vision/ultralytics_yolo'
sys.path.insert(0,str(_SAMPLE/'runtime/python'))
def _has(args,key):return any(a==key or a.startswith(key+'=') for a in args)

if __name__=='__main__':
    args=list(sys.argv[1:])
    if not _has(args,'--platform'):args += ['--platform','x5']
    target=_SAMPLE/'conversion/yolo26/export_yolo26_pose_bpu.py'
    sys.argv=[str(target)]+args
    runpy.run_path(str(target),run_name="__main__")
