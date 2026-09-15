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
    aliases={'--image-path': '--image-dir', '--ann-path': '--annotation'}
    args=[aliases.get(a.split("=",1)[0],a.split("=",1)[0])+("="+a.split("=",1)[1] if "=" in a else "") for a in args]
    if not _has(args,'--family'):args += ['--family','yolo26']
    if not _has(args,'--image-dir'):args += ['--image-dir','../../../../datasets/dotav1/val/images']
    if not _has(args,'--json-save-path'):args += ['--json-save-path','results_obb.json']
    if not _has(args,'--conf-thres'):args += ['--conf-thres','0.25']
    if not _has(args,'--nms-thres'):args += ['--nms-thres','0.7']
    if not _has(args,'--limit'):args += ['--limit','0']
    if not _has(args,'--angle-sign'):args += ['--angle-sign','1.0']
    if not _has(args,'--angle-offset'):args += ['--angle-offset','0.0']
    if not _has(args,'--strides'):args += ['--strides','8,16,32']
    target=_SAMPLE/'evaluator/eval_yolo_obb.py'
    sys.argv=[str(target)]+args
    runpy.run_path(str(target),run_name="__main__")
