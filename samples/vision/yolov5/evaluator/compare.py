"""Same-board fixed-image source/unified comparison with complete evidence."""
from pathlib import Path
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from dataclasses import asdict
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples._shared.runtime_meta import RuntimeMetadata
from samples._shared.platforms import require_execution_target
from samples._shared.assets import verify_asset_file
from samples.vision.yolov5.runtime.python.model_binding import SAMPLE_DIR,ANCHORS,resolve_selection
from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task,_threshold
from samples.vision.yolov5.evaluator.source_reference import load_legacy,source_paths


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if hasattr(value,'quant_type'):
        q=value.quant_type
        return dict(quant_type=str(getattr(q,'name',q)),scale=getattr(value,'scale',[]),zero_point=getattr(value,'zero_point',[]),axis=getattr(value,'axis',None))
    if isinstance(value,Path):return str(value)
    raise TypeError(f'Unsupported evidence value {type(value).__name__}.')


def run_comparison(selection,image,image_path,output_dir,*,resize_type=None,score_thres=.25,nms_thres=.45,priority=0,bpu_cores=None,runtime_factory=None):
    """Retain native inputs/outputs and owned results; never turn mismatch into pass.

    Requires a new output directory, valid board identity and prepared files.
    runtime_factory is only a host seam; identity is still gated. Returns summary;
    failed execution leaves error evidence and re-raises. No downloads occur.
    """
    actual=require_execution_target(selection.target)
    directory=Path(output_dir).expanduser().resolve()
    if directory.exists():raise FileExistsError(f'Evidence directory already exists: {directory}')
    _threshold(score_thres,'score_thres');_threshold(nms_thres,'nms_thres')
    if type(priority) is not int or not 0<=priority<=255:raise ValueError('priority must be 0..255.')
    if bpu_cores is not None and (not bpu_cores or any(type(v) is not int or v<0 for v in bpu_cores)):raise ValueError('Invalid BPU cores.')
    resize=(0 if selection.target=='x5' else 1) if resize_type is None else resize_type
    if resize not in (0,1):raise ValueError('resize_type must be 0 or 1.')
    cores=[0] if bpu_cores is None else bpu_cores
    records={s:{} for s in ('legacy','unified')};summary=dict(started_utc=datetime.now(timezone.utc).isoformat(),argv=list(sys.argv),cwd=str(Path.cwd()),target=actual,asset_id=selection.asset.reference,publisher_sha256=selection.asset.sha256,model_path=str(selection.model_path.resolve()),model_sha256=None,image_path=str(Path(image_path).resolve()),image_sha256=None,decoded_shape=list(image.shape),decoded_dtype=str(image.dtype),resize_type=resize,score_thres=score_thres,nms_thres=nms_thres,priority=priority,bpu_cores=cores,host_versions=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__),metadata={},code_sha256={},passed=False)
    summary['source_ref']='ac115717197920355fc390bb04299b20e6436864' if selection.target=='x5' else '380e1a2bf42041af54be6f34935e50197cfadff9'
    summary['board_identity']={str(p):p.read_text().strip() if p.is_file() else None for p in [Path('/sys/class/boardinfo/soc_name'),Path('/sys/class/boardinfo/board_type')]}
    directory.mkdir(parents=True,exist_ok=False);failure=None
    try:
        summary['model_sha256']=_hash(selection.model_path);summary['image_sha256']=_hash(image_path)
        verify_asset_file(selection.asset,selection.model_path)
        code=list((SAMPLE_DIR/'runtime/python').glob('*.py'))+list((SAMPLE_DIR/'evaluator').glob('*.py'))+list(source_paths(selection.target))+[ROOT/'samples/_shared'/n for n in ['assets.py','platforms.py','runtime_meta.py','quantization.py','image.py']]
        summary['code_sha256']={str(p.relative_to(ROOT)):_hash(p) for p in code}
        if runtime_factory is None:
            from samples._shared.model_runner import _default_runtime_factory
            runtime_factory=_default_runtime_factory()
        def factory(side):
            def create(path):
                runtime=runtime_factory(path)
                summary['metadata'][side]=asdict(RuntimeMetadata.from_runtime(runtime))
                class Recorder:
                    def __getattr__(self,name):return getattr(runtime,name)
                    def run(self,values):
                        records[side]['inputs']={n:a.copy() for n,a in values[runtime.model_names[0]].items()}
                        out=runtime.run(values)
                        records[side]['outputs']={n:a.copy() for n,a in out[runtime.model_names[0]].items()}
                        return out
                return Recorder()
            return create
        old=load_legacy(selection,factory('legacy'),resize_type=resize,score_thres=score_thres,nms_thres=nms_thres,anchors=ANCHORS)
        old.set_scheduling_params(priority=priority,bpu_cores=cores)
        expected=old.predict(image)
        if selection.target=='x5':
            boxes=np.asarray([v[2:] for v in expected],np.float32).reshape(-1,4);scores=np.asarray([v[1] for v in expected],np.float32);ids=np.asarray([v[0] for v in expected],np.int32)
        else:boxes,scores,ids=expected
        records['legacy']['result']=dict(boxes=boxes,scores=scores,class_ids=ids)
        runner=RuntimeModelRunner(selection,runtime_factory=factory('unified'));binding=runner.load();runner.set_scheduling_params(priority=priority,bpu_cores=cores)
        result=YOLOv5Task(runner,binding,score_thres=score_thres,nms_thres=nms_thres).predict(image,resize_type=resize)
        records['unified']['result']=dict(boxes=result.boxes,scores=result.scores,class_ids=result.class_ids)
        checks={};maxdiff={}
        for category in ('inputs','outputs','result'):
            left,right=records['legacy'][category],records['unified'][category]
            checks[category+'_names']=set(left)==set(right)
            for name in set(left)&set(right):
                a,b=left[name],right[name];key=category+'.'+name
                # Source S IDs are int64; only the explicitly normalized result ID dtype differs.
                shape_ok=a.shape==b.shape;dtype_ok=a.dtype==b.dtype or (category=='result' and name=='class_ids')
                finite=np.isfinite(a).all() and np.isfinite(b).all()
                atol=0 if category=='inputs' or name=='class_ids' else (1e-4 if category=='result' and name=='boxes' else 1e-5)
                checks[key]=bool(shape_ok and dtype_ok and finite and np.allclose(a,b,rtol=0,atol=atol))
                maxdiff[key]=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if shape_ok and a.size else (0 if shape_ok else None)
        summary.update(checks=checks,max_abs_diff=maxdiff,tolerances=dict(inputs=0,raw=1e-5,boxes=1e-4,scores=1e-5,class_ids=0,rtol=0),passed=all(checks.values()))
    except Exception as exc:
        summary['error']=dict(type=type(exc).__name__,message=str(exc));failure=exc
    finally:
        arrays={}
        for side,categories in records.items():
            for category,values in categories.items():
                for index,(name,value) in enumerate(values.items()):
                    filename=f'{side}_{category}_{index}.npy';path=directory/filename;np.save(path,value)
                    arrays[filename]=dict(tensor_name=name,shape=list(value.shape),dtype=str(value.dtype),sha256=_hash(path))
        summary['arrays']=arrays;summary['finished_utc']=datetime.now(timezone.utc).isoformat();summary['return_code']=2 if failure else (0 if summary['passed'] else 1)
        (directory/'comparison.json').write_text(json.dumps(summary,default=_json,allow_nan=False,indent=2)+'\n')
    if failure:raise failure
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('x5','s100','s100p','s600'),required=True);p.add_argument('--variant');p.add_argument('--asset-id');p.add_argument('--model-path');p.add_argument('--test-img');p.add_argument('--output-dir',required=True)
    p.add_argument('--resize-type',type=int,choices=(0,1));p.add_argument('--score-thres',type=float,default=.25);p.add_argument('--nms-thres',type=float,default=.45);p.add_argument('--priority',type=int,default=0);p.add_argument('--bpu-cores',type=int,nargs='+')
    a=p.parse_args(argv)
    try:
        selection=resolve_selection(a.target,variant=a.variant,asset_id=a.asset_id,model_path=a.model_path)
        path=Path(a.test_img).expanduser() if a.test_img else SAMPLE_DIR/'test_data'/('bus.jpg' if a.target=='x5' else 'kite.jpg')
        image=cv2.imread(str(path))
        if image is None:raise ValueError(f'Cannot read image: {path}')
        summary=run_comparison(selection,image,path,a.output_dir,resize_type=a.resize_type,score_thres=a.score_thres,nms_thres=a.nms_thres,priority=a.priority,bpu_cores=a.bpu_cores)
        print(json.dumps(dict(passed=summary['passed'],return_code=summary['return_code'],evidence=str(Path(a.output_dir).resolve()))));return summary['return_code']
    except (ValueError,OSError,RuntimeError,ImportError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
