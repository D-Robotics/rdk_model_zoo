"""Compare fresh-process source/unified ByteTrack captures; retain both full runs."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples._shared.platforms import require_execution_target
from samples.vision.bytetrack.runtime.python.model_binding import resolve_selection
from samples.vision.yolov5.evaluator.compare import _hash


def compare_captures(left_dir,right_dir):
    """Require exact frame/input/ID identity and declared raw/track tolerances."""
    dirs=[Path(left_dir),Path(right_dir)];records=[json.loads((d/'capture.json').read_text()) for d in dirs]
    left,right=records;checks={}
    for name in ('target','asset_id','model_sha256','video_sha256','priority','bpu_cores','metadata','code_sha256','board_identity'):
        checks[name]=name in left and name in right and left[name]==right[name]
    checks['successful_runs']=left['return_code']==right['return_code']==0
    checks['frame_count']=len(left['frames'])==len(right['frames']) and len(left['frames'])>0
    for a,b in zip(left['frames'],right['frames']):
        prefix=f'frame{a["frame"]}'
        checks[prefix+'.identity']=a['frame']==b['frame'] and a['image']['sha256']==b['image']['sha256']
        for side,(directory,frame) in enumerate(zip(dirs,(a,b))):
            checks[f'{prefix}.image_digest.{side}']=_hash(directory/frame['image']['file'])==frame['image']['sha256']
        for category in ('inputs','outputs'):
            aa,bb=a.get(category,{}),b.get(category,{})
            checks[prefix+'.'+category+'.names']=set(aa)==set(bb) and bool(aa)
            for key in set(aa)&set(bb):
                x=np.load(dirs[0]/aa[key]['file'],allow_pickle=False);y=np.load(dirs[1]/bb[key]['file'],allow_pickle=False)
                checks[prefix+'.'+category+'.'+key+'.binding']=all(_hash(directory/record['file'])==record['sha256'] and list(array.shape)==record['shape'] and str(array.dtype)==record['dtype'] for directory,record,array in zip(dirs,(aa[key],bb[key]),(x,y)))
                checks[prefix+'.'+category+'.'+key]=bool(x.shape==y.shape and x.dtype==y.dtype and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,rtol=0,atol=0 if category=='inputs' else 1e-5))
        x,y=a.get('tracks'),b.get('tracks');checks[prefix+'.track_count']=x is not None and y is not None and len(x)==len(y)
        if x is not None and y is not None:
            for i,(u,v) in enumerate(zip(x,y)):
                checks[f'{prefix}.track{i}']=bool(u['track_id']==v['track_id'] and np.allclose(u['tlbr'],v['tlbr'],rtol=0,atol=1e-4) and abs(u['score']-v['score'])<=1e-5)
    return dict(passed=all(checks.values()),checks=checks,tolerances=dict(raw=1e-5,boxes=1e-4,scores=1e-5,inputs=0,track_ids=0,rtol=0))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',choices=('s100','s100p','s600'),required=True);p.add_argument('--model-path');p.add_argument('--asset-id');p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-frames',type=int,default=30)
    a=p.parse_args(argv)
    try:
        require_execution_target(a.target);selection=resolve_selection(a.target,asset_id=a.asset_id,model_path=a.model_path)
        if a.max_frames<1:raise ValueError('max-frames must be positive.')
        directory=Path(a.output_dir).expanduser().resolve();directory.mkdir(parents=True,exist_ok=False)
        report=dict(started_utc=datetime.now(timezone.utc).isoformat(),argv=list(sys.argv),cwd=str(Path.cwd()),runs=[],passed=False,return_code=2)
        try:
            for side in ('legacy','unified'):
                command=[sys.executable,str(Path(__file__).with_name('capture.py')),'--side',side,'--target',a.target,'--model-path',str(selection.model_path),'--asset-id',selection.asset.reference,'--input',str(Path(a.input).expanduser().resolve()),'--output-dir',str(directory/side),'--max-frames',str(a.max_frames)]
                start=datetime.now(timezone.utc).isoformat();run=subprocess.run(command,text=True,capture_output=True)
                report['runs'].append(dict(side=side,argv=command,cwd=str(Path.cwd()),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),return_code=run.returncode,stdout=run.stdout,stderr=run.stderr))
            if all(v['return_code']==0 for v in report['runs']):
                report.update(compare_captures(directory/'legacy',directory/'unified'));report['return_code']=0 if report['passed'] else 1
        except Exception as exc:report['error']=dict(type=type(exc).__name__,message=str(exc))
        finally:
            report['finished_utc']=datetime.now(timezone.utc).isoformat();(directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=report['passed'],return_code=report['return_code'],evidence=str(directory))));return report['return_code']
    except (ValueError,OSError,RuntimeError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
