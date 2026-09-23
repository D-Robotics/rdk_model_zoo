"""Capture one fresh-process legacy or unified ByteTrack stream for comparison."""
from pathlib import Path
import argparse,hashlib,importlib,importlib.util,json,sys,types
from dataclasses import asdict
from datetime import datetime,timezone
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples._shared.platforms import require_execution_target
from samples._shared.runtime_meta import RuntimeMetadata
from samples._shared.assets import verify_asset_file
from samples.vision.bytetrack.runtime.python.model_binding import SAMPLE_DIR,resolve_selection
from samples.vision.yolov5.evaluator.compare import _hash,_json
from samples.vision.yolov5.evaluator.source_reference import load_legacy,source_paths
from samples.vision.yolov5.runtime.python.model_binding import ANCHORS


def _legacy_task(selection,factory):
    """Use the source wrapper's actual predict/person-filter/update implementation."""
    detector=load_legacy(selection,factory,resize_type=1,score_thres=.25,nms_thres=.45,anchors=ANCHORS)
    detector_module=sys.modules['yolov5_source_s']
    source=ROOT/'platforms/s/samples/vision/bytetrack'
    before=list(sys.path);old=sys.modules.get('yolov5')
    sys.path.insert(0,str(source/'3rdparty'))
    sys.modules['yolov5']=types.SimpleNamespace(YoloV5X=lambda cfg:detector,YOLOv5Config=detector_module.YOLOv5Config)
    try:
        spec=importlib.util.spec_from_file_location('bytetrack_legacy_source',source/'runtime/python/bytetrack.py')
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        task=mod.ByteTrack(mod.ByteTrackConfig(str(selection.model_path)))
    finally:
        sys.path[:]=before
        if old is None:sys.modules.pop('yolov5',None)
        else:sys.modules['yolov5']=old
    return task


def capture_frames(selection,frames,output_dir,*,side,video_path,runtime_factory=None,priority=0,bpu_cores=(0,)):
    """Capture a bounded iterable of BGR frames, complete native arrays and tracks.

    Each CLI side runs in its own fresh process, so source-global track counters
    start together. This function's factory seam supports local tests only; real
    board identity is always required. Output directory must be new.
    """
    if side not in ('legacy','unified'):raise ValueError('Unknown side.')
    actual=require_execution_target(selection.target)
    if type(priority) is not int or not 0<=priority<=255 or not bpu_cores or any(type(c) is not int or c<0 for c in bpu_cores):raise ValueError('Invalid scheduling parameters.')
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=False)
    summary=dict(side=side,started_utc=datetime.now(timezone.utc).isoformat(),argv=list(sys.argv),cwd=str(Path.cwd()),target=actual,asset_id=selection.asset.reference,model_path=str(selection.model_path.resolve()),publisher_sha256=selection.asset.sha256,video_path=str(Path(video_path).resolve()),frames=[],return_code=2,source_ref='380e1a2bf42041af54be6f34935e50197cfadff9',priority=priority,bpu_cores=list(bpu_cores))
    summary['board_identity']={str(p):p.read_text().strip() if p.is_file() else None for p in [Path('/sys/class/boardinfo/soc_name'),Path('/sys/class/boardinfo/board_type')]}
    summary['versions']=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__)
    import importlib.metadata
    current={};failure=None
    def save_array(filename,array):
        path=directory/filename;np.save(path,array)
        return dict(file=filename,sha256=_hash(path),shape=list(array.shape),dtype=str(array.dtype))
    try:
        summary['model_sha256']=_hash(selection.model_path);summary['video_sha256']=_hash(video_path)
        verify_asset_file(selection.asset,selection.model_path)
        for package in ('scipy','lap','cython-bbox'):summary['versions'][package]=importlib.metadata.version(package)
        source=ROOT/'platforms/s/samples/vision/bytetrack'
        code=list((SAMPLE_DIR/'runtime/python').rglob('*.py'))+list((SAMPLE_DIR/'evaluator').glob('*.py'))+list((SAMPLE_DIR.parent/'yolov5/runtime/python').glob('*.py'))+list((SAMPLE_DIR.parent/'yolov5/evaluator').glob('*.py'))+list(source_paths('s100'))+list((source/'runtime/python').glob('*.py'))+list((source/'3rdparty/tracker').glob('*.py'))+[ROOT/'samples/_shared'/n for n in ('assets.py','platforms.py','runtime_meta.py','quantization.py','image.py')]
        summary['code_sha256']={str(p.relative_to(ROOT)):_hash(p) for p in code}
        if runtime_factory is None:
            from samples._shared.model_runner import _default_runtime_factory
            runtime_factory=_default_runtime_factory()
        def factory(path):
            runtime=runtime_factory(path);summary['metadata']=asdict(RuntimeMetadata.from_runtime(runtime))
            class Recorder:
                def __getattr__(self,name):return getattr(runtime,name)
                def run(self,values):
                    current['inputs']={n:save_array(f'{current["frame"]:05d}_input_{i}.npy',a) for i,(n,a) in enumerate(values[runtime.model_names[0]].items())}
                    raw=runtime.run(values)
                    current['outputs']={n:save_array(f'{current["frame"]:05d}_output_{i}.npy',a) for i,(n,a) in enumerate(raw[runtime.model_names[0]].items())}
                    return raw
            return Recorder()
        if side=='legacy':
            task=_legacy_task(selection,factory);task.set_scheduling_params(priority=priority,bpu_cores=list(bpu_cores))
        else:
            from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
            from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
            from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
            runner=RuntimeModelRunner(selection,runtime_factory=factory);binding=runner.load();runner.set_scheduling_params(priority=priority,bpu_cores=list(bpu_cores))
            task=ByteTrackTask(YOLOv5Task(runner,binding))
        for index,frame in enumerate(frames,1):
            current=dict(frame=index,image=save_array(f'{index:05d}_image.npy',frame));summary['frames'].append(current)
            tracks=task.predict(frame)
            rows=[dict(track_id=int(t.track_id),tlbr=[float(v) for v in t.tlbr],score=float(t.score)) for t in tracks]
            if any(not np.isfinite(row['tlbr']+[row['score']]).all() for row in rows):
                raise ValueError('Tracker returned non-finite state; capture failed, not a numerical pass.')
            current['tracks']=rows
        if not summary['frames']:raise ValueError('No video frames decoded.')
        summary['return_code']=0
    except Exception as exc:
        summary['error']=dict(type=type(exc).__name__,message=str(exc));failure=exc
    finally:
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        (directory/'capture.json').write_text(json.dumps(summary,indent=2,default=_json,allow_nan=False)+'\n')
    if failure:raise failure
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--side',choices=('legacy','unified'),required=True);p.add_argument('--target',choices=('s100','s100p','s600'),required=True);p.add_argument('--model-path');p.add_argument('--asset-id');p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-frames',type=int,default=30)
    a=p.parse_args(argv);cap=None
    try:
        if a.max_frames<1:raise ValueError('max-frames must be positive for bounded evidence capture.')
        selection=resolve_selection(a.target,asset_id=a.asset_id,model_path=a.model_path)
        cap=cv2.VideoCapture(str(Path(a.input).expanduser()))
        if not cap.isOpened():raise ValueError('Cannot open prepared video.')
        def frames():
            for _ in range(a.max_frames):
                ok,image=cap.read()
                if not ok:break
                yield image
        result=capture_frames(selection,frames(),a.output_dir,side=a.side,video_path=Path(a.input).expanduser())
        print(json.dumps(dict(side=a.side,frames=len(result['frames']),return_code=0)));return 0
    except (ValueError,OSError,RuntimeError,ImportError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2
    finally:
        if cap is not None:cap.release()

if __name__=='__main__':raise SystemExit(main())
