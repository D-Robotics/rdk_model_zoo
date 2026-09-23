"""Explicit prepared-video ByteTrack CLI; no automatic model/video installation."""
import argparse,json,math,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples.vision.bytetrack.runtime.python.model_binding import SAMPLE_DIR,resolve_selection,list_available_assets


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='auto',choices=('auto','x5','s100','s100p','s600'));p.add_argument('--asset-id');p.add_argument('--model-path')
    p.add_argument('--input',default=str(SAMPLE_DIR/'test_data/track_test.mp4'),help='User-prepared video, not bundled or downloaded by this entry.')
    p.add_argument('--output',default=str(SAMPLE_DIR/'test_data/result_unified.mp4'))
    p.add_argument('--records',default=None,help='Optional JSONL with every frame and its owned track records.')
    p.add_argument('--score-thres',type=float,default=.25);p.add_argument('--nms-thres',type=float,default=.45)
    p.add_argument('--track-thresh',type=float,default=.3);p.add_argument('--track-buffer',type=int,default=60);p.add_argument('--match-thresh',type=float,default=.8);p.add_argument('--frame-rate',type=int,default=30);p.add_argument('--mot20',action='store_true')
    p.add_argument('--priority',type=int,default=0);p.add_argument('--bpu-cores',type=int,nargs='+',default=[0]);p.add_argument('--max-frames',type=int,default=0,help='0 processes all video frames.')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--list-models',action='store_true');modes.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None):
    a=build_parser().parse_args(argv);capture=writer=record_file=None
    try:
        if a.list_models:
            for asset in list_available_assets(a.target):print(asset.reference)
            return 0
        if a.dry_run and a.target=='auto':raise ValueError('--dry-run requires explicit --target.')
        from samples.vision.bytetrack.runtime.python.tracking import TrackingConfig
        cfg=TrackingConfig(a.track_thresh,a.track_buffer,a.match_thresh,a.frame_rate,a.mot20)
        if a.max_frames<0 or not 0<=a.priority<=255 or any(x<0 for x in a.bpu_cores):raise ValueError('Invalid max-frames or scheduling parameters.')
        if any(not math.isfinite(x) or not 0<=x<=1 for x in [a.score_thres,a.nms_thres]):raise ValueError('Detection thresholds must be finite [0,1].')
        selected=resolve_selection(a.target,model_path=a.model_path,asset_id=a.asset_id)
        if a.dry_run:
            print(json.dumps(dict(target=selected.target,asset_id=selected.asset.reference,model_path=str(selected.model_path),input=a.input,video_status='user preparation required',board_status='not-run',tracking=asdict(cfg)),indent=2));return 0
        import cv2
        from samples.vision.bytetrack.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
        from samples.vision.bytetrack.runtime.python.visualization import draw_tracks
        from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
        input_path=Path(a.input).expanduser();output_path=Path(a.output).expanduser()
        if not input_path.is_file():raise ValueError(f'Missing video: {input_path}; prepare it explicitly first.')
        if input_path.resolve()==output_path.resolve():raise ValueError('Output video must differ from input video.')
        if a.records and Path(a.records).expanduser().resolve() in (input_path.resolve(),output_path.resolve()):raise ValueError('records must differ from video paths.')
        runner=RuntimeModelRunner(selected);binding=runner.load();runner.set_scheduling_params(priority=a.priority,bpu_cores=a.bpu_cores)
        task=ByteTrackTask(YOLOv5Task(runner,binding,score_thres=a.score_thres,nms_thres=a.nms_thres),config=cfg)
        capture=cv2.VideoCapture(str(input_path))
        if not capture.isOpened():raise ValueError(f'Cannot open video: {input_path}')
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=float(capture.get(cv2.CAP_PROP_FPS))
        fps=fps if math.isfinite(fps) and fps>0 else 30.0
        output_path.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():raise OSError(f'Cannot open video writer: {output_path}')
        if a.records:
            record_path=Path(a.records).expanduser();record_path.parent.mkdir(parents=True,exist_ok=True);record_file=record_path.open('w')
        count=0
        while not a.max_frames or count<a.max_frames:
            ok,frame=capture.read()
            if not ok:break
            tracks=task.predict(frame);writer.write(draw_tracks(frame,tracks));count+=1
            if record_file:record_file.write(json.dumps(dict(frame=count,tracks=[asdict(t) for t in tracks]))+'\n')
        if not count:raise ValueError('Video contained no decodable frames.')
        print(f'Saved {count} tracked frames to {output_path}');return 0
    except (ValueError,OSError,RuntimeError,ImportError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2
    finally:
        if capture is not None:capture.release()
        if writer is not None:writer.release()
        if record_file is not None:record_file.close()

if __name__=='__main__':raise SystemExit(main())
