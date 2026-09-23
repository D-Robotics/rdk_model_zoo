"""YOLOv5 customer CLI: explicit model preparation, inference and rendering."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples.vision.yolov5.runtime.python.model_binding import SAMPLE_DIR,ANCHORS,STRIDES,list_available_assets,resolve_selection


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=('auto','x5','s100','s100p','s600'),default='auto')
    parser.add_argument('--variant',help='X5: n/s/m/l/x-v7.0 or s/m/l/x-v2.0; S: x-672. Omitted uses n-v7.0 / x-672.')
    parser.add_argument('--asset-id');parser.add_argument('--model-path')
    parser.add_argument('--test-img',help='Default X5 bus.jpg; S kite.jpg from sample test_data.')
    parser.add_argument('--label-file',default=str(SAMPLE_DIR/'test_data/coco_classes.names'))
    parser.add_argument('--img-save-path',default=str(SAMPLE_DIR/'test_data/result_unified.jpg'))
    parser.add_argument('--score-thres',type=float,default=.25);parser.add_argument('--nms-thres',type=float,default=.45)
    parser.add_argument('--resize-type',type=int,choices=(0,1),default=None,help='Omitted: X5 stretch(0), S letterbox(1).')
    parser.add_argument('--classes-num',type=int,choices=(80,),default=80,help='Published assets bind exactly 80 classes.')
    parser.add_argument('--anchors',type=lambda s:tuple(float(v) for v in s.split(',')),default=ANCHORS)
    parser.add_argument('--strides',type=lambda s:tuple(int(v) for v in s.split(',')),default=STRIDES,help='Published heads require 8,16,32.')
    parser.add_argument('--priority',type=int,default=0);parser.add_argument('--bpu-cores',nargs='+',type=int,default=[0])
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--list-models',action='store_true');modes.add_argument('--dry-run',action='store_true')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        if args.list_models:
            for a in list_available_assets(args.target):print(a.reference)
            return 0
        if args.dry_run and args.target=='auto':raise ValueError('--dry-run requires an explicit --target.')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in (args.score_thres,args.nms_thres)):raise ValueError('Thresholds must be finite in [0,1].')
        if not 0<=args.priority<=255 or any(v<0 for v in args.bpu_cores):raise ValueError('Invalid scheduling priority/core index.')
        if tuple(args.strides)!=STRIDES:raise ValueError('Published heads require strides 8,16,32.')
        if len(args.anchors)!=18 or any(not math.isfinite(v) or v<=0 for v in args.anchors):raise ValueError('anchors require 18 finite positive numbers.')
        selection=resolve_selection(args.target,variant=args.variant,asset_id=args.asset_id,model_path=args.model_path)
        image_path=Path(args.test_img).expanduser() if args.test_img else SAMPLE_DIR/'test_data'/('bus.jpg' if selection.target=='x5' else 'kite.jpg')
        if args.dry_run:
            print(json.dumps(dict(target=selection.target,variant=selection.variant,asset_id=selection.asset.reference,model_path=str(selection.model_path),test_img=str(image_path),input_protocol='packed_nv12_640' if selection.target=='x5' else 'split_nv12_672',metadata_status='requires runtime metadata',board_status='not-run'),indent=2));return 0
        import cv2
        from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
        from samples.vision.yolov5.runtime.python.visualization import draw_detections
        runner=RuntimeModelRunner(selection);binding=runner.load();runner.set_scheduling_params(priority=args.priority,bpu_cores=args.bpu_cores)
        task=YOLOv5Task(runner,binding,score_thres=args.score_thres,nms_thres=args.nms_thres,anchors=args.anchors)
        image=cv2.imread(str(image_path))
        if image is None:raise ValueError(f'Cannot read image: {image_path}')
        labels=Path(args.label_file).expanduser().read_text().splitlines()
        result=task.predict(image,resize_type=args.resize_type)
        path=Path(args.img_save_path).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),draw_detections(image,result,labels)):raise OSError(f'Cannot save: {path}')
        print(json.dumps(dict(boxes=result.boxes.tolist(),scores=result.scores.tolist(),class_ids=result.class_ids.tolist()),ensure_ascii=False))
        print(f'Saved {len(result.scores)} detections to {path}')
        return 0
    except (ValueError,OSError,RuntimeError,ImportError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
