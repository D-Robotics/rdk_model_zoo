# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SDK-free YOLOWorld CLI until the selected X5 model is actually run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from samples.vision.yoloworld.runtime.python.model_binding import SAMPLE_DIR, list_available_assets, resolve_selection

def parse_prompts(value: str) -> list[str]:
    pieces = [part.strip() for part in value.split(',')]
    if not pieces or any(not part for part in pieces):
        raise ValueError('Prompts must contain one or more nonempty comma-separated words; empty prompts are rejected.')
    if len(pieces) > 32: raise ValueError('At most 32 prompts are supported.')
    return pieces

def build_parser():
    parser = argparse.ArgumentParser(description='YOLOWorld X5 open-vocabulary RGB + offline text detection.')
    parser.add_argument('--target', choices=('auto','x5','s100','s100p','s600'), default='auto')
    parser.add_argument('--model-path', default=None, help='Explicit yolo_world.bin path; requires exact --asset-id.')
    parser.add_argument('--asset-id', default=None, help='Exact manifest identity x5:yoloworld:yolo_world.bin.')
    parser.add_argument('--vocab-file', default=str(SAMPLE_DIR/'test_data/offline_vocabulary_embeddings.json'))
    parser.add_argument('--test-img', default=str(SAMPLE_DIR/'test_data/dog.jpeg'))
    parser.add_argument('--prompts', default='dog', help='Comma-separated vocabulary entries; empty entries are rejected.')
    parser.add_argument('--img-save-path', default=str(SAMPLE_DIR/'test_data/inference.png'))
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', type=int, nargs='+', default=[0])
    parser.add_argument('--score-thres', type=float, default=0.05)
    parser.add_argument('--nms-thres', type=float, default=0.45)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            for asset in list_available_assets(args.target): print(asset.reference)
            return 0
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires explicit --target x5; it never probes a board.')
        prompts = parse_prompts(args.prompts)
        selection = resolve_selection(args.target, model_path=args.model_path, asset_id=args.asset_id)
        if args.dry_run:
            print(json.dumps({'target':selection.target, 'asset_id':selection.asset.reference,
                              'model_path':str(selection.model_path),
                              'image_input':'float32[1,3,640,640] RGB, longest-side resize and top-left zero pad',
                              'text_input':'float32[1,32,512,1] offline embeddings, last prompt fills slots',
                              'outputs':['float32[1,8400,32] class scores','float32[1,8400,4] boxes'],
                              'prompts':prompts, 'score_thres':args.score_thres, 'nms_thres':args.nms_thres}, indent=2))
            return 0
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        if not selection.model_path.is_file(): raise FileNotFoundError(f'Model not found: {selection.model_path}; run model/download.sh explicitly.')
        import cv2
        from samples.vision.yoloworld.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
        from samples.vision.yoloworld.runtime.python.visualization import draw_results, save_image
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Cannot read image: {args.test_img}')
        with Path(args.vocab_file).expanduser().open(encoding='utf-8') as handle: vocabulary = json.load(handle)
        runner = RuntimeModelRunner(selection); binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = YOLOWorldTask(runner, binding, vocabulary, score_thres=args.score_thres, nms_thres=args.nms_thres)
        result = task.predict(image, prompts)
        save_image(args.img_save_path, draw_results(image, result, task.class_names))
        print(json.dumps({'target':selection.target, 'prompts':prompts, 'count':int(len(result.scores)), 'image_saved':str(Path(args.img_save_path).expanduser())}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr); return 2

if __name__ == '__main__': raise SystemExit(main())
