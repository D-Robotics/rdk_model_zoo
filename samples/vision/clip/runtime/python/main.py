# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP native CLI: prepare an exact encoder pair and match text to one image."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.clip.runtime.python.model_binding import SAMPLE_DIR, list_available_assets, resolve_selection


def build_parser():
    """Build parser without importing BPU/ONNX runtimes or initializing BPE."""
    parser = argparse.ArgumentParser(description='CLIP X5 BPU image + CPU ONNX text matching.')
    parser.add_argument('--target', choices=('auto','x5','s100','s100p','s600'), default='auto', help='Actual execution target; only X5 has a published pair.')
    parser.add_argument('--image-asset-id', default=None, help='Exact image asset ID; required with image-model-path.')
    parser.add_argument('--text-asset-id', default=None, help='Exact text asset ID; required with text-model-path.')
    parser.add_argument('--image-model-path', default=None, help='Explicit local .bin; default published img_encoder.bin.')
    parser.add_argument('--text-model-path', default=None, help='Explicit local .onnx; default published text_encoder.onnx.')
    parser.add_argument('--test-img', default=str(SAMPLE_DIR/'test_data/dog.jpg'), help='Input BGR image path.')
    parser.add_argument('--texts', default='a diagram,a dog', help='Comma-separated prompts; empty items are removed.')
    parser.add_argument('--img-save-path', default=str(SAMPLE_DIR/'test_data/inference.png'), help='Annotated image destination; retains source default.')
    parser.add_argument('--priority', type=int, default=0, help='Image encoder priority 0..255.')
    parser.add_argument('--bpu-cores', type=int, nargs='+', default=[0], help='Nonnegative image encoder BPU cores.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true', help='List the exact pair without loading either runtime.')
    mode.add_argument('--dry-run', action='store_true', help='Resolve the pair with explicit target, without board or SDK.')
    return parser


def _run(selection, args):
    import cv2
    from samples.vision.clip.runtime.python.model_runner import RuntimeModelRunner
    from samples.vision.clip.runtime.python.tokenization import PromptTokenizer
    from samples.vision.clip.runtime.python.matching import CLIPTask
    from samples.vision.clip.runtime.python.visualization import draw_scores, save_image

    texts = [text.strip() for text in args.texts.split(',') if text.strip()]
    if not texts:
        raise ValueError('At least one nonempty text prompt is required.')
    image = cv2.imread(str(Path(args.test_img).expanduser()))
    if image is None:
        raise ValueError(f'Cannot read image: {args.test_img}')
    runner = RuntimeModelRunner(selection)
    binding = runner.load()
    runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    task = CLIPTask(runner, binding, PromptTokenizer())
    result = task.predict(image, texts)
    save_image(args.img_save_path, draw_scores(image, texts, result))
    print(json.dumps({'target':selection.target, 'prompts':texts,
                      'scores':result.scores.tolist(), 'order':result.order.tolist(),
                      'image_saved':str(Path(args.img_save_path).expanduser())},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            for asset in list_available_assets(args.target):
                print(asset.reference)
            return 0
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires --target x5; no board detection performed.')
        selection = resolve_selection(args.target, image_asset_id=args.image_asset_id,
                                      text_asset_id=args.text_asset_id,
                                      image_model_path=args.image_model_path,
                                      text_model_path=args.text_model_path)
        if args.dry_run:
            print(json.dumps({'target':selection.target,
                              'image_asset_id':selection.image_asset.reference,
                              'text_asset_id':selection.text_asset.reference,
                              'image_model_path':str(selection.image_model_path),
                              'text_model_path':str(selection.text_model_path),
                              'image_input':'F32[1,3,224,224] RGB [0,1]',
                              'text_input':'I32[N,77] BPE',
                              'outputs':'F32[1,512] and F32[N,512]; cosine, no softmax'},indent=2))
            return 0
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        for path in (selection.image_model_path, selection.text_model_path):
            if not path.is_file():
                raise FileNotFoundError(f'Model not found: {path}; prepare the pair with model/download.sh.')
        return _run(selection, args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
