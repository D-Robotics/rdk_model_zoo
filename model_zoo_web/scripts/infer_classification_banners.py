#!/usr/bin/env python3
"""Run pinned float ONNX classifiers on clean generated photos, then render covers.

The native classification preparation and Top-K functions own the model math.
This command does not download weights, compile models, contact a board, edit
the catalog or invent scores. Inputs, logits and receipts remain reproducible.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

CANVAS = (1672, 941)
BACKGROUND = (236, 236, 238)
PHOTO = (40, 40, 862, 862)
PANEL = (941, 40, 1632, 901)
BLUE = (46, 100, 234)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def font(size, bold=False):
    names = ('Inter-SemiBold.ttf', 'DejaVuSans-Bold.ttf', 'Arial Bold.ttf') if bold else (
        'Inter-Regular.ttf', 'DejaVuSans.ttf', 'Arial.ttf')
    roots = ('/usr/share/fonts/truetype/macos', '/usr/share/fonts/truetype/dejavu',
             '/System/Library/Fonts/Supplemental')
    for name in names:
        for root in roots:
            p = Path(root) / name
            if p.is_file():
                return ImageFont.truetype(str(p), size)
    raise FileNotFoundError('A readable Inter, DejaVu Sans or Arial font is required')


def render_classification(photo, topk):
    """Use the existing light-gray / white / blue Top-5 banner composition."""
    if len(topk) != 5:
        raise ValueError('Exactly five actual predictions are required')
    canvas = Image.new('RGB', CANVAS, BACKGROUND)
    x, y, width, height = PHOTO
    fitted = ImageOps.fit(photo.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width-1, height-1), radius=40, fill=255)
    canvas.paste(fitted, (x, y), mask)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(PANEL, radius=40, fill='white')
    draw.text((989, 76), 'Top-5', font=font(30), fill=(119, 124, 133))
    for index, row in enumerate(topk):
        label = row['label'].split(',')[0].strip().replace('_', ' ')
        score = row['score']
        if not 0 <= score <= 1:
            raise ValueError('Softmax score is outside [0,1]')
        y = (132, 295, 458, 621, 784)[index]
        chosen = font(42 if index == 0 else 40, index == 0)
        for size in range(42 if index == 0 else 40, 19, -1):
            chosen = font(size, index == 0)
            if draw.textlength(label, font=chosen) <= 430:
                break
        draw.text((989, y), label, font=chosen, fill=(24, 24, 28))
        percent = f'{score * 100:.1f}%'
        face = font(42 if index == 0 else 40, index == 0)
        draw.text((1583-draw.textlength(percent, font=face), y), percent,
                  font=face, fill=(24, 24, 28))
        bar_y = y+54
        draw.rounded_rectangle((989, bar_y, 1583, bar_y+13), radius=6,
                               fill=(226, 230, 235))
        fill = round(score*594)
        if fill:
            draw.rounded_rectangle((989, bar_y, 989+max(fill, 13), bar_y+13),
                                   radius=6, fill=BLUE if index == 0 else (143, 160, 198))
    return canvas


def file_record(path):
    p = Path(path)
    return {'file': p.name, 'sha256': digest(p), 'size_bytes': p.stat().st_size}


def run(row, source, output, labels, native_root, threads):
    from utils.py_utils.classification import topk_from_scores
    from utils.py_utils.classification_host import prepare_rgb, float_input

    graph = Path(row['onnx_path'])
    if digest(graph) != row['onnx_sha256']:
        raise ValueError('Float graph differs from the pinned conversion source')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output/'source.webp')
    contract = {'size':224, 'crop_pct':224/256, **row['normalization'],
                'geometry':'resize_shorter_center_crop', 'interpolation':'pil_bicubic',
                'color':'RGB', 'layout':'NCHW', 'dtype':'float32', 'class_count':1000,
                'output':'logits', 'batch':1}
    rgb = prepare_rgb(source, contract)
    value = float_input(rgb, contract)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1,3,224,224] or len(outputs) != 1:
        raise ValueError('Expected the recorded single-input/single-output classifier')
    started = time.perf_counter()
    logits = session.run([outputs[0].name], {inputs[0].name:value})[0]
    elapsed = time.perf_counter()-started
    if logits.shape != (1,1000) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError('Invalid ImageNet logits')
    decoded = topk_from_scores(logits, top_k=5, labels=labels, softmax=True)
    topk = [{'class_id':int(i), 'label':label, 'score':float(score)}
            for i, label, score in zip(decoded.class_ids, decoded.labels, decoded.scores)]
    shifted = logits.astype(np.float64)-float(logits.max())
    probabilities = np.exp(shifted); probabilities /= probabilities.sum()
    probabilities = probabilities.astype(np.float32)
    np.savez_compressed(output/'outputs.npz', logits=logits, probabilities=probabilities)
    with Image.open(source) as photo:
        image_size = list(photo.size)
        rendered = render_classification(photo, topk)
    rendered.save(output/'banner.webp', 'WEBP', quality=92, method=6)
    prediction = {'schema_version':1, 'task':'classification', 'model':{
        'catalog_id':row['id'], 'name':row['model_name'], 'variant':row['model_size'],
        'source_checkpoint_sha256':row['checkpoint_sha256'], 'onnx_sha256':row['onnx_sha256']},
        'image':{'sha256':digest(source), 'dimensions':image_size}, 'topk':topk}
    write_json(output/'predictions.json', prediction)
    helpers = ['utils/py_utils/classification.py', 'utils/py_utils/classification_host.py',
               'utils/py_utils/tensor_io.py']
    receipt = {'schema_version':1, 'kind':'host-float-model-inference-banner',
        'status':'passed', 'model':{**prediction['model'],
            'source_weight_url':row['checkpoint_url'], 'checkpoint_revision':row['checkpoint_revision'],
            'native_source_commit':row['native_commit']},
        'runtime':{'framework':'ONNX Runtime', 'version':ort.__version__,
                   'provider':'CPUExecutionProvider', 'threads':threads,
                   'single_inference_wall_seconds':elapsed, 'timing_scope':'host reference only; not a benchmark'},
        'input':{'name':inputs[0].name, 'shape':list(value.shape), 'dtype':str(value.dtype),
                 'tensor_sha256':hashlib.sha256(value.tobytes()).hexdigest(), 'contract':contract},
        'native_helpers':{p:digest(native_root/p) for p in helpers},
        'postprocessing':'native topk_from_scores, softmax=True; stable descending class IDs',
        'render':{'style':'Existing MobileNet/ResNet light-gray, rounded photo and white Top-5 panel with blue bars',
                  'dimensions':list(CANVAS), 'rendered_predictions':topk},
        'files':{name:file_record(output/name) for name in ('source.webp','outputs.npz','predictions.json','banner.webp')},
        'board_execution':'not-run', 'dataset_accuracy':'not-run',
        'note':'Host float reference from the exact conversion ONNX; not a BIN/HBM board result.',
        'renderer_sha256':digest(__file__)}
    write_json(output/'provenance.json', receipt)
    return {'stem':row['stem'], 'status':'passed', 'top1':topk[0], 'output':str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stem', action='append', default=[])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.native_root.resolve()))
    labels = ast.literal_eval((args.native_root/'datasets/imagenet/imagenet_classes.names').read_text())
    if set(labels) != set(range(1000)):
        raise ValueError('Expected the native ordered 1000-class ImageNet labels')
    rows = json.loads(args.plan.read_text())
    for row in rows:
        if args.stem and row['stem'] not in args.stem:
            continue
        source = args.inputs/row['stem']/'source.webp'
        if not source.is_file():
            continue
        result = run(row, source, args.output/row['stem'], labels,
                     args.native_root, args.threads)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
