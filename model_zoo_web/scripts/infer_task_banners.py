#!/usr/bin/env python3
"""Render real float-model outputs on generated inputs, with a pinned receipt.

This host-only command reuses native sample preprocessing and decoding. It
does not compile, download, edit the catalog or contact hardware. Raw outputs
stay in the run directory; the site receives the banner and compact receipts.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps
from PIL import __version__ as pillow_version

from infer_classification_banners import digest, file_record, font, write_json


def panel(photo, rows, *, cosine=False):
    canvas = Image.new('RGB', (1672, 941), (236, 236, 238))
    image = ImageOps.fit(photo.convert('RGB'), (862, 862), Image.Resampling.LANCZOS)
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 861, 861), radius=40, fill=255)
    canvas.paste(image, (40, 40), mask)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((941, 40, 1632, 901), radius=40, fill='white')
    for index, item in enumerate(rows):
        y = 132 + 163 * index
        label = item['label'].split(',')[0].replace('_', ' ')
        for size in range(40, 18, -1):
            face = font(size, index == 0)
            if draw.textlength(label, font=face) <= 430:
                break
        draw.text((989, y), label, font=face, fill=(24, 24, 28))
        number = f"{item['score']:.3f}" if cosine else f"{item['score']*100:.1f}%"
        face = font(40, index == 0)
        draw.text((1583-draw.textlength(number, font=face), y), number,
                  font=face, fill=(24, 24, 28))
        draw.rounded_rectangle((989, y+54, 1583, y+67), radius=6, fill=(226, 230, 235))
        width = round(max(0, min(1, item['score'])) * 594)
        if width:
            draw.rounded_rectangle((989, y+54, 989+max(width, 13), y+67),
                                   radius=6, fill=(46, 100, 234))
    return canvas


def pair(left, right):
    canvas = Image.new('RGB', (1672, 941), (236, 236, 238))
    for x, photo in ((40, left), (856, right)):
        image = ImageOps.fit(photo.convert('RGB'), (776, 861), Image.Resampling.LANCZOS)
        mask = Image.new('L', image.size)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, 775, 860), radius=40, fill=255)
        canvas.paste(image, (x, 40), mask)
    return canvas


def mask_overlay(photo, mask):
    pixels = np.asarray(photo).copy()
    pixels[mask] = np.uint8(pixels[mask].astype(np.float32)*.64 + np.array([49, 203, 220])*.36)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pixels, contours, -1, (255, 255, 255), max(2, photo.width//500))
    return Image.fromarray(pixels)


def depth_image(value, size):
    restored = cv2.resize(value, size, interpolation=cv2.INTER_LINEAR)
    lo, hi = map(float, np.percentile(restored, [2, 98]))
    scaled = np.uint8(np.clip((restored-lo)/max(hi-lo, 1e-8), 0, 1)*255)
    colored = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored), {'method': '2nd–98th percentile TURBO', 'range': [lo, hi],
                                      'units': 'relative model depth; no metric distance'}


def stage_binding(sample, stage, session):
    from utils.py_utils.runtime_meta import RuntimeMetadata
    from utils.py_utils.sam_binding import StageBinding
    metadata = {'model_name': 'source_float_'+stage}
    for side, values in (('input', session.get_inputs()), ('output', session.get_outputs())):
        if any(v.type != 'tensor(float)' for v in values):
            raise ValueError('SAM source graph must expose float32 tensors')
        metadata.update({side+'_names': [v.name for v in values],
                         side+'_shapes': {v.name: v.shape for v in values},
                         side+'_dtypes': {v.name: 'float32' for v in values}})
    return StageBinding(sample, 's100', stage, RuntimeMetadata.from_mapping(metadata))


def run(row, native, output, threads):
    source = native / row['input']
    committed = subprocess.check_output(['git', '-C', str(native), 'show', row['native_commit']+':'+row['input']])
    if hashlib.sha256(committed).hexdigest() != digest(source):
        raise ValueError('Input differs from its pinned native source commit')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output/'source.png')
    photo = Image.open(source).convert('RGB')
    bgr = np.asarray(photo)[:, :, ::-1].copy()
    raw, forwards, feeds, sessions = {}, [], [], {}
    for role, graph in row['graphs'].items():
        if digest(graph['path']) != graph['sha256']:
            raise ValueError('Graph differs from the conversion source')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions[role] = ort.InferenceSession(graph['path'], options, providers=['CPUExecutionProvider'])

    def forward(role, values):
        session = sessions[role]
        metadata = session.get_inputs()
        if set(values) != {v.name for v in metadata}:
            raise ValueError('Feed names differ from source graph')
        for item in metadata:
            value = values[item.name]
            if len(value.shape) != len(item.shape) or any(
                isinstance(wanted, int) and observed != wanted for observed, wanted in zip(value.shape, item.shape)
            ) or not np.isfinite(value).all():
                raise ValueError('Feed shape or values differ from source graph')
            feeds.append({'role': role, 'name': item.name, 'shape': list(value.shape),
                          'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()})
        metadata = session.get_outputs()
        values = session.run(None, values)
        if len(values) != len(metadata):
            raise ValueError('Missing source output')
        result = {}
        for item, value in zip(metadata, values):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Expected every finite float32 source output')
            result[item.name] = value
            raw[role+'__'+item.name] = value
            forwards.append({'role': role, 'name': item.name, 'shape': list(value.shape),
                             'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()})
        return result

    kind = row['kind']
    session = next(iter(sessions.values()))
    extra = {}
    result_image = None
    if kind == 'classification-panel':
        from utils.py_utils.classification_host import prepare_rgb, float_input
        from utils.py_utils.classification import topk_from_scores
        contract = {'size': 224, 'resize_shorter': 256, 'crop_pct': 224/256,
                    'geometry': 'resize_shorter_center_crop', 'interpolation': 'pil_bicubic',
                    'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                    'color': 'RGB', 'layout': 'NCHW', 'dtype': 'float32',
                    'class_count': 1000, 'output': 'logits', 'batch': 1}
        value = float_input(prepare_rgb(source, contract), contract)
        values = forward('model', {session.get_inputs()[0].name: value})
        labels = ast.literal_eval((native/'datasets/imagenet/imagenet_classes.names').read_text())
        decoded = topk_from_scores(next(iter(values.values())), top_k=5, labels=labels, softmax=True)
        rows = [{'class_id': int(i), 'label': label, 'score': float(score)}
                for i, label, score in zip(decoded.class_ids, decoded.labels, decoded.scores)]
        banner = panel(photo, rows)
        extra = {'topk': rows, 'visualization': {'method': 'Native ImageNet softmax Top-5'}}
    elif kind == 'similarity-panel':
        from samples.vision.clip.runtime.python.matching import prepare_inputs, CLIPMatcher
        from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer
        prompts = ['a photo of a golden retriever', 'a photo of a tabby cat', 'a photo of a wolf',
                   'a photo of a red fox', 'a photo of a rabbit']
        prepared = prepare_inputs(bgr, prompts, PromptTokenizer())
        images = forward('image', {sessions['image'].get_inputs()[0].name: prepared.tensors['image']})
        texts = forward('text', {sessions['text'].get_inputs()[0].name: prepared.tensors['texts']})
        matched = CLIPMatcher.postprocess(None, {'image_feature': next(iter(images.values())),
                                                'text_features': next(iter(texts.values()))})
        rows = [{'label': prompts[i], 'score': float(matched.scores[i])} for i in matched.order]
        banner = panel(photo, rows, cosine=True)
        extra = {'similarities': rows, 'prompts': prompts,
                 'visualization': {'method': 'Native L2 feature cosine; raw decimal scores, not probabilities'}}
    elif kind == 'prompt-mask-panel':
        from utils.py_utils.sam_tensor_io import prepare_encoder_input, prepare_decoder_input
        from utils.py_utils.sam_stages import DecoderStage
        encoder = stage_binding(row['source'], 'encoder', sessions['encoder'])
        decoder = stage_binding(row['source'], 'decoder', sessions['decoder'])
        prepared = prepare_encoder_input(bgr, encoder)
        embedding = forward('encoder', prepared.tensors)['image_embeddings']
        box = [95, 55, 420, 480] if row['source'] == 'mobile_sam' else None
        prepared = prepare_decoder_input(embedding, decoder, box)
        values = forward('decoder', prepared.tensors)
        decoded = DecoderStage(None, decoder).post_process(values)
        mask = cv2.resize(decoded['mask'].astype(np.uint8), photo.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        result_image = mask_overlay(photo, mask)
        left = photo.copy()
        draw = ImageDraw.Draw(left)
        sx, sy = photo.width/512, photo.height/512
        prompts = {'box': box} if box else {'positive_points': [[248, 210], [302, 315]], 'fixed_in_decoder': True}
        if box:
            draw.rectangle(tuple(v*(sx if i%2 == 0 else sy) for i, v in enumerate(box)), outline=(46, 100, 234), width=5)
        else:
            for x, y in prompts['positive_points']:
                x, y = x*sx, y*sy
                draw.ellipse((x-10, y-10, x+10, y+10), fill=(46, 100, 234), outline='white', width=3)
        banner = pair(left, result_image)
        raw['selected_mask'] = mask
        extra = {'prompts': prompts, 'mask_index': decoded['mask_index'], 'predicted_iou': decoded['iou'],
                 'visualization': {'method': 'Native highest-IoU decoder candidate; actual mask overlay',
                                   'selected_mask_shape': list(mask.shape), 'foreground_pixels': int(mask.sum())}}
    elif kind == 'segmentation-overlay':
        from utils.py_utils.yoloe26_geometry import letterbox
        from utils.py_utils.yoloe26_decode import decode_candidates, restore_masks
        from samples.vision.yoloe.runtime.python.yoloe import PF26Contract
        pixels, context = letterbox(bgr)
        value = np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255
        values = forward('model', {session.get_inputs()[0].name: value})
        contract = PF26Contract()
        boxes, scores, ids, coefficients = decode_candidates([values[n] for n in contract.required_roles], .5, 100, True)
        boxes, masks = restore_masks(boxes, coefficients, values['protos'][0], context)
        labels = (native/'samples/vision/yoloe/test_data/classes.names').read_text().splitlines()
        selected = [i for i in range(len(boxes)) if (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1]) > photo.width*photo.height*.01][:6]
        if not selected:
            raise ValueError('No confident foreground result for this cover')
        full = np.zeros((photo.height, photo.width), np.uint8)
        detections = []
        for i in selected:
            x1, y1, x2, y2 = np.asarray(boxes[i], dtype=np.int32)
            roi = masks[i]
            if roi.shape != (y2-y1, x2-x1):
                raise ValueError('Native ROI geometry differs from its actual box')
            full[y1:y2, x1:x2] |= roi
            detections.append({'class_id': int(ids[i]), 'label': labels[ids[i]], 'score': float(scores[i]),
                               'box': boxes[i].tolist()})
        result_image = mask_overlay(photo, full.astype(bool))
        banner = pair(photo, result_image)
        raw['rendered_mask'] = full
        extra = {'detections': detections, 'confidence_threshold': .5, 'masks': True,
                 'visualization': {'method': 'Native PF26 Top-K and ROI mask restoration; no NMS',
                                   'rendered_instances': selected, 'foreground_pixels': int(full.sum())}}
    elif kind == 'feature-panel':
        from samples.vision.dinov2.runtime.python.embedding import prepare_image
        prepared = prepare_image(bgr, SimpleNamespace(input_name='input', input_shape=(1, 3, 224, 224)))
        values = forward('model', prepared.tensors)
        features = values['patch_feat'][0]
        if features.shape != (256, 384) or values['cls_feat'].shape != (1, 384):
            raise ValueError('Unexpected DINOv2 dual-output contract')
        centered = features.astype(np.float64)-features.mean(axis=0, dtype=np.float64)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        basis = basis[:3]
        for axis in range(3):
            if basis[axis, np.argmax(np.abs(basis[axis]))] < 0:
                basis[axis] *= -1
        projected = centered@basis.T
        lo, hi = np.percentile(projected, [2, 98], axis=0)
        colors = np.uint8(np.clip((projected-lo)/np.maximum(hi-lo, 1e-8), 0, 1).reshape(16, 16, 3)*255)
        result_image = Image.fromarray(colors).resize(photo.size, Image.Resampling.NEAREST)
        raw['pca_basis'] = basis
        rgb = np.uint8(np.clip((prepared.tensors['input'][0].transpose(1, 2, 0)*[.229, .224, .225]+[.485, .456, .406])*255, 0, 255))
        banner = pair(Image.fromarray(rgb), result_image)
        extra = {'visualization': {'method': 'Per-image 3-component PCA, canonicalized signs, 2nd–98th percentile scaling',
                                   'semantics': 'False-color patch features, not semantic labels or segmentation'}}
    elif kind == 'matting-panel':
        from samples.vision.modnet.runtime.python.modnet import preprocess_image, MODNetMatting
        value, context = preprocess_image(bgr)
        values = forward('model', {session.get_inputs()[0].name: value})
        matte = MODNetMatting.postprocess(None, values['matte'], context)
        alpha = matte.astype(np.float32)[:, :, None]/255
        result_image = Image.fromarray(np.uint8(np.clip(np.asarray(photo)*alpha+np.array([229, 240, 247])*(1-alpha), 0, 255)))
        raw['restored_alpha'] = matte
        banner = pair(photo, result_image)
        extra = {'visualization': {'method': 'Native MODNet restored alpha, composite onto light blue',
                                   'alpha_range': [int(matte.min()), int(matte.max())]}}
    elif kind == 'depth-panel':
        if row['source'] == 'depth_anything_v2':
            from samples.vision.depth_anything_v2.conversion.export_small import preprocess
            value = preprocess(bgr)
            values = forward('model', {session.get_inputs()[0].name: value})
            depth = values['depth'][0]
        else:
            from samples.vision.yolo26_depth.conversion.prepare_calibration import prepare_tensor
            from samples.vision.yolo26_depth.runtime.python.yolo26_depth import make_context, restore_log_depth
            value, geometry = prepare_tensor(bgr, 's100', 'nv12')
            values = forward('model', {session.get_inputs()[0].name: value})
            log_depth = next(iter(values.values())).reshape(192, 192)
            depth = restore_log_depth(log_depth, make_context(*bgr.shape[:2], 'nv12', 'n'))
        result_image, visualization = depth_image(depth, photo.size)
        banner = pair(photo, result_image)
        extra = {'visualization': visualization}
    else:
        raise ValueError('Unsupported reference-banner kind')

    np.savez_compressed(output/'outputs.npz', **raw)
    banner.save(output/'banner.webp', 'WEBP', quality=92, method=6)
    files = {name: file_record(output/name) for name in ('source.png', 'outputs.npz', 'banner.webp')}
    if result_image is not None:
        result_image.save(output/'result.png')
        files['result.png'] = file_record(output/'result.png')
    helpers = {}
    for name in row['helpers']:
        path = native/name
        committed = subprocess.check_output(['git', '-C', str(native), 'show', row['native_commit']+':'+name])
        if hashlib.sha256(committed).hexdigest() != digest(path):
            raise ValueError('Native helper differs from its pinned commit')
        helpers[name] = digest(path)
    receipt = {'schema_version': 2, 'kind': 'host-float-model-inference-banner', 'status': 'passed',
               'source': row['source'], 'model_url': row['checkpoint_url'], 'model_file': row['checkpoint_file'],
               'checkpoint_sha256': row['checkpoint_sha256'], 'commit': row['native_commit'],
               'input': row['input'], 'input_sha256': digest(source), 'native_helpers': helpers,
               'graphs': row['graphs'], 'framework': 'ONNX Runtime '+ort.__version__,
               'graph_io': {role: {side: [{'name': v.name, 'shape': v.shape, 'dtype': v.type}
                                         for v in getattr(session, 'get_'+side)()]
                                   for side in ('inputs', 'outputs')}
                            for role, session in sessions.items()},
               'provider': 'CPUExecutionProvider', 'runtime': 'host-float-reference',
               'versions': {'onnxruntime': ort.__version__, 'numpy': np.__version__,
                            'opencv': cv2.__version__, 'pillow': pillow_version},
               'board_execution': 'not-run', 'dataset_accuracy': 'not-run',
               'feeds': feeds, 'forward_outputs': forwards, 'files': files,
               'renderer_sha256': digest(__file__), **extra}
    write_json(output/'provenance.json', receipt)
    print(json.dumps({'source': row['source'], 'status': 'passed', 'forward_outputs': len(forwards),
                      'topk': extra.get('topk'), 'similarities': extra.get('similarities')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'native-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--stem', action='append', default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.native_root.resolve()))
    for row in json.loads(args.plan.read_text()):
        if not args.stem or row['stem'] in args.stem:
            run(row, args.native_root, args.output/row['stem'], args.threads)


if __name__ == '__main__':
    main()
