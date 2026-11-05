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
from PIL import Image, ImageDraw, ImageOps, ImageFont
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


def pair(left, right, *, preserve_source=False):
    canvas = Image.new('RGB', (1672, 941), (236, 236, 238))
    for x, photo in ((40, left), (856, right)):
        image = (ImageOps.contain if preserve_source else ImageOps.fit)(
            photo.convert('RGB'), (776, 861), Image.Resampling.LANCZOS)
        mask = Image.new('L', image.size)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width-1, image.height-1), radius=40, fill=255)
        canvas.paste(image, (x+(776-image.width)//2, 40+(861-image.height)//2), mask)
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
            expected_dtype = {'tensor(float)': np.dtype('float32'), 'tensor(int64)': np.dtype('int64')}.get(item.type)
            if expected_dtype is None or value.dtype != expected_dtype or not np.isfinite(value).all():
                raise ValueError('Source output dtype or finite-value contract differs')
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
        if row['source'] == '3dresnet':
            import importlib
            native_video = importlib.import_module('samples.vision.3dresnet.runtime.python.public')
            clip = native/row['clip']
            value = np.load(clip, allow_pickle=False)
            if value.dtype != np.float32 or value.shape != (1,3,16,112,112) or not np.isfinite(value).all():
                raise ValueError('Expected the complete prepared temporal input')
            frame_hashes = [hashlib.sha256(value[0,:,i].tobytes()).hexdigest() for i in range(16)]
            if len(set(frame_hashes)) < 2:
                raise ValueError('Video banner cannot infer from a repeated still image')
            preview = np.uint8(np.rint(np.clip(value[0,:,8].transpose(1,2,0)
                *np.array(native_video.STD,np.float32)+np.array(native_video.MEAN,np.float32),0,1)*255))
            np.testing.assert_array_equal(np.asarray(photo), preview)
            values = forward('model', {'images': value})
            rows = native_video.predictions(values['logits'], row['labels'])
            extra['video'] = {'input':row['clip'], 'sha256':digest(clip), 'shape':list(value.shape),
                              'dtype':'float32','preview_frame':8,'frame_sha256':frame_hashes}
        elif row['source'] == 'vit':
            from samples.vision.vit.runtime.python.imagenet import preprocess, predictions
            value = preprocess(source)
            values = forward('model', {session.get_inputs()[0].name: value})
            decoded = predictions(values['logits'], row['labels'])
            rows = [{'class_id': r['class_id'], 'label': r['label'], 'score': r['probability']} for r in decoded]
        else:
            value = float_input(prepare_rgb(source, contract), contract)
            values = forward('model', {session.get_inputs()[0].name: value})
            labels = ast.literal_eval((native/'datasets/imagenet/imagenet_classes.names').read_text())
            decoded = topk_from_scores(next(iter(values.values())), top_k=5, labels=labels, softmax=True)
            rows = [{'class_id': int(i), 'label': label, 'score': float(score)}
                    for i, label, score in zip(decoded.class_ids, decoded.labels, decoded.scores)]
        banner = panel(photo, rows)
        extra.update(topk=rows, visualization={'method': 'Native source logits softmax Top-5'})
    elif kind == 'similarity-panel':
        from samples.vision.clip.runtime.python.matching import prepare_inputs, CLIPMatcher
        from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer
        prompts = row.get('prompts', ['a photo of a golden retriever', 'a photo of a tabby cat', 'a photo of a wolf',
                                    'a photo of a red fox', 'a photo of a rabbit'])
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
        box = row.get('box', [95, 55, 420, 480]) if row['source'] == 'mobile_sam' else None
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
        banner = pair(left, result_image, preserve_source=True)
        raw['selected_mask'] = mask
        extra = {'prompts': prompts, 'mask_index': decoded['mask_index'], 'predicted_iou': decoded['iou'],
                 'visualization': {'method': 'Native highest-IoU decoder candidate; actual mask overlay',
                                   'selected_mask_shape': list(mask.shape), 'foreground_pixels': int(mask.sum())}}
    elif kind == 'detection-overlay' and row['source'] == 'fcos':
        from samples.vision.fcos.runtime.python.torchvision import preprocess, decode
        from infer_vision_banners import overlay
        value, size = preprocess(bgr)
        values = forward('model', {session.get_inputs()[0].name: value})
        decoded = decode(values, size)
        boxes, scores, ids = (decoded[k] for k in ('boxes', 'scores', 'labels'))
        result_image, selected = overlay(photo, boxes, scores, ids, row['labels'])
        banner = ImageOps.fit(result_image, (1672, 941), Image.Resampling.LANCZOS)
        extra = {'detections': [{'class_id': r['class_id'], 'label': r['label'], 'score': r['score'],
                                 'box': boxes[r['prediction_index']].tolist()} for r in selected],
                 'confidence_threshold': .5,
                 'visualization': {'method': 'Native ResNet50-FPN FCOS preprocessing, all three dense heads, exact stride anchors and class-aware NMS',
                                   'rendered_predictions': selected, 'rendered_size': [1672, 941], 'layout': 'full-bleed'}}
    elif kind == 'detection-overlay':
        from samples.vision.fasterrcnn.runtime.python.detect import (
            letterbox_geometry, generate_anchors, FasterRCNNTask, pack_bank, unpack_bank_outputs, tail)
        from infer_vision_banners import overlay
        context = letterbox_geometry(photo.height, photo.width)
        height, width = context.net_size
        canvas = np.zeros((320, 320, 3), np.uint8)
        canvas[:height, :width] = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        value = canvas[:, :, ::-1].astype(np.float32)/np.float32(255)
        value = (value-np.asarray([.485, .456, .406], np.float32))/np.asarray([.229, .224, .225], np.float32)
        value = np.ascontiguousarray(value.transpose(2, 0, 1)[None])
        first = forward('parta', {sessions['parta'].get_inputs()[0].name: value})
        task = object.__new__(FasterRCNNTask)
        task.anchors = generate_anchors()
        proposals, count, features = task.middle(first)
        if not count:
            raise ValueError('No model proposals for the new detector input')
        bank = pack_bank(task.pool(proposals, features, count))
        second = forward('partb', {sessions['partb'].get_inputs()[0].name: bank})
        logits, regressions = unpack_bank_outputs(second['cls_packed'], second['reg_packed'])
        boxes, scores, ids = tail(logits, regressions, proposals, count)
        boxes = context.to_original(boxes)
        labels = (native/'samples/vision/fasterrcnn/model/coco91_labels.txt').read_text().splitlines()
        if len(labels) != 91 or labels != row['labels']:
            raise ValueError('Faster R-CNN COCO vocabulary differs from its native labels')
        result_image, selected = overlay(photo, boxes, scores, ids, labels)
        scale = max(1672/photo.width, 941/photo.height)
        crop_width, crop_height = 1672/scale, 941/scale
        crop = [(photo.width-crop_width)/2, (photo.height-crop_height)/2,
                (photo.width+crop_width)/2, (photo.height+crop_height)/2]
        for prediction in selected:
            box = boxes[prediction['prediction_index']]
            if box[0] < crop[0] or box[1] < crop[1] or box[2] > crop[2] or box[3] > crop[3]:
                raise ValueError('Use a wider source photograph to keep every displayed detection inside the banner')
        banner = ImageOps.fit(result_image, (1672, 941), Image.Resampling.LANCZOS)
        detections = [{'class_id': r['class_id'], 'label': r['label'], 'score': r['score'],
                       'box': boxes[r['prediction_index']].tolist()} for r in selected]
        raw.update(proposals=proposals, decoded_boxes=boxes, decoded_scores=scores, decoded_class_ids=ids)
        extra = {'detections': detections, 'confidence_threshold': .5,
                 'visualization': {'method': 'Exact conversion ONNX PartA, native RPN/RoIAlign, packed ONNX PartB and native tail',
                                   'proposals': count, 'rendered_predictions': selected,
                                   'source_crop_xyxy': crop, 'rendered_size': [1672, 941],
                                   'layout': 'full-bleed; corner radius is applied by site CSS'}}
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
    elif kind == 'tracking-overlay' and row['source'] == 'fasterrcnn':
        from samples.vision.fasterrcnn.conversion.reference import SourceFloatDetector
        from samples.vision.fasterrcnn.runtime.python.cli import resolve_selection
        from samples.vision.fasterrcnn.runtime.python.detect import FasterRCNNTask, PairedRuntime, pack_bank, unpack_bank_outputs
        from samples.vision.fasterrcnn.runtime.python.tracking import PersonTrackerTask
        class SourceCallbacks(PairedRuntime):
            def run_parta(self, tensors):
                return forward('parta', tensors)
            def run_partb(self, pooled):
                values = forward('partb', {'bank': pack_bank(pooled)})
                return unpack_bank_outputs(values['cls_packed'], values['reg_packed'])
        detector = SourceFloatDetector.__new__(SourceFloatDetector)
        FasterRCNNTask.__init__(detector, runner=SourceCallbacks(resolve_selection('s100', variant='mbv3-320-native-20261103-o1')))
        tracks = PersonTrackerTask(detector).predict(bgr)
        if not tracks:
            raise ValueError('No actual person tracks on this banner input')
        result_image = photo.copy(); draw = ImageDraw.Draw(result_image)
        for track in tracks:
            box = tuple(track.tlbr); text = f'ID {track.track_id}  {track.score:.2f}'
            draw.rectangle(box, outline=(46, 100, 234), width=4)
            face = font(24, True); width = draw.textlength(text, font=face)
            top = max(0, box[1]-33)
            draw.rectangle((box[0], top, box[0]+width+12, top+32), fill=(46, 100, 234))
            draw.text((box[0]+6, top+2), text, font=face, fill='white')
        banner = ImageOps.fit(result_image, (1672, 941), Image.Resampling.LANCZOS)
        extra = {'tracks': [{'id': t.track_id, 'box': list(t.tlbr), 'score': t.score} for t in tracks],
                 'visualization': {'method': 'Both source Faster R-CNN stages, Native RPN/RoIAlign, person filter and real CPU ByteTrack update',
                                   'rendered_track_ids': [t.track_id for t in tracks], 'rendered_size': [1672, 941], 'layout': 'full-bleed'}}
    elif kind == 'tracking-overlay':
        from samples.vision.bytetrack.conversion.reference import SourceFloatDetector
        from samples.vision.bytetrack.runtime.python.cli import resolve_selection
        from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
        detector = SourceFloatDetector(resolve_selection(row['target'],variant=row['variant']),row['graphs']['model']['path'],threads)
        # Share the checked forward function so every source output is retained.
        detector.infer = lambda feed: forward('model',feed)
        tracks = ByteTrackTask(detector).predict(bgr)
        records = [{'id':t.track_id,'box':list(t.tlbr),'score':t.score} for t in tracks]
        selected = [t for t in records if (t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1]) >= photo.width*photo.height*.005]
        if not selected:
            raise ValueError('The actual model and tracker yielded no visible tracks')
        result_image = photo.copy()
        draw=ImageDraw.Draw(result_image);face=font(24,True)
        for track in selected:
            box=track['box'];label=f"ID {track['id']}  {track['score']:.2f}"
            color=(0,108,238);draw.rectangle(tuple(box),outline=color,width=3)
            x=max(0,min(photo.width-200,box[0]));y=max(0,box[1]-32)
            draw.rectangle((x,y,x+draw.textlength(label,font=face)+18,y+32),fill=color)
            draw.text((x+9,y+2),label,font=face,fill='white')
        banner=ImageOps.fit(result_image,(1672,941),Image.Resampling.LANCZOS)
        extra={'tracks':records,'visualization':{'method':'Source YOLOv5x ONNX, Native raw-F32 person decoding and first actual CPU ByteTrack update',
            'frame_index':1,'rendered_track_ids':[t['id']for t in selected],'rendered_size':[1672,941],
            'layout':'full-bleed; actual first-frame IDs, not a MOT accuracy evaluation'}}
    elif kind in ('ocr-detection','ocr-recognition'):
        from samples.vision.paddle_ocr.runtime.python.cli import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.ocr import OCRPipeline
        pair = resolve_pair(row['target'],det_asset_id=row['detector_asset_id'],rec_asset_id=row['recognizer_asset_id'])
        pipeline=OCRPipeline(pair,lambda _: {},lambda _: {})
        if kind=='ocr-detection':
            values=forward('model',pipeline.prepare_detection(bgr))
            result=pipeline.postprocess_detection(values,bgr)
            if not result.boxes:raise ValueError('No actual OCR text boxes for this input')
            result_image=photo.copy();draw=ImageDraw.Draw(result_image)
            for box in result.boxes:
                points=[tuple(v) for v in box.tolist()];draw.line(points+[points[0]],fill=(0,108,238),width=4)
            extra={'detection':{'boxes':[b.tolist()for b in result.boxes]},
                'visualization':{'method':'Official BGR normalization, exact source detector probability map and Native DB polygons/crops',
                                 'rendered_boxes':[b.tolist()for b in result.boxes]}}
        else:
            roi=row['word_roi'];x0,y0,x1,y1=roi
            if not (0<=x0<x1<=photo.width and 0<=y0<y1<=photo.height):raise ValueError('Word crop outside input photograph')
            values=forward('model',pipeline.prepare_recognition(bgr[y0:y1,x0:x1]))
            text=pipeline.decode_recognition(values)
            if not text:raise ValueError('Actual recognizer returned an empty word')
            result_image=photo.copy();draw=ImageDraw.Draw(result_image);face=font(46,True)
            width=draw.textlength(text,font=face);draw.rectangle((54,806,94+width,890),fill=(0,108,238))
            draw.text((74,817),text,font=face,fill='white')
            extra={'recognition':{'text':text,'ctc_indices':next(iter(values.values()))[0].argmax(axis=-1).tolist()},
                'visualization':{'method':'Official BGR aspect resize and right padding, full source logits, Native 97-class CTC decode',
                                 'word_roi':roi,'roi_source':'manual inference-input crop, not a detector output','rendered_text':text}}
        banner=ImageOps.fit(result_image,(1672,941),Image.Resampling.LANCZOS)
    elif kind == 'speech-transcript':
        import soundfile as sf
        from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
        from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline,TensorNames
        from samples.speech.paraformer.conversion.export import signature
        audio_path=native/row['audio_input']
        committed=subprocess.check_output(['git','-C',str(native),'show',row['native_commit']+':'+row['audio_input']])
        if hashlib.sha256(committed).hexdigest()!=digest(audio_path):raise ValueError('Audio differs from pinned source input')
        support={}
        for name in ('samples/speech/paraformer/model/am.mvn','samples/speech/paraformer/tests/fixtures/published-tokens.json'):
            committed=subprocess.check_output(['git','-C',str(native),'show',row['native_commit']+':'+name])
            if hashlib.sha256(committed).hexdigest()!=digest(native/name):raise ValueError('Speech support file differs from pinned source')
            support[name]=digest(native/name)
        wave,rate=sf.read(audio_path,dtype='float32')
        features=ParaformerFrontend(native/'samples/speech/paraformer/model/am.mvn').pre_process(wave,rate)
        vocabulary=json.loads((native/'samples/speech/paraformer/tests/fixtures/published-tokens.json').read_text())
        names=TensorNames(signature('encoder','input')[0][0],signature('encoder','output')[0][0],
            signature('predictor','input')[0][0],signature('predictor','output')[0][0],signature('predictor','output')[1][0],
            *[v[0]for v in signature('decoder','input')],signature('decoder','output')[0][0])
        pipeline=ParaformerPipeline(*[(lambda feed,role=role:forward(role,feed))for role in ('encoder','predictor','decoder')],names,vocabulary)
        result=pipeline.predict(features.tensor,features.valid_frames)
        if not result.decoder_executed or not result.text:raise ValueError('No actual speech decoder readout')
        result_image=ImageOps.fit(photo,(1672,941),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(result_image)
        draw.rectangle((0,680,1672,941),fill=(245,248,252))
        mono=wave.mean(axis=1)if wave.ndim==2 else wave
        edges=np.linspace(0,len(mono),701,dtype=int)
        for i in range(700):
            chunk=mono[edges[i]:edges[i+1]];amplitude=float(np.max(np.abs(chunk)))if len(chunk)else 0
            y=min(60,round(amplitude/max(float(np.max(np.abs(mono))),1e-8)*60))
            x=65+round(i*1542/700);draw.line((x,758-y,x,758+y),fill=(0,108,238),width=2)
        face=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',46)
        draw.text((65,842),result.text,font=face,fill=(25,31,41))
        banner=result_image
        extra={'native_support_files':support,'audio':{'input':row['audio_input'],'sha256':digest(audio_path),'sample_rate':int(rate),'sample_count':len(wave)},
               'recognition':{'text':result.text,'token_ids':list(result.token_ids),'token_count':result.token_count},
               'visualization':{'method':'Actual Native WAV frontend, all three source ONNX stages, CPU CIF and greedy transcription; generated microphone photo is a visual backdrop',
                                'rendered_text':result.text,'waveform':'actual audio amplitude, not generated waveform','photo_role':'illustration backdrop; not the ASR inference input'}}
    elif kind == 'plate-readout':
        from samples.vision.lprnet.model.prepare_input import prepare_bgr
        from samples.vision.lprnet.runtime.python.lprnet import decode_plate
        x0, y0, x1, y1 = row['plate_roi']
        if not (0 <= x0 < x1 <= photo.width and 0 <= y0 < y1 <= photo.height):
            raise ValueError('Plate crop is outside the input photograph')
        value = prepare_bgr(bgr[y0:y1, x0:x1])
        values = forward('model', {session.get_inputs()[0].name: value})
        logits, = values.values()
        if logits.shape != (1, 68, 18, 1):
            raise ValueError('Expected complete LPRNet logits')
        plate = decode_plate(logits.reshape(68, 18))
        if not plate:
            raise ValueError('The model produced an empty plate readout')
        result_image = ImageOps.fit(photo, (1672, 941), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(result_image)
        face = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 48)
        width = round(draw.textlength(plate, font=face))
        draw.rectangle((68, 806, 108 + width, 890), fill=(0, 95, 205))
        draw.text((88, 810), plate, font=face, fill='white')
        banner = result_image
        extra = {'recognition': {'plate': plate, 'ctc_indices': logits[0, :, :, 0].argmax(axis=0).tolist()},
                 'visualization': {'method': 'Native CTC readout of the complete source logits on a generated plate crop',
                                   'plate_roi': row['plate_roi'],
                                   'roi_source': 'manual input crop; no detector prediction',
                                   'rendered_text': plate}}
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
    elif kind == 'semantic-wipe' and row['source'] == 'lanenet':
        from samples.vision.lanenet.runtime.python.lanenet import image_to_tensor, LaneNetSegmenter
        from utils.py_utils.runtime_meta import RuntimeMetadata
        values = forward('model', {session.get_inputs()[0].name: image_to_tensor(bgr)})
        metadata = RuntimeMetadata.from_mapping({'input_names': ['images'],
            'input_shapes': {'images': [1, 3, 256, 512]}, 'input_dtypes': {'images': 'float32'},
            'output_names': list(values), 'output_shapes': {n: list(v.shape) for n,v in values.items()},
            'output_dtypes': {n: str(v.dtype) for n,v in values.items()}})
        task = LaneNetSegmenter.__new__(LaneNetSegmenter)
        task.binding = SimpleNamespace(metadata=metadata, input_name='images',
                                       embedding_name='instance_seg_logits', binary_name='binary_seg_pred')
        labels = task.postprocess(values).binary
        mask = cv2.resize(labels, photo.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any(): raise ValueError('No predicted lane pixels on the source input')
        result_image = mask_overlay(photo, mask)
        banner = ImageOps.fit(result_image, (1672, 941), Image.Resampling.LANCZOS)
        raw['lane_mask'] = labels
        extra = {'visualization': {'method': 'All three source heads retained; Native discrete lane-label decoding and actual mask overlay, without invented instance IDs',
                                   'foreground_pixels': int(mask.sum()), 'mask_grid_shape': list(labels.shape),
                                   'rendered_size': [1672, 941], 'layout': 'full-bleed'}}
    elif kind == 'semantic-wipe' and row['source'] == 'unet':
        from samples.vision.unet.runtime.python.vgg import preprocess, postprocess
        value, context = preprocess(source)
        values = forward('model', {'images': value})
        labels = postprocess(values['logits'], context)
        raw['class_ids'] = labels
        palette = np.zeros((21, 3), dtype=np.uint8)
        for index in range(21):
            code = index
            for bit in range(8):
                for channel in range(3):
                    palette[index, channel] |= ((code >> channel) & 1) << (7-bit)
                code >>= 3
        pixels = np.asarray(photo).copy()
        foreground = labels != 0
        pixels[foreground] = np.uint8(pixels[foreground]*.52 + palette[labels[foreground]]*.48)
        result_image = Image.fromarray(pixels)
        banner = ImageOps.fit(result_image, (1672, 941), Image.Resampling.LANCZOS)
        classes, counts = np.unique(labels, return_counts=True)
        extra = {'visualization': {'method': 'Native source PIL letterbox; all VOC21 logits, softmax, unpad, resize scores, argmax; standard VOC palette on actual foreground',
                                   'class_pixels': {str(int(i)): int(n) for i, n in zip(classes, counts)},
                                   'alpha': .48}}
    elif kind == 'semantic-wipe':
        from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegSegmenter
        from samples.vision.pp_liteseg.runtime.python.cli import colorize, CITYSCAPES_CLASS_NAMES
        from utils.py_utils.runtime_meta import RuntimeMetadata
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        metadata = RuntimeMetadata.from_mapping({
            'input_names': [inp.name], 'input_shapes': {inp.name: inp.shape},
            'input_dtypes': {inp.name: 'float32'},
            'output_names': [out.name], 'output_shapes': {out.name: out.shape},
            'output_dtypes': {out.name: 'float32'},
        })
        task = PPLiteSegSegmenter.__new__(PPLiteSegSegmenter)
        task.binding = SimpleNamespace(selection=SimpleNamespace(float_logits=True),
                                       input_name=inp.name, output_name=out.name, metadata=metadata)
        prepared = task.preprocess(bgr)
        values = forward('model', prepared.tensors)
        labels = task.postprocess(values[out.name])
        raw['class_ids'] = labels
        colors = Image.fromarray(colorize(labels)[:, :, ::-1]).resize(photo.size, Image.Resampling.NEAREST)
        result_image = Image.blend(photo, colors, .52)
        base = ImageOps.fit(photo, (1672, 941), Image.Resampling.LANCZOS)
        overlay = ImageOps.fit(result_image, base.size, Image.Resampling.LANCZOS)
        banner = base.copy()
        cut = round(base.width*.45)
        banner.paste(overlay.crop((cut, 0, base.width, base.height)), (cut, 0))
        ImageDraw.Draw(banner).line((cut, 0, cut, base.height), fill='white', width=2)
        classes, counts = np.unique(labels, return_counts=True)
        extra = {'visualization': {'method': 'Native normalized RGB preprocessing and CPU logits argmax; Cityscapes palette overlay; aligned full-scene wipe',
                                   'class_pixels': {CITYSCAPES_CLASS_NAMES[int(i)]: int(n) for i, n in zip(classes, counts)},
                                   'alpha': .52, 'original_fraction': .45}}
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
