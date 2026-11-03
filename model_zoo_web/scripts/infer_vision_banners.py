#!/usr/bin/env python3
"""Visualize real pinned ONNX depth/features/matting/detection/segmentation outputs.

Generated images are inputs only. Native sample preparation and decode stages
are reused without loading a board SDK. Large raw forward tensors stay in the
Workbench run; the site keeps compact actual predictions and their provenance.
"""
from __future__ import annotations
import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

from infer_classification_banners import digest, file_record, font, write_json


def pair(left, right, labels):
    canvas = Image.new('RGB', (1672,941), (236,236,238))
    draw = ImageDraw.Draw(canvas)
    for x, image, label in ((40,left,labels[0]), (856,right,labels[1])):
        photo = ImageOps.fit(image.convert('RGB'), (776,861), Image.Resampling.LANCZOS)
        mask = Image.new('L', photo.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,775,860),radius=40,fill=255)
        canvas.paste(photo,(x,40),mask)
        face = font(27)
        box = draw.textbbox((0,0),label,font=face)
        width = box[2]-box[0]+28
        draw.rounded_rectangle((x+25,65,x+25+width,113),radius=12,fill='white')
        draw.text((x+39,69),label,font=face,fill=(24,24,28))
    return canvas


def overlay(photo, boxes, scores, ids, labels, masks=None):
    canvas = photo.convert('RGB').copy()
    width,height = canvas.size
    selected=[]
    for i in np.argsort(-scores,kind='stable'):
        box=boxes[i]
        if scores[i] < .5 or min(box[:2]) < 8 or box[2] > width-8 or box[3] > height-8:
            continue
        if max(0,box[2]-box[0])*max(0,box[3]-box[1]) < width*height*.01:
            continue
        selected.append(int(i))
        if len(selected)==6:break
    if not selected:
        raise ValueError('No confident complete foreground instances to render')
    if masks is not None:
        layer=np.zeros((height,width,4),np.uint8)
        for i in selected:
            layer[masks[i]>0,:3]=(49,203,220);layer[masks[i]>0,3]=92
        canvas=Image.alpha_composite(canvas.convert('RGBA'),Image.fromarray(layer)).convert('RGB')
    draw=ImageDraw.Draw(canvas);scale=width/1672;face=font(max(18,round(26*scale)))
    rendered=[]
    for i in selected:
        x1,y1,x2,y2=map(float,boxes[i]);label=labels[int(ids[i])]
        if masks is None:
            draw.rounded_rectangle((round(x1),round(y1),round(x2),round(y2)),
                                   radius=max(6,round(13*scale)),outline='white',width=max(2,round(4*scale)))
        else:
            contours,_=cv2.findContours(masks[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points=[tuple(map(int,p)) for p in contour.reshape(-1,2)]
                if len(points)>2:draw.line(points+[points[0]],fill='white',width=max(2,round(2.2*scale)),joint='curve')
        text=f'{label} {round(float(scores[i])*100)}%'
        ink=draw.textbbox((0,0),text,font=face);tw=ink[2]-ink[0]+14;th=ink[3]-ink[1]+12
        tx=max(4,min(width-tw-4,round((x1+x2-tw)/2)));ty=max(4,min(height-th-4,round(y1-th/2)))
        draw.rounded_rectangle((tx,ty,tx+tw,ty+th),radius=8,fill='white')
        draw.text((tx+7-ink[0],ty+6-ink[1]),text,font=face,fill=(20,20,20))
        rendered.append({'prediction_index':i,'class_id':int(ids[i]),'label':label,'score':float(scores[i])})
    return canvas,rendered


def run(row, source, output, native, raw_dir, threads):
    graph=Path(row['onnx_path'])
    if digest(graph)!=row['onnx_sha256']:raise ValueError('Float graph identity differs')
    photo=Image.open(source).convert('RGB');bgr=np.asarray(photo)[:,:,::-1].copy()
    kind=row['kind'];context=None
    if kind=='depth':
        from samples.vision.depth_anything_v2.conversion.export_small import preprocess
        value=preprocess(bgr)
    elif kind=='embedding':
        from samples.vision.dinov2.runtime.python.embedding import prepare_image
        prepared=prepare_image(bgr,SimpleNamespace(input_name='input',input_shape=(1,3,224,224)))
        value=prepared.tensors['input'];context=prepared.context
    elif kind=='matting':
        from samples.vision.modnet.runtime.python.modnet import preprocess_image
        value,context=preprocess_image(bgr)
    else:
        from samples.vision.ultralytics_yolo.runtime.python.detect import resize_with_transform
        pixels,context=resize_with_transform(bgr,(640,640),0 if kind=='detection' else 1)
        value=np.ascontiguousarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None],dtype=np.float32)/255
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    if kind=='instance-segmentation':options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs();names=[x.name for x in session.get_outputs()]
    if len(inputs)!=1 or inputs[0].shape!=list(value.shape):raise ValueError('Input contract differs from actual float graph')
    started=time.perf_counter();raw_values=session.run(names,{inputs[0].name:value});elapsed=time.perf_counter()-started
    raw=dict(zip(names,raw_values));assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in raw_values)
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,output/'source.webp')
    raw_dir.mkdir(parents=True,exist_ok=True);raw_path=raw_dir/(row['stem']+'-forward.npz')
    if raw_path.exists():raise FileExistsError('Use a fresh inference attempt')
    np.savez_compressed(raw_path,**raw)
    model={'catalog_id':row['id'],'name':row['model_name'],'variant':row['model_size'],
           'source_checkpoint_sha256':row['checkpoint_sha256'],'onnx_sha256':row['onnx_sha256']}
    prediction={'schema_version':1,'task':kind,'model':model,
                'image':{'sha256':digest(source),'dimensions':list(photo.size)}}
    render={'style':'Existing RDK Model Zoo photographic banners','dimensions':[1672,941]}
    arrays=raw
    if kind=='depth':
        depth=raw['depth'][0]
        if depth.shape!=(518,686):raise ValueError('Unexpected relative-depth output')
        restored=cv2.resize(depth,photo.size,interpolation=cv2.INTER_LINEAR)
        lo,hi=np.percentile(restored,[2,98]);norm=np.clip((restored-lo)/max(hi-lo,1e-8),0,1)
        colored=cv2.cvtColor(cv2.applyColorMap(np.uint8(norm*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        banner=pair(photo,Image.fromarray(colored),('RGB','Relative depth'))
        prediction['visualization']={'method':'2nd–98th percentile TURBO color scale','values':'relative native depth; no metric distance',
                                     'native_shape':list(depth.shape),'minimum':float(depth.min()),'maximum':float(depth.max())}
    elif kind=='embedding':
        features=raw['patch_feat'][0]
        if features.shape!=(256,384) or raw['cls_feat'].shape!=(1,384):raise ValueError('Unexpected DINOv2 dual output')
        centered=features.astype(np.float64)-features.mean(axis=0,dtype=np.float64)
        _,_,basis=np.linalg.svd(centered,full_matrices=False);basis=basis[:3]
        for axis in range(3):
            if basis[axis,np.argmax(np.abs(basis[axis]))]<0:basis[axis]*=-1
        projected=centered@basis.T;lo,hi=np.percentile(projected,[2,98],axis=0)
        colors=np.uint8(np.clip((projected-lo)/np.maximum(hi-lo,1e-8),0,1).reshape(16,16,3)*255)
        feature_image=Image.fromarray(colors).resize(photo.size,Image.Resampling.NEAREST)
        banner=pair(photo,feature_image,('RGB','Patch features · PCA'))
        arrays={**raw,'pca_basis':basis.astype(np.float32),'patch_rgb':colors}
        prediction['visualization']={'method':'Per-image three-component PCA of the actual 16x16 patch features; canonicalized basis signs',
                                     'semantics':'False-color feature projection, not class labels or segmentation',
                                     'patch_shape':[16,16,384],'cls_shape':[1,384]}
    elif kind=='matting':
        from samples.vision.modnet.runtime.python.modnet import MODNetMatting
        matte=MODNetMatting.postprocess(None,raw['matte'],context)
        alpha=matte.astype(np.float32)[:,:,None]/255
        foreground=np.uint8(np.clip(np.asarray(photo)*alpha+np.array([229,240,247])*(1-alpha),0,255))
        banner=pair(photo,Image.fromarray(foreground),('RGB','Model alpha matte'))
        arrays={**raw,'restored_alpha':matte}
        prediction['visualization']={'method':'Native MODNet postprocess; composite actual alpha onto light blue',
                                     'alpha_min':int(matte.min()),'alpha_max':int(matte.max())}
    else:
        if kind=='detection':
            from samples.vision.yolov5.runtime.python.detection import decode_x5,scale_coords_back
            from samples.vision.yolov5.runtime.python.cli import ANCHORS
            anchors=np.asarray(ANCHORS,dtype=np.float32).reshape(3,3,2)
            ordered=sorted(names,key=lambda name:-raw[name].shape[1])
            boxes,scores,ids=decode_x5(raw,ordered,640,anchors,.5,.45)
            boxes=scale_coords_back(boxes,photo.width,photo.height,640,640,0)
            labels=(native/'datasets/coco/coco_classes.names').read_text().splitlines()
            if len(labels)!=80:raise ValueError('Unexpected COCO vocabulary')
            masks=None
        else:
            from samples.vision.yoloe.runtime.python.yoloe import decode_x5,PF11Contract
            boxes,scores,ids,masks=decode_x5(raw,PF11Contract(),context,.5,.7)
            labels=(native/'samples/vision/yoloe/test_data/classes.names').read_text().splitlines()
            if len(labels)!=4585:raise ValueError('Unexpected PF vocabulary')
        banner,rendered=overlay(photo,boxes,scores,ids,labels,masks)
        prediction['detections']=[{'class_id':int(c),'label':labels[int(c)],'score':float(s),'box_xyxy':list(map(float,b))}
                                  for b,s,c in zip(boxes,scores,ids)]
        render.update(rendered_predictions=rendered,confidence_threshold=.5,
                      policy='Up to six actual foreground results; omit source-edge boxes and boxes smaller than 1% of the image')
        arrays={'boxes':boxes,'scores':scores,'class_ids':ids}
        if masks is not None:arrays['masks']=masks
        prediction['postprocessing']='Native X5 reference decoding; host float input, not deployment BIN inference'
    np.savez_compressed(output/'outputs.npz',**arrays)
    banner.save(output/'banner.webp','WEBP',quality=92,method=6)
    write_json(output/'predictions.json',prediction)
    receipt={'schema_version':1,'kind':'host-float-model-inference-banner','status':'passed',
             'model':{**model,'source_weight_url':row['checkpoint_url'],'checkpoint_revision':row['checkpoint_revision'],
                      'native_source_commit':row['native_commit']},
             'runtime':{'framework':'ONNX Runtime','version':ort.__version__,'provider':'CPUExecutionProvider','threads':threads,
                        'single_inference_wall_seconds':elapsed,'timing_scope':'host reference only; not a benchmark'},
             'input':{'name':inputs[0].name,'shape':list(value.shape),'dtype':str(value.dtype),
                      'tensor_sha256':hashlib.sha256(value.tobytes()).hexdigest(),
                      'geometry':asdict(context) if context is not None else 'Native fixed-size source preprocessing'},
             'native_forward_outputs':[{'name':name,'shape':list(v.shape),'dtype':str(v.dtype)} for name,v in raw.items()],
             'raw_forward_retained':{'run_relative_file':'raw/'+raw_path.name,'sha256':digest(raw_path),'size_bytes':raw_path.stat().st_size},
             'render':render,'files':{name:file_record(output/name) for name in ('source.webp','outputs.npz','predictions.json','banner.webp')},
             'board_execution':'not-run','dataset_accuracy':'not-run',
             'note':'Host float reference from the pinned conversion ONNX; visualizations are computed from actual outputs.',
             'renderer_sha256':digest(__file__)}
    write_json(output/'provenance.json',receipt)
    return {'stem':row['stem'],'status':'passed','task':kind,'outputs':receipt['native_forward_outputs']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','native-root','inputs','output','raw-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stem',action='append',default=[]);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();sys.path.insert(0,str(a.native_root.resolve()))
    import json
    for row in json.loads(a.plan.read_text()):
        if a.stem and row['stem'] not in a.stem:continue
        source=a.inputs/row['stem']/'source.webp'
        if source.is_file():print(json.dumps(run(row,source,a.output/row['stem'],a.native_root,a.raw_dir,a.threads)),flush=True)


if __name__=='__main__':main()
