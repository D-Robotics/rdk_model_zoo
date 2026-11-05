# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""FCOS ResNet50-FPN: explicit RGB featuremap input and complete dense heads.

This independent 91-slot COCO model does not reuse the historical EfficientNet
heads, anchor centers or output dequantization rules.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from utils.py_utils.converted_featuremap import Runner,select

VARIANT='tv-r50-fpn-coco91-512-featuremap-int16-cpuptq-o1-20261105'
SAMPLE=Path(__file__).resolve().parents[2]
STRIDES=(8,16,32,64,128)
COUNTS=tuple((512//s)**2 for s in STRIDES)


def preprocess(image):
    if isinstance(image,(str,Path)):image=cv2.imread(str(image))
    if not isinstance(image,np.ndarray)or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Expected a nonempty BGR uint8 image')
    height,width=image.shape[:2]
    if not min(height,width):raise ValueError('Empty image')
    rgb=cv2.cvtColor(cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    value=rgb.astype(np.float32).transpose(2,0,1)[None]/np.float32(255)
    mean=np.asarray([.485,.456,.406],np.float32)[None,:,None,None]
    std=np.asarray([.229,.224,.225],np.float32)[None,:,None,None]
    return np.ascontiguousarray((value-mean)/std),(height,width)


def _sigmoid(value):
    result=np.empty_like(value);positive=value>=0
    result[positive]=1/(1+np.exp(-value[positive]))
    e=np.exp(value[~positive]);result[~positive]=e/(1+e)
    return result


def _nms(boxes,scores,labels,threshold):
    order=np.argsort(-scores,kind='stable');kept=[]
    while order.size:
        first=order[0];kept.append(first);rest=order[1:]
        left=np.maximum(boxes[first,:2],boxes[rest,:2]);right=np.minimum(boxes[first,2:],boxes[rest,2:])
        overlap=np.maximum(right-left,0).prod(axis=1)
        area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        union=area[first]+area[rest]-overlap
        iou=np.divide(overlap,union,out=np.zeros_like(overlap),where=union>0)
        order=rest[~((labels[rest]==labels[first])&(iou>threshold))]
    return np.asarray(kept,dtype=np.int64)


def decode(outputs,original_size,score_threshold=.2,nms_threshold=.6):
    shapes={'cls_logits':(1,5456,91),'bbox_regression':(1,5456,4),'bbox_ctrness':(1,5456,1)}
    if set(outputs)!=set(shapes):raise ValueError('All three FCOS dense outputs are required')
    for name,shape in shapes.items():
        value=outputs[name]
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Wrong FCOS head '+name)
    if not 0<score_threshold<1 or not 0<nms_threshold<=1:raise ValueError('Invalid detection threshold')
    boxes=[];scores=[];labels=[];offset=0
    for stride,count in zip(STRIDES,COUNTS,strict=True):
        cls=outputs['cls_logits'][0,offset:offset+count]
        center=outputs['bbox_ctrness'][0,offset:offset+count]
        regression=outputs['bbox_regression'][0,offset:offset+count]
        score=np.sqrt(_sigmoid(cls)*_sigmoid(center)).ravel()
        ids=np.flatnonzero(score>score_threshold)
        ids=ids[np.argsort(-score[ids],kind='stable')[:1000]]
        points=ids//91;category=ids%91;grid=512//stride
        centers=np.column_stack((points%grid,points//grid)).astype(np.float32)*np.float32(stride)
        distance=regression[points]*np.float32(stride)
        box=np.concatenate((centers-distance[:,:2],centers+distance[:,2:]),axis=1)
        np.clip(box,0,512,out=box)
        boxes.append(box);scores.append(score[ids]);labels.append(category)
        offset+=count
    boxes=np.concatenate(boxes);scores=np.concatenate(scores);labels=np.concatenate(labels)
    keep=_nms(boxes,scores,labels,nms_threshold)[:100]
    boxes,scores,labels=boxes[keep],scores[keep],labels[keep]
    height,width=original_size
    boxes=boxes*np.asarray([width/512,height/512,width/512,height/512],np.float32)
    return {'boxes':boxes,'scores':scores,'labels':labels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['s100'],default='s100');p.add_argument('--variant',default=VARIANT)
    p.add_argument('--asset-id');p.add_argument('--model-path',type=Path);p.add_argument('--test-img',type=Path,required=True)
    p.add_argument('--backend',choices=['board','source-onnx'],default='board');p.add_argument('--source-onnx',type=Path)
    p.add_argument('--conf-thres',type=float,default=.2);p.add_argument('--iou-thres',type=float,default=.6);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();selection=select(SAMPLE,a.target,a.variant,asset_id=a.asset_id,model_path=a.model_path)
    if a.dry_run:
        print(json.dumps({'target':selection.target,'asset_id':selection.asset.reference,'profile':selection.profile}));return 0
    value,size=preprocess(a.test_img);outputs=Runner(selection,backend=a.backend,source_onnx=a.source_onnx)(value)
    result=decode(outputs,size,a.conf_thres,a.iou_thres)
    print(json.dumps({'backend':a.backend,'asset_id':selection.asset.reference,**{k:v.tolist()for k,v in result.items()}}));return 0


if __name__=='__main__':raise SystemExit(main())
