# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""ViT-B/16 ImageNet-1k F32 featuremap inference, separate from CIFAR-10 assets."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
from utils.py_utils.converted_featuremap import Runner,select

VARIANT='tv-b16-imagenet1k-224-featuremap-int16-cpuptq-o1-20261105'
SAMPLE=Path(__file__).resolve().parents[2]


def preprocess(source):
    """Use the official weights' PIL bilinear resize and center-crop policy."""
    if isinstance(source,(str,Path)):
        with Image.open(source)as image:rgb=image.convert('RGB')
    elif isinstance(source,Image.Image):rgb=source.convert('RGB')
    else:raise TypeError('Supply an image path or PIL RGB image')
    width,height=rgb.size
    if min(width,height)<=0:raise ValueError('Empty image')
    size=(256,int(256*height/width))if width<=height else(int(256*width/height),256)
    rgb=rgb.resize(size,Image.Resampling.BILINEAR)
    left,top=int(round((rgb.width-224)/2)),int(round((rgb.height-224)/2))
    rgb=rgb.crop((left,top,left+224,top+224))
    value=np.asarray(rgb).astype(np.float32).transpose(2,0,1)/np.float32(255)
    mean=np.asarray([.485,.456,.406],np.float32)[:,None,None]
    std=np.asarray([.229,.224,.225],np.float32)[:,None,None]
    return np.ascontiguousarray(((value-mean)/std)[None])


def predictions(logits, labels, top_k=5):
    if logits.shape!=(1,1000)or logits.dtype!=np.float32 or not np.isfinite(logits).all():
        raise ValueError('Expected F32 ImageNet-1k logits [1,1000]')
    if len(labels)!=1000 or not 1<=top_k<=1000:raise ValueError('Wrong label count or top-k')
    values=logits[0].astype(np.float64);score=np.exp(values-values.max());score/=score.sum()
    ids=np.argsort(-score,kind='stable')[:top_k]
    return [{'class_id':int(i),'label':labels[i],'probability':float(score[i])}for i in ids]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['s100'],default='s100')
    p.add_argument('--variant',default=VARIANT)
    p.add_argument('--asset-id');p.add_argument('--model-path',type=Path)
    p.add_argument('--test-img',type=Path,required=True)
    p.add_argument('--backend',choices=['board','source-onnx'],default='board')
    p.add_argument('--source-onnx',type=Path);p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();selection=select(SAMPLE,a.target,a.variant,asset_id=a.asset_id,model_path=a.model_path)
    if a.dry_run:
        print(json.dumps({'target':selection.target,'asset_id':selection.asset.reference,'profile':selection.profile}));return 0
    labels=json.loads((SAMPLE/'model/imagenet1k_labels.json').read_text())
    outputs=Runner(selection,backend=a.backend,source_onnx=a.source_onnx)(preprocess(a.test_img))
    print(json.dumps({'backend':a.backend,'target':selection.target,'asset_id':selection.asset.reference,
                      'topk':predictions(outputs['logits'],labels,a.top_k)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
