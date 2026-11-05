# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Independent VGG16/VOC21 UNet featuremap variant with source letterboxing."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from PIL import Image
from utils.py_utils.converted_featuremap import Runner,select
SAMPLE=Path(__file__).resolve().parents[2]
VARIANT='vgg16-voc21-letterbox-512-featuremap-int16-cpuptq-o1-20261105-r02'


def preprocess(source):
    if isinstance(source,(str,Path)):
        with Image.open(source)as image:rgb=image.convert('RGB')
    elif isinstance(source,Image.Image):rgb=source.convert('RGB')
    else:raise TypeError('Expected an image path or PIL image')
    width,height=rgb.size;scale=min(512/width,512/height);nw,nh=int(width*scale),int(height*scale)
    canvas=Image.new('RGB',(512,512),(128,128,128));left,top=(512-nw)//2,(512-nh)//2
    canvas.paste(rgb.resize((nw,nh),Image.Resampling.BICUBIC),(left,top))
    value=np.asarray(canvas).astype(np.float32)/np.float32(255)
    return np.ascontiguousarray(value.transpose(2,0,1)[None]),{'original_size':[height,width],'resized':[nh,nw],'padding':[top,left]}


def postprocess(logits,context):
    if logits.dtype!=np.float32 or logits.shape!=(1,21,512,512)or not np.isfinite(logits).all():
        raise ValueError('Expected complete F32 VOC21 logits [1,21,512,512]')
    score=logits[0].transpose(1,2,0);score=np.exp(score-score.max(axis=2,keepdims=True));score/=score.sum(axis=2,keepdims=True)
    nh,nw=context['resized'];top,left=context['padding'];height,width=context['original_size']
    restored=cv2.resize(score[top:top+nh,left:left+nw],(width,height),interpolation=cv2.INTER_LINEAR)
    return restored.argmax(axis=2).astype(np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['s100'],default='s100');p.add_argument('--variant',default=VARIANT)
    p.add_argument('--asset-id');p.add_argument('--model-path',type=Path);p.add_argument('--test-img',type=Path,required=True)
    p.add_argument('--backend',choices=['board','source-onnx'],default='board');p.add_argument('--source-onnx',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();selection=select(SAMPLE,a.target,a.variant,asset_id=a.asset_id,model_path=a.model_path)
    if a.dry_run:print(json.dumps({'asset_id':selection.asset.reference,'profile':selection.profile}));return 0
    value,context=preprocess(a.test_img);outputs=Runner(selection,backend=a.backend,source_onnx=a.source_onnx)(value)
    labels=postprocess(outputs['logits'],context)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(labels).save(a.output)
    ids,counts=np.unique(labels,return_counts=True)
    print(json.dumps({'backend':a.backend,'asset_id':selection.asset.reference,'shape':list(labels.shape),'class_pixels':{str(i):int(n)for i,n in zip(ids,counts)}}));return 0


if __name__=='__main__':raise SystemExit(main())
