#!/usr/bin/env python3
"""Prepare unnormalized RGB/NCHW float32 calibration tensors for the Atto D2 recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def prepare(images, output, count=200):
    import numpy as np
    from PIL import Image
    root=Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from utils.py_utils.tensor_io import resize_shorter_center_crop
    paths=sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))[:count]
    if count <= 0 or len(paths)!=count:raise ValueError('The frozen image set does not contain the requested count')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,path in enumerate(paths):
        with Image.open(path) as image:
            image,_,_=resize_shorter_center_crop(image.convert('RGB'),224,256)
            data=np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()
        target=output/f'{index:04d}.rgb';data.tofile(target)
        rows.append({'name':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'tensor_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest={'count':count,'shape':[1,3,224,224],'dtype':'float32','range':[0,255],
        'normalization':'fused in mapper; not applied to calibration tensors',
        'geometry':'PIL bicubic shortest edge 256, rounded center crop 224',
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':rows}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=200)
    args=parser.parse_args()
    result=prepare(args.images,args.output,args.count)
    print(json.dumps({'status':'prepared','count':result['count'],'shape':result['shape']}))


if __name__=='__main__':main()
