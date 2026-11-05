#!/usr/bin/env python3
"""Prepare target-specific RGB/NCHW float32 classification calibration tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def prepare(images, output, count=200, *, target='x5', normalization=None):
    import numpy as np
    from PIL import Image
    if target not in ('x5', 's100', 's100p', 's600'):
        raise ValueError('unknown calibration target')
    if target != 'x5':
        if not isinstance(normalization, dict) or set(normalization) != {'mean', 'std'}:
            raise ValueError('S calibration requires the exact checkpoint mean and std')
        mean = np.asarray(normalization['mean'], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(normalization['std'], dtype=np.float32).reshape(3, 1, 1)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('normalization must be finite with positive standard deviations')
    root=Path(__file__).resolve().parents[3]
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
        destination=output/f'{index:04d}.rgb'
        if target == 'x5':
            data.tofile(destination)
        else:
            # hb_compile 3.5.3 loads ONNX-domain tensors without applying the
            # NV12 runtime mean/scale. X5 Mapper uses raw RGB calibration.
            destination=output/f'{index:04d}.npy'
            np.save(destination, ((data / 255 - mean) / std)[None], allow_pickle=False)
        rows.append({'name':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'tensor_filename':destination.name,
                     'tensor_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    manifest={'count':count,'shape':[1,3,224,224],'dtype':'float32',
        'range':[0,255] if target == 'x5' else None,
        'tensor_domain':'raw-rgb' if target == 'x5' else 'normalized-onnx-input',
        'normalization':normalization if target != 'x5' else 'fused in mapper; not applied to calibration tensors',
        'geometry':'PIL bicubic shortest edge 256, rounded center crop 224',
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':rows}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--target', choices=('x5','s100','s100p','s600'), default='x5')
    parser.add_argument('--normalization-json', type=Path, help='Export receipt containing the checkpoint normalization; required for S')
    args=parser.parse_args()
    normalization = json.loads(args.normalization_json.read_text())['normalization'] if args.normalization_json else None
    result=prepare(args.images,args.output,args.count,target=args.target,normalization=normalization)
    print(json.dumps({'status':'prepared','count':result['count'],'shape':result['shape']}))


if __name__=='__main__':main()
