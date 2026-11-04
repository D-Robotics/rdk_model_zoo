# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare distinct real-image PP-OCRv3 normalized BGR NPY calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from samples.vision.paddle_ocr.conversion.preprocess_v3 import prepare


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('det','rec'),required=True)
    p.add_argument('--images',type=Path,required=True,help='Detector images or already cropped recognition words')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=100)
    a=p.parse_args(argv)
    paths=sorted(v for v in a.images.iterdir()if v.suffix.lower()in ('.png','.jpg','.jpeg','.bmp'))
    if a.count<1 or len(paths)<a.count:raise ValueError('Insufficient distinct real calibration images')
    rng=np.random.default_rng(42);paths=[paths[int(i)]for i in rng.choice(len(paths),a.count,replace=False)]
    a.output.mkdir(parents=True,exist_ok=False);rows=[];seen=set()
    for i,path in enumerate(paths):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:raise ValueError('Duplicate calibration image bytes')
        seen.add(digest);value=prepare(cv2.imread(str(path)),a.role)
        destination=a.output/(f'{i:06d}.npy');np.save(destination,value)
        rows.append({'source':str(path),'source_sha256':digest,'tensor':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    a.output.with_name(a.output.name+'-manifest.json').write_text(json.dumps({'count':len(rows),'seed':42,
        'role':a.role,'preprocessing':'official PP-OCRv3 BGR normalization; no fused compiler preprocessing',
        'records':rows},indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
