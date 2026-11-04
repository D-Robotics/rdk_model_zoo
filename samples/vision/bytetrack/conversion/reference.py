# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host source-ONNX detector and real CPU ByteTrack; never executes an HBM."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from samples.vision.bytetrack.runtime.python.cli import resolve_selection
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
from samples.vision.yolov5.runtime.python.detection import bind_model, YOLOv5Task, resized_image
from utils.py_utils.assets import sha256_file


class SourceFloatDetector:
    """Feed source RGB floats, then use the audited 672 Native decode geometry.

    Physical NV12 preparation remains distinct from this host reference. The
    converted registry supplies the actual compiled head contract; no board SDK
    or second dequantization is involved.
    """
    def __init__(self, selection, onnx_path, threads=2):
        import onnxruntime as ort
        path = Path(__file__).with_name('converted_targets_20261104.json')
        profiles = json.loads(path.read_text())['profiles']
        profile, = [p for p in profiles if p['variant'] == selection.variant
                    and p['target']['platform'] == selection.target]
        if sha256_file(onnx_path) != profile['components']['model']['onnx_sha256']:
            raise ValueError('Source graph differs from the accepted conversion')
        metadata = {'model_name':'source-float-detector'}
        for singular, plural in [('input','inputs'), ('output','outputs')]:
            rows = profile['io'][plural]
            metadata.update({singular+'_names':[r['name'] for r in rows],
                singular+'_shapes':{r['name']:r['shape'] for r in rows},
                singular+'_dtypes':{r['name']:r['dtype'] for r in rows}})
        binding = bind_model(selection, metadata)
        self.native = YOLOv5Task(runner=self.infer, binding=binding)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
        if [(i.name, i.shape) for i in self.session.get_inputs()] != [('images',[1,3,672,672])]:
            raise ValueError('Unexpected ByteTrack source graph input')

    def preprocess(self, image):
        physical = self.native.preprocess(image)
        pixels = resized_image(image, 672, 672, 0)
        value = np.ascontiguousarray(pixels[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
        return SimpleNamespace(tensors={'images':value}, context=physical.context)

    def infer(self, tensors):
        return {i.name:v for i,v in zip(self.session.get_outputs(), self.session.run(None,tensors), strict=True)}

    def postprocess(self, outputs, context):
        return self.native.postprocess(outputs, context)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('s100','s100p','s600'),required=True)
    p.add_argument('--variant',required=True)
    p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=30)
    a=p.parse_args(argv)
    if a.frames < 2: raise ValueError('At least two ordered real frames are required')
    a.output.mkdir(parents=True,exist_ok=False)
    detector=SourceFloatDetector(resolve_selection(a.target,variant=a.variant),a.onnx)
    tracker=ByteTrackTask(detector);video=cv2.VideoCapture(str(a.input));records=[]
    if not video.isOpened(): raise ValueError('Cannot open the supplied real video')
    try:
        for index in range(a.frames):
            ok,image=video.read()
            if not ok: break
            prepared=tracker.preprocess(image);raw=tracker.infer(prepared.tensors)
            tracks=tracker.postprocess(raw,prepared.context)
            np.savez_compressed(a.output/(f'frame-{index:03d}-outputs.npz'),**raw)
            records.append({'frame':index+1,'input_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                'outputs':[{'name':n,'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()}for n,v in raw.items()],
                'tracks':[{'id':t.track_id,'box':list(t.tlbr),'score':t.score}for t in tracks]})
    finally: video.release()
    ids=[{t['id']for t in r['tracks']}for r in records]
    stable=any(left & right for left,right in zip(ids,ids[1:]))
    passed=len(records)==a.frames and stable and all(len(r['outputs'])==3 for r in records)
    report={'schema_version':1,'status':'passed'if passed else'failed',
        'scope':'source float ONNX + Native person decoding + real CPU tracker; no HBM/board execution or MOT accuracy',
        'video':{'path':str(a.input),'sha256':sha256_file(a.input)},
        'onnx_sha256':sha256_file(a.onnx),'asset_id':detector.native.binding.selection.asset.reference,
        'frame_count':len(records),'stable_adjacent_id':stable,'frames':records,'board':'not-run'}
    (a.output/'host-pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'frames':len(records),'stable_adjacent_id':stable}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
