# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source split-ONNX detection and real CPU person tracking; no HBM execution."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from samples.vision.fasterrcnn.runtime.python.cli import resolve_selection,load_descriptor
from samples.vision.fasterrcnn.runtime.python.detect import (
    FasterRCNNTask,PairedRuntime,PreparedInput,letterbox_geometry,pack_bank,unpack_bank_outputs)
from samples.vision.fasterrcnn.runtime.python.tracking import PersonTrackerTask
from utils.py_utils.assets import sha256_file

VARIANT='mbv3-320-native-20261103-o1'


class SourcePairedRuntime(PairedRuntime):
    def __init__(self,selection,parta,partb):
        import onnxruntime as ort
        super().__init__(selection)
        descriptor=load_descriptor(VARIANT);self.sessions={};self.last_raw={}
        for role,path in [('parta',parta),('partb',partb)]:
            key='parta_f0only_op16_sha256'if role=='parta'else'partb_packed_op16_sha256'
            if sha256_file(path)!=descriptor['onnx'][key]:raise ValueError('Wrong source graph '+role)
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            self.sessions[role]=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        if [(x.name,x.shape)for x in self.sessions['parta'].get_inputs()]!=[('images',[1,3,320,320])]:
            raise ValueError('Wrong source PartA input')
        if [(x.name,x.shape)for x in self.sessions['partb'].get_inputs()]!=[('bank',[1,256,1050,7])]:
            raise ValueError('Wrong source PartB input')
    def load(self):return self
    def run_parta(self,tensors):
        self.last_raw={}
        session=self.sessions['parta'];out={x.name:v for x,v in zip(session.get_outputs(),session.run(None,tensors),strict=True)}
        self.last_raw.update(out);return out
    def run_partb(self,pooled):
        session=self.sessions['partb'];out={x.name:v for x,v in zip(session.get_outputs(),session.run(None,{'bank':pack_bank(pooled)}),strict=True)}
        self.last_raw.update(out)
        return unpack_bank_outputs(out['cls_packed'],out['reg_packed'])


class SourceFloatDetector(FasterRCNNTask):
    def __init__(self,parta,partb):
        selection=resolve_selection('s100',variant=VARIANT)
        super().__init__(runner=SourcePairedRuntime(selection,parta,partb))
    def preprocess(self,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected BGR uint8 image')
        context=letterbox_geometry(*image.shape[:2]);height,width=context.net_size
        canvas=np.zeros((320,320,3),np.uint8)
        canvas[:height,:width]=cv2.resize(image,(width,height),interpolation=cv2.INTER_LINEAR)
        value=canvas[:,:,::-1].astype(np.float32)/np.float32(255)
        value=(value-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
        return PreparedInput({'images':np.ascontiguousarray(value.transpose(2,0,1)[None])},context)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parta-onnx',type=Path,required=True);p.add_argument('--partb-onnx',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--frames',type=int,default=30)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.frames<30:raise ValueError('Tracking acceptance requires at least 30 ordered real frames')
    a.output.mkdir(parents=True,exist_ok=False);detector=SourceFloatDetector(a.parta_onnx,a.partb_onnx)
    tracker=PersonTrackerTask(detector);video=cv2.VideoCapture(str(a.video));frames=[]
    if not video.isOpened():raise ValueError('Unreadable real video')
    try:
        for i in range(a.frames):
            ok,image=video.read()
            if not ok:break
            tracks=tracker.predict(image);raw=detector.runner.last_raw
            np.savez_compressed(a.output/(f'frame-{i:03d}-outputs.npz'),**raw)
            frames.append({'frame':i+1,'input_sha256':__import__('hashlib').sha256(image.tobytes()).hexdigest(),
                           'outputs':list(raw),'proposals':detector.last_proposals,
                           'tracks':[{'id':t.track_id,'box':list(t.tlbr),'score':t.score}for t in tracks]})
    finally:video.release()
    ids=[{t['id']for t in row['tracks']}for row in frames]
    stable=any(x&y for x,y in zip(ids,ids[1:]))
    passed=len(frames)==a.frames and stable and all(set(x['outputs'])=={'rpn_obj','rpn_deltas','feat0','cls_packed','reg_packed'}for x in frames)
    report={'schema_version':1,'status':'passed'if passed else'failed','scope':'source PartA ONNX -> Native RPN/RoIAlign -> packed PartB ONNX -> Native person filter -> real CPU ByteTrack',
            'source_checkpoint_sha256':load_descriptor(VARIANT)['source']['checkpoint_sha256'],
            'onnx_sha256':{'parta':sha256_file(a.parta_onnx),'partb':sha256_file(a.partb_onnx)},
            'video':{'path':str(a.video),'sha256':sha256_file(a.video)},'frame_count':len(frames),'stable_adjacent_ids':stable,
            'all_five_detector_outputs_preserved':all(len(row['outputs'])==5 for row in frames),'frames':frames,
            'board':'not-run','tracking_accuracy':'not-run'}
    (a.output/'tracking-host-pipeline.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'frames':len(frames),'stable_adjacent_ids':stable}),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
