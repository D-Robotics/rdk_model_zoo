# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the public trained Chinese LPRNet with source-exact alphabet and heads."""
from pathlib import Path
import ast,hashlib,importlib.util,json,random,sys
import cv2,numpy as np,onnx,onnxruntime as ort,torch
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True,help='Pinned LPRNet_Pytorch source checkout')
p.add_argument('--output',type=Path,required=True,help='Fresh export/calibration directory')
a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
from samples.vision.lprnet.runtime.python.lprnet import CHARS as native_chars

def identity(p):
    p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}

assert identity(source/'weights/Final_LPRNet_model.pth')['sha256']=='d1ca22ca7a251a90702a166c8ad8ce7f05acb39f5ef8055139a33d462716d1ad'
assert identity(source/'model/LPRNet.py')['sha256']=='94a49696268ff942d92d58285ddeacab9b2ef35a4102bd73862518729e0dfc14'
facts={'repository':'https://github.com/sirius-ai/LPRNet_Pytorch','commit':'7c976664b3f3879efabeaff59c7a117e49d5f29e','checkout':str(source)}
chars=next(ast.literal_eval(node.value)for node in ast.parse((source/'data/load_data.py').read_text()).body if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CHARS'for t in node.targets))
assert tuple(chars)==tuple(native_chars)and len(chars)==68
spec=importlib.util.spec_from_file_location('lprnet_source',source/'model/LPRNet.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
model=module.build_lprnet(lpr_max_len=8,phase='test',class_num=68,dropout_rate=0).eval()
checkpoint=source/'weights/Final_LPRNet_model.pth';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
assert all(torch.isfinite(t).all()for t in model.state_dict().values())
out.mkdir(exist_ok=False);cal=out/'calibration';cal.mkdir()
images=sorted((source/'data/test').glob('*.jpg'));assert len(images)==1000
selected=sorted(random.Random(42).sample(images,200));records=[]
for index,p in enumerate(selected):
    image=cv2.imread(str(p));assert image is not None
    value=(cv2.resize(image,(94,24),interpolation=cv2.INTER_LINEAR).astype(np.float32)-127.5)*.0078125
    value=np.ascontiguousarray(value.transpose(2,0,1)[None]);q=cal/f'{index:03d}.npy';np.save(q,value)
    records.append({'source':identity(p),'label':p.stem,'tensor':identity(q)})
manifest=out/'calibration-manifest.json';manifest.write_text(json.dumps({'dataset':'1000 cropped plate photographs published by source repository','split':'Source test directory, explicit 200-image PTQ partition; remaining 800 excluded from calibration for future evaluation','source_commit':facts['commit'],'seed':42,'count':200,'preprocessing':'BGR, OpenCV linear resize 94x24, (pixel-127.5)*0.0078125, float32 NCHW','source_loader':identity(source/'data/load_data.py'),'records':records,'board_accuracy':'not-run'},ensure_ascii=False,indent=2))
cases=[(str(i),np.load(records[i]['tensor']['path'],allow_pickle=False))for i in [0,99,199]]
cases.append(('random-seed0',np.random.default_rng(0).uniform(-1,1,(1,3,24,94)).astype(np.float32)))
expected=[]
with torch.inference_mode():
    for _,x in cases:expected.append(model(torch.from_numpy(x)).numpy().copy()[...,None])

class ChannelSpatialPool(torch.nn.Module):
    """Exact 4D unbatched MaxPool3d semantics as channel stride and 2D spatial pool."""
    def __init__(self,layer):super().__init__();self.stride=layer.stride;self.kernel=layer.kernel_size
    def forward(self,x):
        return torch.nn.functional.max_pool2d(x[:,::self.stride[0]],self.kernel[1:],self.stride[1:])

for index,layer in enumerate(model.backbone):
    if isinstance(layer,torch.nn.MaxPool3d):
        assert layer.kernel_size[0]==1 and layer.padding==0 and layer.dilation==1 and not layer.ceil_mode
        model.backbone[index]=ChannelSpatialPool(layer)

class Export(torch.nn.Module):
    def __init__(self,source):super().__init__();self.source=source
    def forward(self,images):return self.source(images).unsqueeze(-1)

wrapper=Export(model).eval();path=out/'model.onnx'
with torch.inference_mode():
    for (_,x),baseline in zip(cases,expected):np.testing.assert_array_equal(wrapper(torch.from_numpy(x)).numpy(),baseline)
    torch.onnx.export(wrapper,torch.from_numpy(cases[0][1]),str(path),input_names=['images'],output_names=['logits'],opset_version=11,dynamo=False,do_constant_folding=True)
onnx.checker.check_model(str(path));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);checks=[]
for (name,x),baseline in zip(cases,expected):
    actual=session.run(None,{'images':x})[0];assert actual.shape==baseline.shape==(1,68,18,1)and actual.dtype==np.float32
    finite=bool(np.isfinite(actual).all());violations=int(np.count_nonzero(~np.isclose(actual,baseline,atol=1e-4,rtol=1e-3)))
    row={'case':name,'finite':finite,'violations':violations,'max_abs':float(np.max(np.abs(actual-baseline))),'status':'passed'if finite and not violations else'failed'}
    checks.append(row);print(json.dumps(row),flush=True)
report={'status':'passed'if all(r['status']=='passed'for r in checks)else'failed','source':facts,'checkpoint':identity(checkpoint),'source_definition':identity(source/'model/LPRNet.py'),'license':'Apache-2.0','license_file':identity(source/'LICENSE'),'exact_native_alphabet_match':True,'classes':chars,'raw_adapter_bit_exact':True,'adapter':'4D source MaxPool3d channel stride expressed as exact channel slicing plus spatial MaxPool2d; final singleton output axis only','parameter_count':sum(p.numel()for p in model.parameters()),'onnx':identity(path),'calibration_manifest':identity(manifest),'input':{'name':'images','shape':[1,3,24,94],'dtype':'float32','color':'BGR'},'outputs':[{'name':'logits','shape':[1,68,18,1],'dtype':'float32'}],'atol':1e-4,'rtol':1e-3,'checks':checks,'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'board':'not-run','dataset_accuracy':'not-run'}
(out/'export.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(report['status']!='passed')
