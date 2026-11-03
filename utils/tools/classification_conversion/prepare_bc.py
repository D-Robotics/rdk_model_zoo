"""Use the verified S SDK 3.7.0 stages through NodeInfo, retaining same-target BC."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
import numpy as np
import onnxruntime as ort
import torch
import yaml
from horizon_tc_ui import hb_compile

def identity(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest(),'size_bytes':path.stat().st_size}

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['model_parameters']['onnx_model'])
manifest=json.loads(args.manifest.read_text())
if manifest.get('tensor_domain')!='normalized-onnx-input' or manifest['count']!=200:
    raise ValueError('Classifier PTQ requires the frozen 200-image ONNX-domain tensor set')
cal=Path(cfg['calibration_parameters']['cal_data_dir']);files=sorted(cal.glob('*.npy'))
if len(files)!=200:raise ValueError('Incomplete calibration directory')
for path,row in zip(files,manifest['files']):
    if path.name!=row['tensor_filename'] or identity(path)['sha256']!=row['tensor_sha256']:
        raise ValueError('Calibration bytes differ from the frozen manifest')
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    raise ValueError('The cached GPU SDK has no CUDA ORT provider')
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
cpu=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider'])
gpu=ort.InferenceSession(str(source),sess_options=options,
    providers=[('CUDAExecutionProvider',{'use_tf32':0}), 'CPUExecutionProvider'])
if gpu.get_providers()[0]!='CUDAExecutionProvider':raise ValueError('CUDA provider did not load')
if len(cpu.get_inputs())!=1:raise ValueError('This preparation path requires one classifier input')
name=cpu.get_inputs()[0].name;checks=[]
cases=[(str(index),np.load(files[index],allow_pickle=False)) for index in (0,52,143)]
cases.append(('random-seed0',np.random.default_rng(0).normal(size=(1,3,224,224)).astype(np.float32)))
for case,x in cases:
    reference=cpu.run(None,{name:x});actual=gpu.run(None,{name:x})
    for expected,value in zip(reference,actual):np.testing.assert_allclose(value,expected,atol=1e-4,rtol=1e-3)
    checks.append({'case':case,'status':'passed','max_abs':max(float(np.max(np.abs(a-b))) for a,b in zip(reference,actual))})
del cpu,gpu
torch.cuda.empty_cache()
conf=hb_compile.params_check(str(args.config))
if conf.remove_node_name or conf.remove_node_type:raise ValueError('The classifier BC must retain its entire public IO')
ptq=hb_compile.ptq_model_build(conf)
hb_compile.hbm_build(conf,ptq,skip='compile')
out=Path(cfg['model_parameters']['working_dir']);prefix=cfg['model_parameters']['output_model_file_prefix']
bc=out/(prefix+'_quantized_model.bc');csv=out/(prefix+'_node_info.csv')
if not bc.is_file() or not csv.is_file() or len(csv.read_text().splitlines()[0].split(','))<7:
    raise ValueError('Missing same-target BC or full original NodeInfo')
snapshots=[out/(prefix+suffix) for suffix in ('_ptq_model.onnx','_optimized_float_model.onnx','_quant_info.json')]
for path in snapshots:
    if not path.is_file():raise FileNotFoundError(path)
result={'schema_version':1,'status':'quantized-same-target','target':{'march':conf.march},
    'bc':identity(bc),'native_node_csv':identity(csv),'ptq_snapshots':[identity(path) for path in snapshots],
    'onnx':identity(source),'ptq_config':identity(args.config),'calibration_manifest':identity(args.manifest),
    'calibration_count':200,'comparison_input_count':1,'cpu_cuda_gate':{'atol':1e-4,'rtol':1e-3,'cases':checks,
        'providers':['CPUExecutionProvider','CUDAExecutionProvider'],'TF32':False},
    'producer':identity(__file__),'versions':{'torch':torch.__version__,'onnxruntime':ort.__version__},
    'board':'not-run','note':'SDK-native params_check -> ptq_model_build -> hbm_build(skip=compile); no cross-march reuse'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'bc':result['bc'],'cpu_cuda_gate':checks}),flush=True)
