# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Freeze a PP-OCRv3 Paddle2ONNX graph and check every output against Paddle CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from samples.vision.paddle_ocr.conversion.preprocess_v3 import prepare

PINNED = {'det': {'program_sha256': 'c4bfb1b05d9d1d5a760801eaf6d20180ef7e47bcc675fb17d1f3a89da5fef427', 'parameters_sha256': '83676ec730627ab4502f401410a4b6a3ce1c0bb98fa249b71db055b6bddae051'}, 'rec': {'program_sha256': 'f0288362f754e815030929477f7b1f96fe2e913e197537288a01bd43f768d0ab', 'parameters_sha256': '2cea2de604d8a75fd9151b54ea2dd027659fb8cb777cde377224581110dbff13'}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('det','rec'),required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--raw-onnx',type=Path,required=True)
    p.add_argument('--image',type=Path,action='append',required=True,help='Three or more real detector images or already cropped recognition words')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if len(a.image)<3:raise ValueError('At least three real verification inputs are required')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for filename,key in [('inference.pdmodel','program_sha256'),('inference.pdiparams','parameters_sha256')]:
        if sha(a.model_dir/filename)!=PINNED[a.role][key]:raise ValueError('Paddle source model differs from pinned PP-OCRv3 English')
    import onnx,onnxruntime as ort,paddle.inference as pi
    a.output.mkdir(parents=True,exist_ok=False)
    h,w=(640,640)if a.role=='det'else(48,320)
    graph=onnx.load(a.raw_onnx)
    if len(graph.graph.input)!=1 or len(graph.graph.output)!=1:raise ValueError('Expected one complete input/output')
    for dim,size in zip(graph.graph.input[0].type.tensor_type.shape.dim,[1,3,h,w],strict=True):
        dim.ClearField('dim_param');dim.dim_value=size
    graph=onnx.shape_inference.infer_shapes(graph,strict_mode=True,data_prop=True)
    model=a.output/'model.onnx';onnx.save(graph,model);onnx.checker.check_model(model)
    config=pi.Config(str(a.model_dir/'inference.pdmodel'),str(a.model_dir/'inference.pdiparams'))
    config.disable_gpu();config.set_cpu_math_library_num_threads(2);config.switch_ir_optim(False)
    predictor=pi.create_predictor(config);input_name,=predictor.get_input_names();output_name,=predictor.get_output_names()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
    cases=[(str(image),prepare(cv2.imread(str(image)),a.role))for image in a.image]
    cases.append(('random-seed0',np.random.default_rng(0).uniform(-1,1,(1,3,h,w)).astype(np.float32)))
    checks=[]
    for index,(label,value)in enumerate(cases):
        predictor.get_input_handle(input_name).reshape(value.shape);predictor.get_input_handle(input_name).copy_from_cpu(value)
        predictor.run();left=predictor.get_output_handle(output_name).copy_to_cpu();right,=session.run(None,{session.get_inputs()[0].name:value})
        shape=(1,1,640,640)if a.role=='det'else(1,40,97)
        if left.shape!=shape or right.shape!=shape or left.dtype!=np.float32 or right.dtype!=np.float32 or not np.isfinite(right).all():raise ValueError('Unexpected complete output contract')
        np.testing.assert_allclose(right,left,atol=1e-4,rtol=1e-3)
        np.save(a.output/(str(index)+'.npy'),value)
        checks.append({'input':label,'passed':True,'max_abs':float(np.abs(right-left).max()),'shape':list(right.shape)})
    (a.output/'source-parity.json').write_text(json.dumps({'status':'passed','role':a.role,
        'source':PINNED[a.role],'raw_onnx_sha256':sha(a.raw_onnx),'onnx_sha256':sha(model),
        'producer_sha256':sha(__file__),'checks':checks,'atol':1e-4,'rtol':1e-3,
        'scope':'Paddle CPU / ONNX CPU source equivalence; no board or dataset accuracy'},indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
