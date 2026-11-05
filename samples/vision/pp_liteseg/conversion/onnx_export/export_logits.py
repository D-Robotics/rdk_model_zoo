"""Export the official STDC1 Cityscapes checkpoint with float logits.

Run this in the isolated Paddle 3.0/Paddle2ONNX 2.0.1 export environment.
All checkpoint, static-model and ONNX paths belong on the conversion server.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

SOURCE_COMMIT='3c4db66de1d9d59d0628ed87590b6308a2f4aa2a'
CHECKPOINT_SHA256='86543ad69437495f8e7688d497a52d2472ee098efa918dcfa551803570f5edf3'


def identity(path):
    path=Path(path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes':path.stat().st_size}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Pinned PaddleSeg source directory')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New server output directory')
    parser.add_argument('--samples',type=Path,nargs=3,required=True,help='Three real input images for the source/ONNX gate')
    args=parser.parse_args(argv)
    source=args.source.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True)
    if identity(checkpoint)['sha256']!=CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from the registered official STDC1 variant')
    if (source/'.git').exists():
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if commit!=SOURCE_COMMIT:raise ValueError('Use the pinned PaddleSeg source revision')
    for image in args.samples:image.resolve(strict=True)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source))
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import paddle
    import paddle2onnx
    import yaml
    from argparse import Namespace
    from paddleseg.cvlibs import Config,SegBuilder
    from paddleseg.core.export import export

    paddle.set_device('cpu');paddle.seed(42)
    cfg=Config(str(source/'configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml'))
    cfg.dic['model']['backbone']['pretrained']=None
    cfg.dic['model']['num_classes']=19
    frozen=output/'source-config.yml';frozen.write_text(yaml.safe_dump(cfg.dic,sort_keys=False))
    model=SegBuilder(cfg).model;state=paddle.load(str(checkpoint));expected=model.state_dict()
    # Upstream introduced three frozen Constant(1) parameters after releasing
    # this checkpoint. They implement the original literal (1 - attention).
    constants=[f'ppseg_head.arm_list.{i}._scale'for i in range(3)]
    if set(expected)-set(state)!=set(constants) or set(state)-set(expected):
        raise ValueError('Missing or unexpected checkpoint tensors')
    for key in constants:
        value=expected[key]
        if not value.stop_gradient or tuple(value.shape)!=(1,) or not np.array_equal(value.numpy(),np.ones([1],np.float32)):
            raise ValueError('Upstream attention constant differs')
        state[key]=paddle.ones_like(value)
    if set(expected)!=set(state) or any(tuple(expected[k].shape)!=tuple(state[k].shape)for k in expected):
        raise ValueError('Checkpoint keys or tensor shapes differ')
    model.set_state_dict(state);model.eval()
    export(Namespace(config=str(frozen),model_path=str(checkpoint),save_dir=str(output/'paddle'),
                     input_shape=[1,3,512,1024],output_op='none',for_fd=False),model=model)
    static=output/'paddle'
    if not (static/'model.json').is_file() or not (static/'model.pdiparams').is_file():
        raise ValueError('Paddle export did not produce the expected static model')
    graph=output/'pp_liteseg_stdc1_cityscapes_1024x512_logits.onnx'
    command=[str(Path(sys.executable).with_name('paddle2onnx')),'--model_dir',str(static),
             '--model_filename','model.json','--params_filename','model.pdiparams','--save_file',str(graph),
             '--opset_version','11','--enable_auto_update_opset','False']
    with (output/'paddle2onnx.log').open('w')as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    onnx.checker.check_model(onnx.load(str(graph)))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    if len(session.get_inputs())!=1 or session.get_inputs()[0].shape!=[1,3,512,1024]:
        raise ValueError('ONNX input ABI differs')
    if len(session.get_outputs())!=1 or session.get_outputs()[0].shape!=[1,19,512,1024]:
        raise ValueError('ONNX logits ABI differs')
    checks=[]
    for number,image in enumerate([*args.samples,None]):
        if image is None:
            x=np.random.default_rng(42).uniform(-1,1,[1,3,512,1024]).astype(np.float32)
        else:
            bgr=cv2.imread(str(image))
            if bgr is None:raise ValueError(f'Cannot decode {image}')
            rgb=cv2.cvtColor(cv2.resize(bgr,(1024,512),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB).astype(np.float32)
            x=np.ascontiguousarray(((rgb/255-.5)/.5).transpose(2,0,1)[None])
        with paddle.no_grad():reference=model(paddle.to_tensor(x))[0].numpy()
        actual=session.run(None,{session.get_inputs()[0].name:x})[0]
        if reference.shape!=actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError('Non-finite or mismatched logits')
        left=reference.astype(np.float64).ravel();right=actual.astype(np.float64).ravel()
        cosine=float(np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right)))
        if cosine<.95:raise ValueError(f'Pre-argmax source/ONNX cosine failed: {cosine}')
        checks.append({'input':identity(image)if image else 'random-seed42','status':'passed',
                       'logits_cosine':cosine,'max_abs':float(np.max(np.abs(reference-actual))),
                       'argmax_agreement':float(np.mean(reference.argmax(1)==actual.argmax(1)))})
    source_files=['configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml',
                  'configs/_base_/cityscapes.yml','paddleseg/models/pp_liteseg.py',
                  'paddleseg/models/backbones/stdcnet.py','paddleseg/models/layers/tensor_fusion.py',
                  'paddleseg/transforms/transforms.py']
    report={'schema_version':1,'status':'passed','source_commit':SOURCE_COMMIT,
            'source_files':{name:identity(source/name)for name in source_files},
            'checkpoint':identity(checkpoint),'exporter':identity(__file__),'onnx':identity(graph),
            'strict_trained_checkpoint_load':True,'upstream_non_trainable_constants':{'keys':constants,'value':1.0},
            'input':{'name':session.get_inputs()[0].name,'shape':[1,3,512,1024],'dtype':'float32'},
            'outputs':[{'name':session.get_outputs()[0].name,'shape':[1,19,512,1024],'dtype':'float32'}],
            'normalization':{'format':'RGB','scale':1/255,'mean':[.5]*3,'std':[.5]*3},
            'source_host_gate':{'metric':'matching pre-argmax logits cosine','threshold':.95,'cases':checks},
            'versions':{'paddle':paddle.__version__,'paddle2onnx':paddle2onnx.__version__,
                        'onnx':onnx.__version__,'onnxruntime':ort.__version__},'board':'not-run'}
    (output/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','onnx':report['onnx'],'cases':checks}))
    return 0


if __name__=='__main__':raise SystemExit(main())
