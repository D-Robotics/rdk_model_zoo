# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Strictly export the author's trained VGG16/VOC21 UNet as a new specification."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
from utils.tools.torchvision_conversion import identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-record',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--native-commit',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch,onnx,onnxruntime as ort
    from PIL import Image
    from samples.vision.unet.runtime.python.vgg import preprocess
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False
    facts=json.loads(a.source_record.read_text());source=Path(facts['source_checkout']);checkpoint=Path(facts['checkpoint']['path'])
    if identity(checkpoint)['sha256']!=facts['checkpoint']['sha256']:raise ValueError('Source weight changed')
    sys.path.insert(0,str(source));from nets.unet import Unet
    model=Unet(num_classes=21,pretrained=False,backbone='vgg').cpu().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    spec=importlib.util.spec_from_file_location('upstream_unet_pixel_policy',source/'utils/utils.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    images=sorted(p for p in a.images.iterdir()if p.suffix.lower()in ['.jpg','.jpeg','.png']);assert len(images)==200
    a.output.mkdir(exist_ok=False);cal=a.output/'calibration';cal.mkdir();records=[]
    for i,path in enumerate(images):
        value,context=preprocess(path)
        with Image.open(path)as image:canvas,_,_=upstream.resize_image(image.convert('RGB'),(512,512))
        expected=upstream.preprocess_input(np.array(canvas,np.float32)).transpose(2,0,1)[None]
        np.testing.assert_array_equal(value,expected)
        target=cal/f'{i:04d}.npy';np.save(target,value,allow_pickle=False)
        records.append({'image':identity(path),'tensor':identity(target),'shape':list(value.shape),'dtype':str(value.dtype),'context':context})
    manifest={'count':200,'files':records,'calibration_domain':'exact normalized source ONNX input','geometry':'Source PIL bicubic fit into 512x512, centered RGB128 pad','normalization':{'scale':1/255,'mean':[0,0,0],'std':[1,1,1]},'producer':identity(__file__),'upstream_preprocessor':identity(source/'utils/utils.py'),'source_preprocessing_parity':'bit-exact for all 200 real images'}
    mp=a.output/'calibration.manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    cases=[('real-'+str(i),np.load(records[i]['tensor']['path'],allow_pickle=False))for i in [0,100,199]]
    cases.append(('random-seed0',np.random.default_rng(0).uniform(0,1,(1,3,512,512)).astype(np.float32)))
    expected=[]
    with torch.inference_mode():
        for _,value in cases:expected.append(model(torch.from_numpy(value)).numpy())
        graph=a.output/'model.onnx';torch.onnx.export(model,torch.from_numpy(cases[0][1]),str(graph),input_names=['images'],output_names=['logits'],opset_version=11,do_constant_folding=False,dynamo=False)
    onnx.checker.check_model(str(graph));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider']);checks=[]
    for (name,value),wanted in zip(cases,expected,strict=True):
        got=session.run(None,{'images':value})[0]
        if got.shape!=wanted.shape or got.dtype!=wanted.dtype or not np.isfinite(got).all():raise ValueError('Wrong output shape/dtype/values')
        count=int(np.count_nonzero(~np.isclose(got,wanted,atol=5e-5,rtol=1e-4)))
        checks.append({'input':name,'comparisons':[{'name':'logits','shape':list(got.shape),'dtype':str(got.dtype),'violations':count,'max_abs':float(np.max(np.abs(got-wanted))),'passed':count==0}],'status':'passed'if not count else'failed'})
    report={'schema_version':1,'status':'passed'if all(c['status']=='passed'for c in checks)else'failed',
            'source':{'repository':facts['repository'],'commit':facts['source_commit'],'weight_url':facts['checkpoint']['url'],'checkpoint':facts['checkpoint'],'license':'MIT'},
            'native_commit':a.native_commit,'onnx':identity(graph),'exporter':identity(__file__),'source_implementation':identity(source/'nets/unet.py'),
            'calibration_manifest':identity(mp),'input':{'name':'images','shape':[1,3,512,512],'dtype':'float32'},'outputs':[{'name':'logits','shape':[1,21,512,512],'dtype':'float32'}],
            'parameter_count':sum(p.numel()for p in model.parameters()),'normalization':manifest['normalization'],'geometry':manifest['geometry'],
            'atol':5e-5,'rtol':1e-4,'cases':checks,'source_preprocessing_parity':'all 200 real images bit-exact; original strict UNet tolerances retained',
            'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'board':'not-run','dataset_accuracy':'not-run'}
    (a.output/'export.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'cases':checks}),flush=True)
    return 0 if report['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())
