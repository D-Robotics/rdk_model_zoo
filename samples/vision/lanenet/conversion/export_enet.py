# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export all three trained ENet LaneNet heads on a frozen real road subset."""
from pathlib import Path
import argparse,json,sys
from utils.tools.torchvision_conversion import identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Pinned IrohXu upstream checkout')
    p.add_argument('--images',type=Path,required=True,help='100 distinct frozen TuSimple train images')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-commit',required=True)
    a=p.parse_args()
    import cv2,numpy as np,onnx,onnxruntime as ort,torch
    from samples.vision.lanenet.runtime.python.lanenet import image_to_tensor
    sys.path.insert(0,str(a.source))
    from model.lanenet.LaneNet import LaneNet
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model=LaneNet(arch='ENet').cpu().eval();checkpoint=a.source/'log/best_model.pth'
    if identity(checkpoint)['sha256']!='052bda2bff1ab3aa017a4509966fc80290d2076b3b373e268bdc6865674f5b79':
        raise ValueError('Source trained checkpoint differs')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    if not all(torch.isfinite(value).all()for value in model.state_dict().values()):
        raise ValueError('Nonfinite trained weights')
    dataset=json.loads((a.images/'manifest.json').read_text())
    if dataset['status']!='acquired'or dataset['count']!=100 or dataset['partition']!='TuSimple train only':
        raise ValueError('Expected the frozen distinct real training subset')
    a.output.mkdir(exist_ok=False);cal=a.output/'calibration';cal.mkdir();records=[]
    for index,row in enumerate(dataset['records']):
        image=Path(row['path'])
        if identity(image)['sha256']!=row['sha256']:raise ValueError('Calibration image changed')
        value=image_to_tensor(cv2.imread(str(image)))
        target=cal/(f'{index:04d}.npy');np.save(target,value,allow_pickle=False)
        records.append({'image':identity(image),'archive_member':row['archive_member'],'tensor':identity(target),'shape':list(value.shape),'dtype':str(value.dtype)})
    manifest={'count':100,'partition':'TuSimple train only','dataset_manifest':identity(a.images/'manifest.json'),
              'producer':identity(__file__),'native_preprocess':identity(Path(sys.modules[image_to_tensor.__module__].__file__)),
              'geometry':'OpenCV INTER_AREA RGB stretch 512x256','normalization':{'scale':1/255,'mean':[.485,.456,.406],'std':[.229,.224,.225]},
              'calibration_domain':'exact normalized F32 NCHW source ONNX input','files':records}
    mp=a.output/'calibration.manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    cases=[('real-'+str(i),np.load(records[i]['tensor']['path'],allow_pickle=False))for i in [0,49,99]]
    cases.append(('random-seed0',np.random.default_rng(0).normal(size=(1,3,256,512)).astype(np.float32)))
    names=['instance_seg_logits','binary_seg_pred','binary_seg_logits']
    class Heads(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,value):
            result=self.network(value)
            return tuple(result[name]for name in names)
    wrapper=Heads(model).eval();expected=[]
    with torch.inference_mode():
        for _,value in cases:
            source=model(torch.from_numpy(value));got=wrapper(torch.from_numpy(value))
            for name,x in zip(names,got,strict=True):torch.testing.assert_close(x,source[name],atol=0,rtol=0)
            expected.append([x.numpy()for x in got])
        graph=a.output/'model.onnx'
        torch.onnx.export(wrapper,torch.from_numpy(cases[0][1]),str(graph),input_names=['images'],output_names=names,opset_version=13,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(graph));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(graph),options,providers=['CPUExecutionProvider']);checks=[]
    for (case,value),targets in zip(cases,expected,strict=True):
        got=session.run(None,{'images':value});comparisons=[]
        for name,x,y in zip(names,got,targets,strict=True):
            if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():raise ValueError('Wrong output '+name)
            violations=int(np.count_nonzero(x!=y))if np.issubdtype(x.dtype,np.integer)else int(np.count_nonzero(~np.isclose(x,y,atol=1e-4,rtol=1e-3)))
            comparisons.append({'name':name,'shape':list(x.shape),'dtype':str(x.dtype),'violations':violations,'max_abs':float(np.max(np.abs(x-y))),'passed':not violations})
        checks.append({'input':case,'comparisons':comparisons,'status':'passed'if all(x['passed']for x in comparisons)else'failed'})
    report={'schema_version':1,'status':'passed'if all(x['status']=='passed'for x in checks)else'failed',
            'source':{'repository':'https://github.com/IrohXu/lanenet-lane-detection-pytorch','commit':'0e5aa371d9572ef8d16b2eddbe5af284f649b88a','checkpoint':identity(checkpoint),'weight_url':'https://github.com/IrohXu/lanenet-lane-detection-pytorch/blob/0e5aa371d9572ef8d16b2eddbe5af284f649b88a/log/best_model.pth','license':'MIT'},
            'native_commit':a.native_commit,'exporter':identity(__file__),'source_implementation':identity(a.source/'model/lanenet/LaneNet.py'),
            'onnx':identity(graph),'calibration_manifest':identity(mp),'calibration_count':100,'raw_wrapper_bit_exact':True,
            'input':{'name':'images','shape':[1,3,256,512],'dtype':'float32'},'outputs':[{'name':n,'shape':list(v.shape),'dtype':str(v.dtype)}for n,v in zip(names,expected[0],strict=True)],
            'parameter_count':sum(x.numel()for x in model.parameters()),'normalization':manifest['normalization'],'geometry':manifest['geometry'],'atol':1e-4,'rtol':1e-3,'integer_outputs':'exact element equality; all three source heads retained','cases':checks,
            'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'board':'not-run','dataset_accuracy':'not-run'}
    (a.output/'export.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'cases':checks}),flush=True)
    return 0 if report['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())
