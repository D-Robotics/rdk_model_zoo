"""Prepare immutable S-series calibration data and specialize an OE-generated YAML."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data(args):
    rows=json.loads(args.manifest.read_text())
    if len(rows)!=128: raise ValueError('Expected the frozen 128 training crops')
    args.output.mkdir(parents=True,exist_ok=False)
    target=args.output/'images';target.mkdir()
    receipt=[]
    for row in rows:
        p=Path(row['image'])
        if p.parent.name!='bounding_box_train' or sha(p)!=row['image_sha256']: raise ValueError('Calibration image identity differs')
        with Image.open(p) as im:
            x=np.asarray(im.convert('RGB').resize((128,256),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/np.float32(255)
        x=np.ascontiguousarray((x-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])[None]
        out=target/(p.stem+'.npy');np.save(out,x)
        receipt.append({'image':str(p),'image_sha256':sha(p),'tensor':out.name,'tensor_sha256':sha(out),'shape':list(x.shape),'dtype':str(x.dtype)})
    (args.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'samples':len(receipt),'input':'normalized RGB float32 NCHW [1,3,256,128]','manifest_sha256':sha(args.output/'manifest.json')}))

def config(args):
    import yaml
    x=yaml.safe_load(args.template.read_text())
    x['model_parameters'].update(onnx_model=str(args.onnx),march={'s100':'nash-e','s100p':'nash-m','s600':'nash-p'}[args.target],working_dir=str(args.workdir),output_model_file_prefix='osnet_x1_0')
    x['input_parameters'].update(input_name='images',input_type_train='featuremap',input_type_rt='featuremap',input_shape='1x3x256x128',input_batch=1)
    # Normalization is in the caller and calibration tensors, not duplicated in the compiler.
    for key in ('input_layout_train','input_layout_rt','mean_value','scale_value','std_value','input_space_and_range','norm_type'):
        x['input_parameters'].pop(key,None)
    x['calibration_parameters']['cal_data_dir']=str(args.calibration)
    if args.precision == 'int16':
        x['calibration_parameters']['quant_config']={'model_config':{'all_node_type':'int16'}}
    x['compiler_parameters'].update(compile_mode='latency',optimize_level='O2',jobs=8,core_num=1,input_source={'images':'ddr'})
    with args.output.open('x') as f: yaml.safe_dump(x,f,sort_keys=False)
    print(args.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    d=s.add_parser('data');d.add_argument('--manifest',type=Path,required=True);d.add_argument('--output',type=Path,required=True)
    c=s.add_parser('config')
    for key in ('template','onnx','workdir','calibration','output'):c.add_argument('--'+key,type=Path,required=True)
    c.add_argument('--target',choices=('s100','s100p','s600'),required=True)
    c.add_argument('--precision',choices=('default','int16'),default='default')
    a=p.parse_args();{'data':data,'config':config}[a.command](a)
if __name__=='__main__':main()
