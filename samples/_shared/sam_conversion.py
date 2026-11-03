# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate a complete SAM float pair, then generate real encoder/decoder calibration.

This host-only command uses the native sample exporters and the selected upstream
checkpoint. It never loads a board SDK, downloads inputs or fabricates embeddings.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', choices=('efficient_sam','mobile_sam'), required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--decoder', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, help='COCO training instances for real MobileSAM box prompts')
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    args = p.parse_args()
    images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'))
    if args.count < 20 or len(images) < args.count:
        raise ValueError('At least 20 distinct representative photographs are required')
    images = images[:args.count]
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sample == 'mobile_sam' and not args.annotations:
        raise ValueError('MobileSAM calibration requires real COCO boxes with the selected training images')
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(args.repo))
    directory = ROOT/'samples/vision'/args.sample/'conversion/scripts'
    enc = module(directory/'export_encoder_onnx.py','sam_encoder_export')
    dec = module(directory/'export_decoder_onnx.py','sam_decoder_export')
    if args.sample == 'efficient_sam':
        _,source = enc._load_model(SimpleNamespace(target='s100', repo=args.repo, checkpoint=args.checkpoint))
        source.image_encoder.img_size = 512
        source.image_encoder.image_embedding_size = 32
        source.prompt_encoder.input_image_size = (512,512)
        source.prompt_encoder.image_embedding_size = (32,32)
        adapted = copy.deepcopy(source)
        enc.split_qkv_linear_modules(adapted,torch)
        encoder = enc._build_encoder_wrapper(adapted,torch)
        decoder = dec.build_decoder(adapted,torch,512,[248.,210.,302.,315.])
        points = torch.tensor([[[[248.,210.],[302.,315.]]]])
        labels = torch.ones((1,1,2))

        def reference(value):
            embedding = source.get_image_embeddings(value)
            masks,iou = source.predict_masks(embedding,points,labels,True,512,512)
            return embedding,(masks[:,0],iou[:,0])
    else:
        import ultralytics
        if ultralytics.__version__ != '8.4.127':
            raise ValueError('This MobileSAM source gate requires ultralytics==8.4.127')
        from ultralytics.models.sam.build import build_mobile_sam
        source = build_mobile_sam(str(args.checkpoint)).eval()
        source.set_imgsz((512,512))
        encoder = source.image_encoder
        decoder = dec._build_decoder_wrapper(torch,source.prompt_encoder,source.mask_decoder)

        def reference(value, box=None):
            embedding = source.image_encoder(value)
            sparse,dense = source.prompt_encoder(points=None,boxes=box,masks=None)
            masks,iou = source.mask_decoder(embedding,source.prompt_encoder.get_dense_pe(),sparse,dense,True)
            return embedding,(masks,iou)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = [ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
                for path in (args.encoder,args.decoder)]
    args.output.mkdir(parents=True)
    report = {'status':'failed','sample':args.sample,'torch_version':torch.__version__,
              'onnxruntime_version':ort.__version__,'checkpoint_sha256':sha(args.checkpoint),
              'encoder_onnx_sha256':sha(args.encoder),'decoder_onnx_sha256':sha(args.decoder),
              'atol':1e-4,'rtol':1e-4,'minimum_cosine':.99999,'cases':[],
              'board':'not-run','accuracy':'not-run'}

    def prepare(image):
        rgb = cv2.cvtColor(cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        value = rgb.astype(np.float32).transpose(2,0,1)[None]
        if args.sample == 'efficient_sam':
            return np.ascontiguousarray(value/255.)
        mean = np.array([123.675,116.28,103.53],np.float32)[None,:,None,None]
        std = np.array([58.395,57.12,57.375],np.float32)[None,:,None,None]
        return np.ascontiguousarray((value-mean)/std)

    def compare(expected, observed):
        require = expected.shape == observed.shape and np.isfinite(observed).all()
        if not require:
            raise ValueError(f'Shape/finite mismatch: {expected.shape}/{observed.shape}')
        a,b = expected.astype(np.float64).ravel(),observed.astype(np.float64).ravel()
        cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        result = {'shape':list(expected.shape),'max_abs_error':float(np.max(np.abs(a-b))),
                  'cosine':cosine,'passed':bool(np.allclose(expected,observed,atol=1e-4,rtol=1e-4) and cosine>=.99999)}
        return result

    try:
        indices = [0,len(images)//3,2*len(images)//3]
        cases = [(str(images[i]),prepare(cv2.imread(str(images[i])))) for i in indices]
        cases.append(('seed0-random',prepare(np.random.default_rng(0).integers(0,256,(512,512,3),dtype=np.uint8))))
        for name,value in cases:
            box = np.array([[185.,120.,380.,445.]],np.float32)
            with torch.no_grad():
                x = torch.from_numpy(value)
                ref_embedding,ref_outputs = reference(x) if args.sample == 'efficient_sam' else reference(x,torch.from_numpy(box))
                native_embedding = encoder(x)
                native_outputs = decoder(native_embedding) if args.sample == 'efficient_sam' else decoder(native_embedding,torch.from_numpy(box))
                isolated = decoder(ref_embedding) if args.sample == 'efficient_sam' else decoder(ref_embedding,torch.from_numpy(box))
            actual_embedding = sessions[0].run(None,{sessions[0].get_inputs()[0].name:value})[0]
            inputs = {'image_embeddings':ref_embedding.numpy()}
            if args.sample == 'mobile_sam': inputs['boxes'] = box
            actual_isolated = sessions[1].run(None,inputs)
            inputs['image_embeddings'] = actual_embedding
            actual_pipeline = sessions[1].run(None,inputs)
            comparisons = {'source_vs_native_embedding':compare(ref_embedding.numpy(),native_embedding.numpy()),
                           'source_vs_onnx_embedding':compare(ref_embedding.numpy(),actual_embedding)}
            for i,role in enumerate(('low_res_masks','iou_predictions')):
                comparisons['source_vs_native_'+role] = compare(ref_outputs[i].numpy(),native_outputs[i].numpy())
                comparisons['native_vs_onnx_'+role] = compare(isolated[i].numpy(),actual_isolated[i])
                comparisons['source_vs_connected_onnx_'+role] = compare(ref_outputs[i].numpy(),actual_pipeline[i])
            report['cases'].append({'input':name,'tensor_sha256':hashlib.sha256(value.tobytes()).hexdigest(),
                                    'comparisons':comparisons})
            if not all(x['passed'] for x in comparisons.values()):
                raise ValueError(f'Float pair parity failed for {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output/'host-smoke.json').write_text(json.dumps(report,indent=2)+'\n')

    boxes = {}
    if args.sample == 'mobile_sam':
        annotations = json.loads(args.annotations.read_text())
        selected = {int(path.stem) for path in images}
        for annotation in annotations['annotations']:
            key = annotation['image_id']
            if key in selected and not annotation.get('iscrowd') and annotation['area']>0:
                if key not in boxes or annotation['area']>boxes[key]['area']:
                    boxes[key] = annotation
        del annotations
        if set(boxes) != selected:
            raise ValueError('Some selected images lack real non-crowd box prompts')
    names = ['encoder','image_embeddings']+(['boxes'] if args.sample == 'mobile_sam' else [])
    for name in names:
        (args.output/name).mkdir()
    manifest = {'sample':args.sample,'count':len(images),'encoder_onnx_sha256':sha(args.encoder),
                'decoder_onnx_sha256':sha(args.decoder),'producer_sha256':sha(Path(__file__)),
                'embedding_source':'real float encoder ONNX on each independent photograph',
                'normalization':'RGB /255' if args.sample == 'efficient_sam' else 'RGB ImageNet mean/std',
                'annotations_sha256':sha(args.annotations) if args.annotations else None,'records':[]}
    for i,path in enumerate(images):
        image = cv2.imread(str(path))
        value = prepare(image)
        embedding = sessions[0].run(None,{sessions[0].get_inputs()[0].name:value})[0]
        arrays = {'encoder':value,'image_embeddings':embedding}
        record = {'source':str(path),'source_sha256':sha(path),'tensors':{}}
        if args.sample == 'mobile_sam':
            annotation = boxes[int(path.stem)]
            x,y,w,h = annotation['bbox']
            height,width = image.shape[:2]
            arrays['boxes'] = np.array([[x/width*512,y/height*512,(x+w)/width*512,(y+h)/height*512]],np.float32).clip(0,512)
            record['coco_annotation_id'] = annotation['id']
        for role,array in arrays.items():
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Calibration requires finite float32 tensors')
            destination = args.output/role/f'{i:04d}.npy'
            np.save(destination,array,allow_pickle=False)
            record['tensors'][role] = {'path':str(destination),'sha256':sha(destination),
                                       'shape':list(array.shape),'dtype':str(array.dtype),
                                       'min':float(array.min()),'max':float(array.max())}
        manifest['records'].append(record)
    (args.output/'calibration.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'host_pipeline':'passed','calibration_count':len(images),'output':str(args.output)}))


if __name__ == '__main__':
    main()
