# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate the S split graph with real PartA -> proposals/ROIAlign -> PartB data."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--image', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.image) < 3 or len(set(args.image)) != len(args.image):
        raise ValueError('Supply at least three distinct real images')
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchvision
    from torchvision.models.detection.image_list import ImageList
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    from export_split import load_model, pack_bank, CHECKPOINT_SHA256, verify_checkpoint_cache
    from prepare_calibration import load_image_tensor, letterbox, rpn_middle, pooled_pad

    if not torch.__version__.startswith('2.6.0') or not torchvision.__version__.startswith('0.21.0'):
        raise ValueError('This frozen source recipe requires Torch 2.6.0 / TorchVision 0.21.0')
    torch.set_num_threads(2)
    model = load_model()
    if verify_checkpoint_cache() != CHECKPOINT_SHA256:
        raise ValueError('Official source checkpoint checksum differs')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = args.onnx/'part_a_f0only.op16.onnx'
    b = args.onnx/'part_b_packed.op16.onnx'
    sessions = [ort.InferenceSession(str(x),sess_options=options,providers=['CPUExecutionProvider']) for x in (a,b)]
    mean = torch.tensor(model.transform.image_mean).view(3,1,1)
    std = torch.tensor(model.transform.image_std).view(3,1,1)
    cases = []
    for image in args.image:
        canvas,_ = letterbox(load_image_tensor(image))
        cases.append((str(image),((canvas-mean)/std)[None]))
    generator = torch.Generator().manual_seed(0)
    cases.append(('seed0-random',torch.randn(1,3,320,320,generator=generator)))
    report = {'status':'failed','torch':torch.__version__,'torchvision':torchvision.__version__,
              'source_checkpoint_sha256':CHECKPOINT_SHA256,
              'onnx_sha256':{'parta':hashlib.sha256(a.read_bytes()).hexdigest(),
                             'partb':hashlib.sha256(b.read_bytes()).hexdigest()},
              'absolute_tolerance':1e-3,'minimum_cosine':.99999,
              'pipeline':'ONNX PartA -> source RPN selection / ROIAlign -> ONNX packed PartB',
              'cases':[],'board':'not-run','accuracy':'not-run'}

    def compare(expected,observed):
        expected = expected.numpy()
        assert expected.shape == observed.shape and np.isfinite(observed).all()
        x,y = expected.astype(np.float64).ravel(),observed.astype(np.float64).ravel()
        error = float(np.max(np.abs(x-y)))
        cosine = float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
        return {'max_abs_error':error,'cosine':cosine,'passed':error<1e-3 and cosine>=.99999}

    try:
        with torch.no_grad():
            for name,value in cases:
                feature_dict = model.backbone(value)
                features = list(feature_dict.values())
                reference_obj,reference_delta = concat_box_prediction_layers(*model.rpn.head(features))
                obj,delta,feat = sessions[0].run(None,{'images':value.numpy()})
                anchors = model.rpn.anchor_generator(ImageList(value,[(320,320)]),features)[0]
                rois = rpn_middle(model,torch.from_numpy(obj),torch.from_numpy(delta),anchors,[(320,320)])
                pooled = model.roi_heads.box_roi_pool({'0':torch.from_numpy(feat)},[rois],[(320,320)])
                padded = torch.from_numpy(pooled_pad(pooled))
                expected_cls,expected_reg = model.roi_heads.box_predictor(model.roi_heads.box_head(padded))
                cls,reg = sessions[1].run(None,{'bank':pack_bank(padded).numpy()})
                comparisons = {'rpn_obj':compare(reference_obj,obj),'rpn_deltas':compare(reference_delta,delta),
                               'feat0':compare(feature_dict['0'],feat),
                               'connected_class_logits':compare(expected_cls,cls[0,:,:,0].T),
                               'connected_box_regression':compare(expected_reg,reg[0,:,:,0].T)}
                report['cases'].append({'input':name,'tensor_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                                        'proposals':len(rois),'comparisons':comparisons})
                if not all(x['passed'] for x in comparisons.values()):
                    raise ValueError(f'Connected split graph parity failed: {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
