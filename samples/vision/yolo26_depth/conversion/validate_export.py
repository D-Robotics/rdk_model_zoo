# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compare the source Depth head, native boundary and exported ONNX on real inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--image', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.image) < 3 or len(set(args.image)) != len(args.image):
        raise ValueError('Supply at least three distinct real images')
    if args.output.exists():
        raise FileExistsError(args.output)
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    import ultralytics
    from ultralytics import YOLO
    import export as native_export

    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    from samples.vision.yolo26_depth.conversion.prepare_calibration import prepare_tensor

    if ultralytics.__version__ != '8.4.105':
        raise ValueError('This source gate requires ultralytics==8.4.105')
    torch.set_num_threads(2)
    source = YOLO(str(args.weights), task='depth').model.cpu().float().eval()
    for module in source.modules():
        if type(module).__name__ == 'Depth':
            module.export = False
    patched = copy.deepcopy(source)
    native_export.torch, native_export.F = torch, F
    patches = native_export.patch_model(patched, 'log')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    cases = []
    for image in args.image:
        value = cv2.imread(str(image))
        tensor, _ = prepare_tensor(value, 's100', 'nv12')
        cases.append((str(image), tensor, hashlib.sha256(image.read_bytes()).hexdigest()))
    cases.append(('seed0-random', np.random.default_rng(0).random((1,3,768,768), dtype=np.float32), None))
    report = {'status': 'failed', 'source_version': ultralytics.__version__,
              'torch_version': torch.__version__, 'patch_counts': patches,
              'source_checkpoint_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              'atol': 1e-4, 'rtol': 1e-4, 'minimum_cosine': .99999, 'cases': [],
              'board': 'not-run', 'accuracy': 'not-run'}

    def compare(expected, observed):
        assert expected.shape == observed.shape
        assert np.isfinite(observed).all()
        a, b = expected.astype(np.float64).ravel(), observed.astype(np.float64).ravel()
        cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        np.testing.assert_allclose(observed, expected, atol=1e-4, rtol=1e-4)
        assert cosine >= .99999, cosine
        return {'max_abs_error': float(np.max(np.abs(a-b))), 'cosine': cosine,
                'shape': list(expected.shape)}

    try:
        for name, tensor, image_sha in cases:
            with torch.no_grad():
                reference = source(torch.from_numpy(tensor)).numpy()
                boundary = patched(torch.from_numpy(tensor)).numpy()
            actual = session.run(None, {session.get_inputs()[0].name:tensor})
            assert len(actual) == 1
            row = {'input': name, 'image_sha256':image_sha,
                   'tensor_sha256':hashlib.sha256(tensor.tobytes()).hexdigest()}
            row['source_vs_native_exp'] = compare(reference, np.exp(boundary).transpose(0,3,1,2))
            row['native_vs_onnx'] = compare(boundary, actual[0])
            report['cases'].append(row)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
