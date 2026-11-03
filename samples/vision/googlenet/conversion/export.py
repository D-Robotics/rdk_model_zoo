#!/usr/bin/env python3
"""Export official TorchVision GoogLeNet weights as a separate crop224 profile."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new attempt directory')
    checkpoint = identity(args.checkpoint)
    if checkpoint['sha256'] != '1378be20a8e875cf1568b8a71654e704449655e34711a959a38b04fb34905cef':
        raise ValueError('checkpoint differs from the frozen official weight')
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    if not torchvision.__version__.startswith('0.21.0'):
        raise ValueError('this recipe is frozen to TorchVision 0.21.0')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # The official weights builder enables transform_input=True and initially
    # loads auxiliary heads. Preserve that behavior before dropping unused heads.
    model = torchvision.models.googlenet(weights=None, aux_logits=True,
        transform_input=True, init_weights=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.aux_logits = False
    model.aux1 = model.aux2 = None
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    files = sorted(args.calibration.glob('*.rgb'))
    if len(files) != 200:
        raise ValueError('requires the frozen 200-image crop224 raw RGB calibration set')
    cases = [(str(i), (np.fromfile(files[i], dtype=np.float32).reshape(1, 3, 224, 224) / 255 - mean) / std)
             for i in (0, 52, 143)]
    cases.append(('random-seed0', np.random.default_rng(0).normal(size=(1, 3, 224, 224)).astype(np.float32)))
    args.output.mkdir(parents=True)
    graph = args.output / 'model.onnx'
    with torch.inference_mode():
        references = [model(torch.from_numpy(x)).numpy() for _, x in cases]
        torch.onnx.export(model, torch.from_numpy(cases[0][1]), str(graph),
            input_names=['data'], output_names=['output'], opset_version=11,
            do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(graph))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    rows = []
    for (name, x), reference in zip(cases, references):
        actual = session.run(None, {'data': x})[0]
        np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
        if actual.shape != (1, 1000) or not np.isfinite(actual).all():
            raise ValueError('expected finite 1000-class logits')
        np.testing.assert_array_equal(np.argsort(actual[0])[-5:], np.argsort(reference[0])[-5:])
        rows.append({'name': name, 'max_abs': float(np.max(np.abs(actual-reference))), 'status': 'passed'})
    report = {'schema_version': 1, 'status': 'passed', 'checkpoint': checkpoint,
        'onnx': identity(graph), 'exporter': identity(__file__), 'implementation': identity(inspect.getfile(type(model))),
        'source_commit': '7af698794eded568735f9519593603c1ec889eba',
        'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                     'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
        'cases': rows, 'atol': 1e-4, 'rtol': 1e-3, 'transform_input': True,
        'normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'preprocessing': 'Pillow bicubic short256, rounded center crop224, ImageNet normalization; independent deployment profile',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'board': 'not-run', 'dataset_accuracy': 'not-run'}
    (args.output / 'host-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'cases': rows}), flush=True)


if __name__ == '__main__':
    main()
