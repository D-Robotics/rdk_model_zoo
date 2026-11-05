#!/usr/bin/env python3
"""Export a pinned local timm checkpoint and check two-input ONNX parity."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Exact installed timm model/tag')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-sample', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--opset', type=int, default=11)
    args = parser.parse_args()
    if args.resolution <= 0 or args.resolution % 2:
        parser.error('resolution must be positive and even')
    if args.output.exists() or args.output.with_suffix('.export.json').exists():
        parser.error('use a new output directory for every attempt')

    import numpy as np
    import onnx
    import onnxruntime as ort
    import timm
    import torch
    from onnxsim import simplify
    from safetensors.torch import load_file
    from timm.layers import set_fused_attn
    from timm.utils import reparameterize_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    set_fused_attn(False)
    model = timm.create_model(args.model, pretrained=False, exportable=True)
    state = load_file(str(args.checkpoint)) if args.checkpoint.suffix == '.safetensors' else torch.load(
        args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    cfg = model.pretrained_cfg
    mean = np.asarray(cfg['mean'], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(cfg['std'], dtype=np.float32).reshape(1, 3, 1, 1)
    image = np.fromfile(args.calibration_sample, dtype=np.float32).reshape(1, 3, args.resolution, args.resolution)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError('calibration sample must contain finite raw RGB values in 0..255')
    samples = [torch.randn(1, 3, args.resolution, args.resolution),
               torch.from_numpy((image / 255 - mean) / std)]
    parameters = sum(p.numel() for p in model.parameters())
    with torch.inference_mode():
        expected = [model(sample).numpy() for sample in samples]
        model = reparameterize_model(model, inplace=False).eval()
        for sample, reference in zip(samples, expected):
            np.testing.assert_allclose(model(sample).numpy(), reference, rtol=1e-3, atol=1e-4)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, samples[0], str(args.output), input_names=['data'],
            output_names=['output'], opset_version=args.opset, do_constant_folding=True, dynamo=False)
    graph, valid = simplify(onnx.load(str(args.output)))
    if not valid:
        raise RuntimeError('ONNX simplification parity check failed')
    onnx.checker.check_model(graph)
    onnx.save(graph, str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    errors = []
    for sample, reference in zip(samples, expected):
        actual = session.run(None, {'data': sample.numpy()})[0]
        if actual.shape != (1, 1000) or not np.isfinite(actual).all():
            raise RuntimeError('expected finite ImageNet-1k logits')
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
        errors.append(float(np.max(np.abs(actual - reference))))
    source = Path(inspect.getfile(type(model)))
    receipt = {'schema_version': 1, 'status': 'passed', 'model': args.model,
        'checkpoint_sha256': sha(args.checkpoint), 'onnx_sha256': sha(args.output),
        'exporter_sha256': sha(Path(__file__)), 'implementation_sha256': sha(source),
        'parameter_count': parameters, 'deployment_parameter_count': sum(p.numel() for p in model.parameters()),
        'opset': args.opset, 'input': {'name': 'data', 'shape': list(samples[0].shape), 'dtype': 'float32'},
        'outputs': [{'name': 'output', 'shape': [1, 1000], 'dtype': 'float32'}],
        'normalization': {'mean': list(cfg['mean']), 'std': list(cfg['std'])},
        'upstream_data_config': {key: cfg.get(key) for key in ('input_size', 'crop_pct', 'crop_mode', 'interpolation')},
        'torch_onnx_max_absolute_errors': errors, 'host_inputs': ['deterministic-random', 'frozen-calibration-image'],
        'method': 'Strict checkpoint load; reparameterization and PyTorch/ORT CPU parity; not dataset accuracy or board inference',
        'versions': {'torch': torch.__version__, 'timm': timm.__version__, 'onnx': onnx.__version__,
                     'onnxruntime': ort.__version__}}
    args.output.with_suffix('.export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'model': args.model, 'max_absolute_errors': errors}), flush=True)


if __name__ == '__main__':
    main()
