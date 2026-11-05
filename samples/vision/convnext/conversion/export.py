#!/usr/bin/env python3
"""Export an explicitly supplied timm ConvNeXt checkpoint and verify ONNX parity."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('atto', 'femto', 'pico', 'nano'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=224)
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error('--resolution must be positive')
    if args.output.exists() or args.output.with_suffix('.export.json').exists():
        parser.error('output already exists; use a new conversion workspace')

    import numpy as np
    import onnx
    import onnxruntime as ort
    import timm
    import torch
    from onnxsim import simplify
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model = timm.create_model('convnext_' + args.variant, pretrained=False, exportable=True)
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    model.eval()
    sample = torch.randn(1, 3, args.resolution, args.resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(sample).numpy()
        torch.onnx.export(model, sample, str(args.output), input_names=['data'],
            output_names=['output'], opset_version=11, do_constant_folding=True, dynamo=False)
    raw = onnx.load(str(args.output))
    simplified, valid = simplify(raw)
    if not valid:
        raise RuntimeError('ONNX simplification parity check failed')
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'data': sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    if actual.shape != (1, 1000) or not np.isfinite(actual).all():
        raise RuntimeError('Exported graph does not produce finite 1000-class logits')
    receipt = {'schema_version': 1, 'status': 'passed', 'variant': args.variant,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parameter_count': sum(p.numel() for p in model.parameters()), 'opset': 11,
        'input': {'name': 'data', 'shape': list(sample.shape), 'dtype': 'float32'},
        'outputs': [{'name': 'output', 'shape': list(actual.shape), 'dtype': 'float32'}],
        'torch_onnx_max_absolute_error': float(np.max(np.abs(actual-expected))),
        'method': 'one deterministic random input; PyTorch vs ONNX Runtime CPU; not dataset accuracy or board inference',
        'versions': {'torch': torch.__version__, 'timm': timm.__version__,
                     'onnx': onnx.__version__, 'onnxruntime': ort.__version__}}
    args.output.with_suffix('.export.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
