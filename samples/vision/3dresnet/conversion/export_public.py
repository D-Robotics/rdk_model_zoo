# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export the trained torchvision R3D-18 with genuine temporal calibration."""
from pathlib import Path
import argparse, importlib, json
from utils.tools.torchvision_conversion import identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-record', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-commit', required=True)
    a = p.parse_args()
    import numpy as np, torch, torchvision, onnx, onnxruntime as ort
    from torchvision.models.video import r3d_18, R3D_18_Weights
    pixel = importlib.import_module('samples.vision.3dresnet.runtime.python.public')
    if torchvision.__version__.split('+')[0] != '0.23.0':
        raise ValueError('This source export requires torchvision 0.23.0')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    checkpoint = json.loads(a.checkpoint_record.read_text())
    if identity(checkpoint['path'])['sha256'] != checkpoint['sha256']:
        raise ValueError('Source checkpoint changed')
    model = r3d_18(weights=None).cpu().eval()
    model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)

    class ExplicitPool(torch.nn.Module):
        def __init__(self, source):
            super().__init__(); self.source = source

        def forward(self, images):
            m = self.source
            x = m.layer4(m.layer3(m.layer2(m.layer1(m.stem(images)))))
            # Reduce the actual temporal and spatial dimensions; keep all 400 logits.
            return m.fc(x.flatten(2).mean(dim=2))

    wrapper = ExplicitPool(model).eval()
    data = json.loads(a.videos.read_text())
    if data['status'] != 'acquired' or data['count'] != 100:
        raise ValueError('Expected 100 frozen, genuine training videos')
    a.output.mkdir(exist_ok=False); cal = a.output/'calibration'; cal.mkdir()
    rows = []
    for i, row in enumerate(data['records']):
        path = Path(row['path'])
        if identity(path)['sha256'] != row['sha256']:
            raise ValueError('Calibration video changed')
        value, context = pixel.preprocess(path)
        target = cal/f'{i:04d}.npy'; np.save(target, value, allow_pickle=False)
        rows.append({'video': identity(path), 'tensor': identity(target), 'context': context,
                     'shape': list(value.shape), 'dtype': str(value.dtype)})
    manifest = {'count': 100, 'files': rows, 'producer': identity(__file__),
                'source_video_manifest': identity(a.videos), 'dataset_scope': data['scope'],
                'calibration_domain': '16 distinct genuine frames; source-normalized RGB F32 NCTHW',
                'normalization': {'scale': 1/255, 'mean': pixel.MEAN, 'std': pixel.STD}}
    mp = a.output/'calibration.manifest.json'; mp.write_text(json.dumps(manifest, indent=2))
    cases = [('real-'+str(i), np.load(rows[i]['tensor']['path'], allow_pickle=False)) for i in (0, 49, 99)]
    cases.append(('random-seed0', np.random.default_rng(0).uniform(-1, 1, (1, 3, 16, 112, 112)).astype(np.float32)))
    expected = []
    with torch.inference_mode():
        for _, value in cases:
            x = torch.from_numpy(value); original = model(x).numpy(); equivalent = wrapper(x).numpy()
            np.testing.assert_allclose(equivalent, original, atol=1e-4, rtol=1e-3)
            expected.append(original)
        graph = a.output/'model.onnx'
        torch.onnx.export(wrapper, torch.from_numpy(cases[0][1]), str(graph), input_names=['images'],
                          output_names=['logits'], opset_version=11, do_constant_folding=False, dynamo=False)
    onnx.checker.check_model(str(graph))
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
    checks = []
    for (name, value), wanted in zip(cases, expected, strict=True):
        got = session.run(None, {'images': value})[0]
        if got.dtype != np.float32 or got.shape != (1, 400) or not np.isfinite(got).all():
            raise ValueError('Expected all 400 finite F32 source logits')
        np.testing.assert_allclose(got, wanted, atol=1e-4, rtol=1e-3)
        checks.append({'input': name, 'status': 'passed', 'comparisons': [
            {'name': 'logits', 'shape': list(got.shape), 'dtype': str(got.dtype),
             'max_abs': float(np.max(np.abs(got-wanted))), 'passed': True}]})
    labels = R3D_18_Weights.KINETICS400_V1.meta['categories']
    (a.output/'labels.json').write_text(json.dumps(labels, indent=2))
    report = {'schema_version': 1, 'status': 'passed', 'native_commit': a.native_commit,
              'source': {'repository': 'https://github.com/pytorch/vision', 'version': '0.23.0',
                         'commit': '824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8', 'license': 'BSD-3-Clause',
                         'checkpoint': checkpoint}, 'onnx': identity(graph), 'exporter': identity(__file__),
              'source_implementation': identity(__import__('inspect').getfile(r3d_18)),
              'calibration_manifest': identity(mp), 'input': {'name': 'images', 'shape': [1,3,16,112,112], 'dtype': 'float32'},
              'outputs': [{'name': 'logits', 'shape': [1,400], 'dtype': 'float32'}],
              'parameter_count': sum(p.numel() for p in model.parameters()), 'normalization': manifest['normalization'],
              'cases': checks, 'atol': 1e-4, 'rtol': 1e-3,
              'wrapper_parity': 'all original 400 logits; exact layers with equivalent flattened spatial/temporal mean',
              'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'onnxruntime': ort.__version__},
              'board': 'not-run', 'dataset_accuracy': 'not-run'}
    (a.output/'export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'passed', 'real_videos': 100, 'cases': len(cases)}), flush=True)


if __name__ == '__main__': main()
