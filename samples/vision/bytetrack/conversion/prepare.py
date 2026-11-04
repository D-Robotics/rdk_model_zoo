# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned YOLOv5x v7 detector for ByteTrack S100/S100P/S600.

Separate calibration/export processes avoid upstream utils package shadowing.
No dependencies or source weights are downloaded by this command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
PROFILES = {'x-v7.0': ('915bbf294bb74c859f0b41f1c23bc395014ea679', '9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859')}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def calibrate(args):
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from samples.vision.yolov5.runtime.python.detection import resized_image
    source = args.images.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['count'] == 200 and manifest['seed'] == 42 and (len(manifest['images']) == 200)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'rgb_f32').mkdir()
    (out / 'normalized').mkdir()
    rows = []
    for row in manifest['images']:
        p = source / row['name']
        assert sha(p) == row['sha256'], p
        pixels = resized_image(cv2.imread(str(p)), 672, 672, 0)
        x = np.ascontiguousarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None], dtype=np.float32)
        assert x.shape == (1, 3, 672, 672) and np.isfinite(x).all()
        dest = out / 'rgb_f32' / (p.stem + '.bin')
        x.tofile(dest)
        np.save(out/'normalized'/(p.stem+'.npy'), x/np.float32(255))
        rows.append({'source': str(p), 'source_sha256': row['sha256'], 'tensor': str(dest), 'sha256': sha(dest), 'shape': list(x.shape), 'dtype': 'float32', 'min': float(x.min()), 'max': float(x.max())})
    report = {'count': 200, 'seed': 42, 'source_manifest_sha256': sha(source / 'manifest.json'), 'runtime_preprocess_sha256': sha(Path(__file__).resolve().parents[4] / 'samples/vision/yolov5/runtime/python/detection.py'), 'geometry': 'S stretch INTER_NEAREST', 'color': 'RGB', 'normalization': 'raw 0..255; compiler /255 once', 'images': rows}
    (out / 'calibration-manifest.json').write_text(json.dumps(report, indent=2) + '\n')

def export(args):
    import os
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    source, checkpoint = (args.source.resolve(), args.checkpoint.resolve())
    commit, digest = PROFILES[args.variant]
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    assert sha(checkpoint) == digest
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = (payload.get('ema') or payload['model']).float().eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Upsample) and (not hasattr(layer, 'recompute_scale_factor')):
            layer.recompute_scale_factor = None
    head = model.model[-1]
    head.export = False
    assert head.nc == 80 and head.nl == 3 and (head.na == 3)
    anchors = (head.anchors * head.stride.view(-1, 1, 1)).detach().numpy()
    np.testing.assert_allclose(anchors.reshape(-1), [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
    cal = json.loads((args.calibration.resolve() / 'calibration-manifest.json').read_text())
    cases = []
    for i in [0, 52, 143]:
        row = cal['images'][i]
        assert sha(row['tensor']) == row['sha256']
        cases.append((row['source'], np.fromfile(row['tensor'], np.float32).reshape(1, 3, 672, 672) / 255))
    cases.append(('random-seed42', np.random.default_rng(42).random((1, 3, 672, 672), dtype=np.float32)))
    references = []
    with torch.no_grad():
        for _, x in cases:
            result = model(torch.from_numpy(x))
            raw = result[1]
            assert len(raw) == 3 and np.isfinite(result[0].numpy()).all()
            references.append([r.permute(0, 2, 3, 1, 4).reshape(1, r.shape[2], r.shape[3], 255).numpy().copy() for r in raw])

    def raw_heads(self, features):
        return tuple((self.m[i](features[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)))
    head.forward = types.MethodType(raw_heads, head)
    names = ['small', 'medium', 'big']
    graph = out / 'model.onnx'
    with torch.no_grad():
        for (_, x), expected in zip(cases, references):
            for actual, target in zip(model(torch.from_numpy(x)), expected):
                np.testing.assert_allclose(actual.numpy(), target, rtol=0, atol=0)
        torch.onnx.export(model, torch.from_numpy(cases[0][1]), str(graph), input_names=['images'], output_names=names, opset_version=11, dynamo=False, do_constant_folding=False)
    onnx.checker.check_model(str(graph))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 3, 672, 672]
    assert [o.shape for o in session.get_outputs()] == [[1, 84, 84, 255], [1, 42, 42, 255], [1, 21, 21, 255]]
    checks = []
    for (name, x), expected in zip(cases, references):
        actual = session.run(None, {'images': x})
        errors = []
        for a, b in zip(actual, expected):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a, b, rtol=0.001, atol=0.0001)
            errors.append(float(np.max(np.abs(a - b))))
        checks.append({'input': name, 'finite': True, 'head_max_abs_error': errors})
    march = {'s100':'nash-e', 's100p':'nash-m', 's600':'nash-p'}[args.target]
    prefix = 'bytetrack_yolov5x_v7_672_nv12_int16_cpuptq_o1_20261104_r02_' + march.replace('-', '')
    config = {'model_parameters': {'onnx_model':str(graph), 'march':march,
        'working_dir':str(out/'compiled'), 'output_model_file_prefix':prefix},
        'input_parameters': {'input_name':'images', 'input_type_rt':'nv12',
        'input_type_train':'rgb', 'input_space_and_range':'bt601_video',
        'input_shape':'1x3x672x672', 'input_layout_train':'NCHW',
        'norm_type':'data_scale', 'scale_value':str(1/255)},
        'calibration_parameters': {'cal_data_dir':str(args.calibration.resolve()/'normalized'),
         'calibration_type':'max', 'max_percentile':.9999,
        'optimization':'set_all_nodes_int16'},
        'compiler_parameters': {'compile_mode':'latency', 'core_num':1,
        'optimize_level':'O1', 'jobs':2, 'debug':True,
        'extra_params':{'input_no_padding':True,'output_no_padding':True}}}
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    report = {'variant': args.variant, 'source_commit': commit, 'checkpoint_sha256': digest, 'license': 'GPL-3.0', 'onnx_sha256': sha(graph), 'config_sha256': sha(out / 'config.yaml'), 'exporter_sha256': sha(__file__), 'checkpoint_size': checkpoint.stat().st_size, 'torch': torch.__version__, 'onnxruntime': ort.__version__, 'raw_head_patch_bit_exact': True, 'reference': 'official eval decoded+raw outputs; raw heads permuted to NHWC before export-only patch', 'checks': checks, 'passed': True, 'atol': 0.0001, 'rtol': 0.001, 'output_names': names, 'prefix': prefix}
    (out / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cal = commands.add_parser('calibrate')
    cal.add_argument('--images', type=Path, required=True)
    cal.add_argument('--output', type=Path, required=True)
    exp = commands.add_parser('export')
    exp.add_argument('--source', type=Path, required=True)
    exp.add_argument('--checkpoint', type=Path, required=True)
    exp.add_argument('--variant', choices=PROFILES, default='x-v7.0')
    exp.add_argument('--target', choices=('s100','s100p','s600'), required=True)
    exp.add_argument('--calibration', type=Path, required=True)
    exp.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return calibrate(args) if args.command == 'calibrate' else export(args)
if __name__ == '__main__':
    main()

