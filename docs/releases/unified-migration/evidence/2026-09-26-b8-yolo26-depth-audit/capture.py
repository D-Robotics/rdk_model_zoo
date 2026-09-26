"""Reproduce source-only YOLO26 Depth migration findings; no SDK or hardware."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[5]
PINS = {'x5': 'ac115717197920355fc390bb04299b20e6436864',
        's': '380e1a2bf42041af54be6f34935e50197cfadff9'}
REL = Path('samples/vision/yolo26_depth')


def isolated(path, names, extra=None):
    """Execute selected source definitions only, without importing the board SDK."""
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    ns = {'np': np, 'cv2': cv2, 'dataclass': dataclass}
    ns.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def main():
    files = []
    for platform, pin in PINS.items():
        base = ROOT/'platforms'/platform/REL
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', pin, str(REL)], cwd=ROOT, text=True).splitlines()
        for rel in paths:
            data = subprocess.check_output(['git', 'show', f'{pin}:{rel}'], cwd=ROOT)
            actual = ROOT/'platforms'/platform/rel
            same = actual.read_bytes() == data
            files.append({'path': str(actual.relative_to(ROOT)), 'source_sha256': hashlib.sha256(data).hexdigest(), 'matches_pin': same})
    assert all(row['matches_pin'] for row in files), 'source snapshot drift'
    xs = ROOT/'platforms/x5'/REL
    ss = ROOT/'platforms/s'/REL
    common = sorted(str(p.relative_to(xs)) for p in xs.rglob('*') if p.is_file()
                    and (ss/p.relative_to(xs)).is_file() and p.read_bytes() == (ss/p.relative_to(xs)).read_bytes())
    manifests = {}
    for platform in PINS:
        manifest = yaml.safe_load((ROOT/f'platforms/{platform}/docs/release/models.yaml').read_text())
        def find(value):
            if isinstance(value, dict):
                if value.get('id') == 'yolo26_depth':
                    return value
                for v in value.values():
                    result = find(v)
                    if result is not None: return result
            if isinstance(value, list):
                for v in value:
                    result = find(v)
                    if result is not None: return result
        family = find(manifest)
        manifests[platform] = family['assets']
    assert len(manifests['x5']) == 5 and len(manifests['s']) == 15
    configs = {}
    for platform, base in [('x5', xs), ('s', ss)]:
        configs[platform] = []
        for path in sorted((base/'conversion/ptq_yamls').glob('*.yaml')):
            c = yaml.safe_load(path.read_text())
            configs[platform].append({'path': str(path.relative_to(ROOT)), 'onnx': c['model_parameters']['onnx_model'],
                'calibration': c['calibration_parameters']['cal_data_dir'],
                'calibration_dtype': c['calibration_parameters']['cal_data_type'],
                'runtime_input_type': c['input_parameters']['input_type_rt']})
    assert len(configs['x5']) == 5 and len(configs['s']) == 24
    s_api = isolated(ss/'runtime/python/yolo26_depth.py', {'letterbox', 'featuremap'})
    x_api = isolated(xs/'runtime/python/yolo26_depth.py', {'LetterboxGeometry', 'letterbox'})
    image = np.arange(37*23*3, dtype=np.uint8).reshape(37,23,3)
    a, ga = x_api['letterbox'](image,768)
    b, gb = s_api['letterbox'](image,768)
    assert np.array_equal(a,b)
    thin = np.zeros((1,10000,3), np.uint8)
    edge = {}
    for name, api in [('x5',x_api), ('s',s_api)]:
        try: api['letterbox'](thin,768)
        except cv2.error: edge[name] = 'cv2.error: source rounds resized height to zero'
        else: raise AssertionError('expected source zero-height failure')
    # Constant-map fixture: interpolation is identity in value; no Torch/SDK needed.
    def constant_resize(depth, height, width):
        assert np.all(depth == depth.flat[0])
        return np.full((height,width), depth.flat[0], np.float32)
    evaluator = isolated(ss/'evaluator/eval_sunrgbd.py', {'CALIBRATION','postprocess_raw'}, {'resize_bilinear': constant_resize})
    log = np.zeros((1,2,2,1), np.float32)
    wrong = evaluator['postprocess_raw'](log, {'original_hw':[2,2]}, 'deployment_scale_fill',768,'s')
    expected = np.exp(log).squeeze()
    assert not np.allclose(wrong,expected)
    audit = {
        'schema_version':'1.0', 'source_pins':PINS, 'files':files,
        'byte_identical_cross_platform_files':common, 'published_assets':manifests, 'conversion_configs':configs,
        'reproductions': {
            'letterbox_37x23_x5_s_identical':bool(np.array_equal(a,b)),
            'source_extreme_aspect_failure':edge,
            'python_cpp_rounding':{'source_dimension':5,'ratio':0.5,'python_round':round(2.5),'cpp_std_round_positive':3,
                'example_image_hw':[1536,5], 'target_size':768},
            's_evaluator_nv12_counterexample':{'variant':'s','calibrated_log_depth':0.0,
                'runtime_exp':float(expected.flat[0]),'source_lite_evaluator_result':float(wrong.flat[0]),
                'meaning':'Source evaluator applies lite calibration again to an already calibrated NV12 log-depth; it supports only deployment_scale_fill.'}},
        'findings':[
            {'id':'YD-A1','area':'S root/conversion README and Python module docstring',
             'finding':'Claims exp and resize4x in graph. Export depth_log_forward ends after clip/scale/bias; both runtimes exp and resize on CPU.',
             'decision':'Canonical docs follow source executable boundary: calibrated log-depth for all X5 and S n/s/m; raw logits only S l/x. Runtime artifact metadata remains unobserved.'},
            {'id':'YD-A2','area':'S root README performance',
             'finding':'Table reports s cosine 0.9984 while prose says all variants pass >=0.999. Evaluator historical table differs and example selects experimental n-lite.',
             'decision':'Retain source tables with provenance and contradiction; no new all-pass claim or assumed published-profile association.'},
            {'id':'YD-A3','area':'S conversion export.py, YAMLs, README',
             'finding':'Exporter emits yolo26n-depth_op11_log.onnx; nine NV12 YAMLs expect yolo26n-depth-log.onnx pattern. All 24 configs expect ./calibration, docs produce calibration_nv12/calibration_lite. Two documented exports reuse an exist_ok=False output directory. Extractor docs use nonexistent --src/--out flags.',
             'decision':'Unify exporter naming and make profile-specific compilation inputs explicit; test actual parsers and every emitted config path. Preserve 9 experimental lite n/s/m configs as experimental, not release assets.'},
            {'id':'YD-A4','area':'S evaluator',
             'finding':'README claims board execution; eval_sunrgbd.py only reads float_raw/quant_raw NPZ and implements lite scale-fill for all five variants; cannot evaluate release n/s/m correctly.',
             'decision':'Consolidate offline metrics and support explicit raw/calibrated boundary with matching geometry; keep X5 validator protocol and numerical comparison capability.'},
            {'id':'YD-A5','area':'Python and C++ geometry',
             'finding':'Both Python sources fail with zero resized dimension for extreme aspect ratios; C++ std::round differs from Python ties-to-even.',
             'decision':'Adopt explicit shared geometry contract, reject collapsed dimensions clearly; use ties-to-even for native parity and regression fixtures.'},
            {'id':'YD-A6','area':'X5 C++ runtime',
             'finding':'Mixed SDK ownership and task logic; output copied as contiguous ignoring aligned shape/strides; input memcpy lacks explicit capacity/layout checks. Rendering uses order-statistic percentiles unlike NumPy interpolated percentiles.',
             'decision':'Separate SDK owner from three stages/predict; validate or handle actual memory layout; align rendering semantics, preserve raw float32 log-depth requirement.'},
            {'id':'YD-A7','area':'S calibration docstring',
             'finding':'Claims exact tensor equivalence between direct RGB calibration and NV12 conversion; chroma subsampling/color conversion means exact equality is not established.',
             'decision':'Describe matching geometry/range and intentional calibration representation, not byte equality.'}],
        'implementation_contract':{
            'targets':['x5','s100','s100p','s600'], 'variants':['n','s','m','l','x'], 'default_variant':'n',
            'cpp_targets':['x5'], 's_lite_variants':['l','x'],
            'nv12_physical_transport':'source runtime supplies one flat uint8 packed buffer on X5 AND S, not generic S split Y/UV',
            'lite_physical_transport':'float32 RGB NCHW [1,3,768,768], INTER_LINEAR scale-fill /255',
            'lite_calibration':{'l':[1.0,-0.2498779296875],'x':[1.0,-0.316650390625]},
            'clip':[-4.0,5.0], 'output_export_shape':[1,192,192,1],
            'output_meaning':'relative depth; no metric calibration claim',
            'stages':'pre_process, forward, post_process, predict only in task class; metadata/SDK/timing/render/files outside',
            'preserve_capabilities':['20 published assets and X5 hashes','X5 C++','all 29 conversion YAMLs with experimental distinction','attention/depth export patches','SUNRGBD archive extraction and both preparation protocols','offline metrics/alignment/fidelity and X5 eval_numeric','bus.jpg','all six levels bilingual README']},
        'status':'source audit only; canonical implementation and validation pending; no acceptance',
        'not_run':['board','real SDK/native build','artifact download/metadata inspection','Torch/ONNX export','OE compilation','SUNRGBD metrics']}
    output = Path(__file__).resolve().parents[1]/'2026-09-26-b8-yolo26-depth-audit.json'
    output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'source_files':len(files),'identical_cross_platform_files':common,'assets':{k:len(v) for k,v in manifests.items()},
                      'configs':{k:len(v) for k,v in configs.items()}, 'reproductions':audit['reproductions'],'output':str(output)},indent=2))


if __name__ == '__main__': main()
