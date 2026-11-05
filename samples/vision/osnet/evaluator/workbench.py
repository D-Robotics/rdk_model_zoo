"""OSNet-specific Workbench receipts and local technical acceptance (no publishing)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from samples.vision.osnet.evaluator.metrics import records
from samples.vision.osnet.model.artifacts import require_artifact


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def measure(args):
    import torch
    source = ROOT / 'samples/vision/osnet/third_party/osnet.py'
    spec = importlib.util.spec_from_file_location('official_osnet', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.osnet_x1_0(num_classes=751, pretrained=False).eval()
    if sha(args.weights) != '2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154':
        raise ValueError('Checkpoint identity differs')
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    parameters, macs = {}, []
    def hook(layer, inputs, output):
        for p in layer.parameters(recurse=False):
            parameters[id(p)] = p.numel()
        if isinstance(layer, torch.nn.Conv2d):
            macs.append(output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, torch.nn.Linear):
            macs.append(output.numel() * layer.in_features)
    handles = [m.register_forward_hook(hook) for m in model.modules()]
    torch.set_num_threads(4)
    with torch.inference_mode():
        model(torch.zeros(1, 3, 256, 128))
    for h in handles: h.remove()
    write(args.output, {'parameter_count': sum(parameters.values()), 'gflops': 2 * sum(macs) / 1e9,
                       'macs': sum(macs), 'input_shape': [1, 3, 256, 128], 'fused': False,
                       'measurement_method': 'Executed inference modules only; Conv2d/Linear MACs x2; excludes normalization, activations, pooling and classifier',
                       'torch': torch.__version__, 'source_sha256': sha(source), 'weights_sha256': sha(args.weights)})


def freeze(args):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        raise ValueError("Use the full immutable Docker image ID")
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    if dirty.strip(): raise ValueError('Commit the reviewed source before freezing a campaign')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    query, gallery = records(args.dataset/'query'), records(args.dataset/'bounding_box_test')
    if (len(query), len(gallery)) != (3368, 15913): raise ValueError('Incomplete Market1501 evaluation')
    require_artifact(args.model, 'spatial-rgb-late8')
    manifest = [{'path': str(p.relative_to(args.dataset)), 'sha256': sha(p)} for p, _, _ in query+gallery]
    manifest_path = args.workspace/'inputs/evaluation'/f'{args.build_id}-manifest.json'
    write(manifest_path, manifest)
    fixture_files = {name: {'path': str((args.fixture/name).resolve()), 'sha256': sha(args.fixture/name)}
                     for name in ('images.dat', 'spatial-rgb-late8.f32', 'manifest.json')}
    campaign = {'schema_version': 1, 'build_id': args.build_id, 'source_commit': commit,
                'guide': {'version': '2.8', 'sha256': sha(args.guide)}, 'model': read(args.measurement),
                'artifact': {'path': str(args.model.resolve()), 'sha256': sha(args.model), 'size_bytes': args.model.stat().st_size,
                             'variant': 'spatial-rgb-late8', 'precision_policy': 'mixed-int8-int16', 'format': 'bin', 'march': 'bayes-e'},
                'dataset': {'path': str(args.dataset.resolve()), 'name': 'Market1501', 'query': 3368, 'gallery': 15913,
                            'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path)},
                'evaluation': {'scope': 'market1501_single_query_l2_no_rerank', 'normalize': True, 'distance': 'squared Euclidean',
                               'reranking': False, 'exclude_pid': [-1], 'retain_pid_zero': True, 'exclude_same_pid_camera': True},
                'acceptance': {'max_rank1_drop': 0.01, 'max_map_drop': 0.01, 'scope': 'local_technical_acceptance',
                               'policy_source': 'OSNet sample quality budget; fixed before full evaluation'},
                'performance': {'runtime_threads': [1, 2], 'cpp_streams': [1, 2], 'rounds': 3, 'frames_per_round': 200,
                                'warmup': 20, 'cpu_policy_change': False, 'runtime_input': 'fixed signed planar RGB'},
                'performance_inputs': fixture_files,
                'conversion': {'mode': 'reuse_verified_bin', 'original_full_conversion_archive_available': False,
                               'float_baseline': 'new FP32 ONNX export from exact original checkpoint',
                               'oe_report': 'regenerate hb_perf analysis of the exact reused BIN', 'image_id': args.image_id},
                'publication': {'authorized': False, 'release_approval': 'not-run'}}
    path = args.workspace/'configs'/f'{args.build_id}.json'
    write(path, campaign)
    path.with_suffix('.sha256').write_text(f'{sha(path)}  {path.name}\n')
    for category in ('conversion', 'benchmark'):
        run = args.workspace/'runs'/category/args.build_id
        for folder in ('input', 'work', 'output', 'logs'): (run/folder).mkdir(parents=True, exist_ok=False)
        shutil.copyfile(path, run/'input/campaign.json')
        write(run/'run.json', {'build_id': args.build_id, 'stage': category, 'status': 'created', 'source_commit': commit, 'campaign_sha256': sha(path)})
    print(path)


def aggregate(args):
    rows = []
    for threads in (1, 2):
        for round_id in (1, 2, 3):
            path = args.logs/f'perf-t{threads}-r{round_id}.log'
            text = path.read_text()
            def number(pattern):
                matches = re.findall(pattern, text)
                if not matches: raise ValueError(f'Missing {pattern} in {path}')
                return float(matches[-1])
            frames = int(number(r'Frame count\s+is:\s*(\d+)'))
            if frames != 200: raise ValueError(f'{path}: expected 200 completed total frames')
            row = {'threads': threads, 'round': round_id, 'frames': frames,
                   'wall_ms': number(r'Program run time:\s*([\d.]+)'),
                   'mean_ms': number(r'Average\s+latency\s+is:\s*([\d.]+)'),
                   'min_ms': min(map(float, re.findall(r'thread min latency:\s*([\d.]+)', text))),
                   'max_ms': max(map(float, re.findall(r'thread max latency:\s*([\d.]+)', text))),
                   'reported_fps': number(r'Frame\s+rate\s+is:\s*([\d.]+)'), 'log': str(path), 'log_sha256': sha(path)}
            rows.append(row)
    measures = []
    for threads in (1, 2):
        selected = [r for r in rows if r['threads'] == threads]
        measures.append({'threads': threads, 'average_latency_ms': sum(r['mean_ms']*r['frames'] for r in selected)/600,
                         'observed_min_latency_ms': min(r['min_ms'] for r in selected), 'observed_max_latency_ms': max(r['max_ms'] for r in selected),
                         'aggregate_fps': 600000/sum(r['wall_ms'] for r in selected)})
    write(args.output, {'tool': 'hrt_model_exec perf', 'implementation': 'native_cpp_cli', 'timing_scope': 'model_runtime',
                       'thread_semantics': 'runtime_submission_concurrency', 'core_id': 1, 'warmup_frames_per_condition': 20,
                       'warmup_scope': 'separate_process_device_warmup', 'runs_per_condition': 3, 'frames_per_run': 200,
                       'input_kind': 'fixed signed planar RGB', 'input_sha256': sha(args.input),
                       'stages': {'preprocess': 'not_measured', 'runtime': 'measured', 'postprocess': 'not_measured', 'end_to_end': 'not_measured'},
                       'measurements': measures, 'raw_measurements': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('measure'); p.add_argument('--weights', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('freeze')
    for key in ('workspace', 'dataset', 'model', 'measurement', 'guide', 'fixture'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--build-id', required=True); p.add_argument('--image-id', required=True)
    p = sub.add_parser('aggregate-runtime')
    for key in ('logs', 'output', 'input'): p.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    {'measure': measure, 'freeze': freeze, 'aggregate-runtime': aggregate}[args.command](args)


if __name__ == '__main__': main()
