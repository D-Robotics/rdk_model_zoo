"""B2 board smoke harness: legacy-vs-unified comparison + CLI + negatives.

Runs entirely on the board from /tmp/rdk-b2.  Compares, per case, the source
wrapper (platforms/ legacy tree, same-board baseline) against the unified
sample entry (resolve_selection -> RuntimeModelRunner -> ClassificationTask)
on the same artifact bytes and input image; ids must match exactly and scores
must agree within 1e-5.  Also runs one CLI end-to-end per sample (run.sh) and,
on s100p, the rejection negatives.  Prints one JSON record at the end.
"""

import datetime
import hashlib
import importlib
import importlib.util
import json
import pathlib
import py_compile
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np

BUNDLE_SHA = 'd54d1cf531cd8f8ce63aaf8685cde74c34a5c0b1443116c36cd4dee8d61a3845'
TARGET = sys.argv[1]
ROOT = pathlib.Path('/tmp/rdk-b2')

record = {
    'target_slot': TARGET,
    'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'entries': [],
    'ok': False,
}

# --- extract bundle (verified) ---
raw = (ROOT / 'b2-code.tar.gz').read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == BUNDLE_SHA, 'bundle sha mismatch: ' + digest
base = ROOT / ('bundle-' + digest[:12])
if not base.exists():
    base.mkdir(parents=True)
    with tarfile.open(ROOT / 'b2-code.tar.gz') as archive:
        for member in archive.getmembers():
            dest = (base / member.name).resolve()
            if base.resolve() != dest and base.resolve() not in dest.parents:
                raise ValueError('unsafe member ' + member.name)
        archive.extractall(base)
sys.path.insert(0, str(base))
record['bundle_sha256'] = digest
record['bundle_dir'] = str(base)


def read_text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ''


record['identity'] = {
    'soc_name': read_text('/sys/class/boardinfo/soc_name'),
    'socinfo': read_text('/sys/class/socinfo/soc_name') or read_text('/sys/class/socinfo/soc'),
    'board_type': read_text('/sys/class/boardinfo/board_type'),
    'device_tree_model': read_text('/sys/device-tree/model').rstrip('\0'),
    'hostname': subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
    'python': sys.version.split()[0],
    'memtotal_kb': (read_text('/proc/meminfo').split() or ['MemTotal:', ''])[1],
}

from samples._shared.platforms import detect_target  # noqa: E402

detected = detect_target()
record['detected'] = detected

# --- compile check (unified + legacy trees in the bundle) ---
compiled = 0
for pattern in ('samples/_shared', 'samples/vision/efficientnet', 'samples/vision/efficientformer',
                'samples/vision/efficientformerv2', 'samples/vision/efficientvit',
                'platforms/x5/samples/vision', 'platforms/s/samples/vision'):
    for path in (base / pattern).rglob('*.py'):
        py_compile.compile(str(path), doraise=True)
        compiled += 1
record['compile_pass'] = True
record['compile_count'] = compiled

# --- artifact integrity vs models.sha256 ---
models = base / 'models'
checked = 0
for line in (base / 'models.sha256').read_text().splitlines():
    expected, rel = line.split()
    data = (base / rel).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, 'artifact digest mismatch: ' + rel
    checked += 1
record['artifact_digest_checks'] = checked

labels = base / 'datasets/imagenet/imagenet_classes.names'

X5_CASES = [
    # sample, variant, model file, wrapper class, test image, cli default variant
    ('efficientnet', 'b2', 'EfficientNet_B2_224x224_nv12.bin', 'EfficientNet', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'b3', 'EfficientNet_B3_224x224_nv12.bin', 'EfficientNet', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'b4', 'EfficientNet_B4_224x224_nv12.bin', 'EfficientNet', 'Scottish_deerhound.JPEG'),
    ('efficientformer', 'l1', 'EfficientFormer_l1_224x224_nv12.bin', 'EfficientFormer', 'bittern.JPEG'),
    ('efficientformer', 'l3', 'EfficientFormer_l3_224x224_nv12.bin', 'EfficientFormer', 'bittern.JPEG'),
    ('efficientformerv2', 's0', 'EfficientFormerv2_s0_224x224_nv12.bin', 'EfficientFormerV2', 'goldfish.JPEG'),
    ('efficientformerv2', 's1', 'EfficientFormerv2_s1_224x224_nv12.bin', 'EfficientFormerV2', 'goldfish.JPEG'),
    ('efficientformerv2', 's2', 'EfficientFormerv2_s2_224x224_nv12.bin', 'EfficientFormerV2', 'goldfish.JPEG'),
    ('efficientvit', 'm5', 'EfficientViT_m5_224x224_nv12.bin', 'EfficientViT', 'hook.JPEG'),
]
S_CASES = [
    ('efficientnet', 'lite0', 'efficientnet_lite0_224x224_nv12.hbm', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'lite1', 'efficientnet_lite1_240x240_nv12.hbm', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'lite2', 'efficientnet_lite2_260x260_nv12.hbm', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'lite3', 'efficientnet_lite3_300x300_nv12.hbm', 'Scottish_deerhound.JPEG'),
    ('efficientnet', 'lite4', 'efficientnet_lite4_380x380_nv12.hbm', 'Scottish_deerhound.JPEG'),
]


def load_legacy(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unified_predict(sample, target, reference, model_path, image, topk=5):
    binding_mod = importlib.import_module(
        'samples.vision.%s.runtime.python.model_binding' % sample)
    runner_mod = importlib.import_module(
        'samples.vision.%s.runtime.python.model_runner' % sample)
    task_mod = importlib.import_module(
        'samples.vision.%s.runtime.python.classification' % sample)
    selection = binding_mod.resolve_selection(target, asset_id=reference, model_path=model_path)
    runner = runner_mod.RuntimeModelRunner(selection)
    binding = runner.load()
    runner.set_scheduling_params(priority=0, bpu_cores=[0])
    task = task_mod.ClassificationTask(runner, binding, top_k=topk)
    return task.predict(image), selection


def cli_check(entry_name, sample, argv):
    run_sh = base / 'samples' / 'vision' / sample / 'runtime' / 'python' / 'run.sh'
    started = time.perf_counter()
    proc = subprocess.run(['bash', str(run_sh)] + argv, capture_output=True, text=True, cwd=str(base))
    entry = {
        'check': entry_name,
        'argv': argv,
        'rc': proc.returncode,
        'elapsed_s': round(time.perf_counter() - started, 2),
        'stdout_tail': proc.stdout.strip()[-400:],
        'stderr_tail': proc.stderr.strip()[-400:],
    }
    if proc.returncode == 0 and 'Top-5 results' in proc.stdout:
        entry['status'] = 'pass'
    else:
        entry['status'] = 'fail'
    record['entries'].append(entry)
    print(json.dumps({k: entry[k] for k in ('check', 'rc', 'status')}), flush=True)
    return entry


def compare_entry(entry, legacy_ids, legacy_scores, result):
    entry.update({
        'legacy_ids': [int(v) for v in legacy_ids],
        'unified_ids': [int(v) for v in result.class_ids],
        'legacy_scores': [float(v) for v in legacy_scores],
        'unified_scores': [float(v) for v in result.scores],
    })
    entry['ids_equal'] = entry['legacy_ids'] == entry['unified_ids']
    entry['max_score_abs_diff'] = float(np.max(np.abs(
        np.asarray(legacy_scores, dtype=np.float64) - np.asarray(result.scores, dtype=np.float64))))
    if entry['ids_equal'] and entry['max_score_abs_diff'] < 1e-5:
        entry['status'] = 'pass'
    else:
        entry['status'] = 'fail'


def analyse_tie(entry, legacy_stage, unified_stage):
    """Explain an id mismatch as a model-level near-tie, or leave it failed.

    Re-runs both implementations with topk=8 and requires (a) every class id
    of either top-5 to appear in both top-8 lists with per-id score agreement
    below 1e-5, and (b) the competing boundary classes to be separated by less
    than 1e-6 in *both* implementations' scores — i.e. the model itself cannot
    distinguish them and the rank-5 pick is arithmetic noise between the two
    softmax implementations, not a behavioral difference.
    """

    l8_ids, l8_scores = legacy_stage(topk=8)
    r8, _ = unified_stage(topk=8)
    u8_ids = [int(v) for v in r8.class_ids]
    u8_scores = [float(v) for v in r8.scores]
    lmap = {int(i): float(s) for i, s in zip(l8_ids, l8_scores)}
    umap = {int(i): float(s) for i, s in zip(u8_ids, u8_scores)}
    union = sorted(set(lmap) | set(umap))
    per_id = {i: [lmap.get(i), umap.get(i)] for i in union}
    per_id_ok = all(
        i in lmap and i in umap and abs(lmap[i] - umap[i]) < 1e-5 for i in union)
    legacy_only = [i for i in entry['legacy_ids'] if i not in entry['unified_ids']]
    unified_only = [i for i in entry['unified_ids'] if i not in entry['legacy_ids']]
    gaps = []
    for a in legacy_only:
        for b in unified_only:
            for name, mapping in (('legacy', lmap), ('unified', umap)):
                if a in mapping and b in mapping:
                    gaps.append([name, a, b, abs(mapping[a] - mapping[b])])
    tie_ok = bool(gaps) and all(g[3] < 1e-6 for g in gaps)
    entry['tie_analysis'] = {
        'per_id_top8': per_id,
        'per_id_ok': per_id_ok,
        'legacy_only': legacy_only,
        'unified_only': unified_only,
        'boundary_gaps': [[g[0], g[1], g[2], g[3]] for g in gaps],
        'tie_ok': tie_ok,
    }
    if per_id_ok and tie_ok:
        entry['status'] = 'pass'
        entry['tie_resolved'] = True


def run_case(entry, legacy_stage, unified_stage):
    started = time.perf_counter()
    try:
        legacy_ids, legacy_scores = legacy_stage(topk=5)
        entry['legacy_elapsed_s'] = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        result, selection = unified_stage(topk=5)
        entry['unified_elapsed_s'] = round(time.perf_counter() - started, 2)
        entry['resolved_asset_id'] = selection.asset_id
        entry['resolved_geometry'] = '%dx%d' % (selection.contract.input_width, selection.contract.input_height)
        compare_entry(entry, legacy_ids, legacy_scores, result)
        if entry['status'] == 'fail':
            analyse_tie(entry, legacy_stage, unified_stage)
    except Exception as exc:  # noqa: BLE001 - record and continue other cases
        entry['status'] = 'fail'
        entry['error'] = repr(exc)[:400]
    record['entries'].append(entry)
    print(json.dumps({'case': entry.get('sample', entry.get('check')),
                      'variant': entry.get('variant'), 'status': entry['status']}), flush=True)


if TARGET in ('x5-8g', 'x5-4g'):
    assert detected == 'x5', 'detected %r, expected x5' % detected
    for sample, variant, filename, wrapper, image_name in X5_CASES:
        model_path = models / filename
        image_path = base / 'samples' / 'vision' / sample / 'test_data' / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        entry = {
            'sample': sample, 'variant': variant, 'file': filename,
            'artifact_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
            'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()[:16],
        }

        def legacy_stage(sample=sample, variant=variant, model_path=model_path, wrapper=wrapper,
                         image=image, topk=5):
            legacy = load_legacy('legacy_%s_%s' % (sample, variant),
                                 base / 'platforms' / 'x5' / 'samples' / 'vision' / sample
                                 / 'runtime' / 'python' / (sample + '.py'))
            cfg = getattr(legacy, wrapper + 'Config')(
                model_path=str(model_path), label_file=str(labels), resize_type=1, topk=topk)
            legacy_model = getattr(legacy, wrapper)(cfg)
            legacy_model.set_scheduling_params(priority=0, bpu_cores=[0])
            idx, prob, _ = legacy_model.predict(image)
            return idx, prob

        def unified_stage(sample=sample, model_path=model_path, image=image, filename=filename,
                          topk=5):
            return unified_predict(sample, 'x5', 'x5:%s:%s' % (sample, filename), model_path,
                                   image, topk=topk)

        run_case(entry, legacy_stage, unified_stage)

    cli_default = {
        'efficientnet': ('b2', 'EfficientNet_B2_224x224_nv12.bin'),
        'efficientformer': ('l3', 'EfficientFormer_l3_224x224_nv12.bin'),
        'efficientformerv2': ('s0', 'EfficientFormerv2_s0_224x224_nv12.bin'),
        'efficientvit': ('m5', 'EfficientViT_m5_224x224_nv12.bin'),
    }
    for sample, (variant, filename) in cli_default.items():
        cli_check('cli-%s-%s' % (sample, variant), sample, [
            '--target', 'x5',
            '--asset-id', 'x5:%s:%s' % (sample, filename),
            '--model-path', str(models / filename),
        ])
elif TARGET in ('s100', 's600'):
    assert detected == TARGET, 'detected %r, expected %s' % (detected, TARGET)
    for sample, variant, filename, image_name in S_CASES:
        model_path = models / TARGET / filename
        image_path = base / 'samples' / 'vision' / sample / 'test_data' / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        entry = {
            'sample': sample, 'variant': variant, 'file': '%s/%s' % (TARGET, filename),
            'artifact_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
            'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()[:16],
        }

        def legacy_stage(variant=variant, model_path=model_path, image=image, topk=5):
            legacy = load_legacy('legacy_s_efficientnet_' + variant,
                                 base / 'platforms' / 's' / 'samples' / 'vision'
                                 / 'efficientnet' / 'runtime' / 'python' / 'efficientnet.py')
            cfg = legacy.EfficientNetConfig(model_path=str(model_path), resize_type=1)
            legacy_model = legacy.EfficientNet(cfg)
            legacy_model.set_scheduling_params(priority=0, bpu_cores=[0])
            pairs = legacy_model.predict(image, topk=topk)
            return [p[0] for p in pairs], [p[1] for p in pairs]

        def unified_stage(sample=sample, target=TARGET, model_path=model_path, image=image,
                          filename=filename, topk=5):
            return unified_predict(sample, target,
                                   's:%s:%s/%s' % (sample, target, filename), model_path, image,
                                   topk=topk)

        run_case(entry, legacy_stage, unified_stage)

    cli_check('cli-efficientnet-lite0', 'efficientnet', [
        '--target', TARGET,
        '--asset-id', 's:efficientnet:%s/efficientnet_lite0_224x224_nv12.hbm' % TARGET,
        '--model-path', str(models / TARGET / 'efficientnet_lite0_224x224_nv12.hbm'),
    ])
elif TARGET == 's100p':
    assert detected == 's100p', 'detected %r, expected s100p' % detected
    binding_mod = importlib.import_module(
        'samples.vision.efficientnet.runtime.python.model_binding')

    entry = {'check': 'resolve_selection s100p raises'}
    try:
        binding_mod.resolve_selection('s100p')
        entry['status'] = 'fail'
        entry['error'] = 'resolve_selection(s100p) unexpectedly returned'
    except Exception as exc:  # noqa: BLE001 - the rejection is the expected path
        entry['error'] = str(exc)[:300]
        entry['status'] = 'pass'
    record['entries'].append(entry)
    print(json.dumps({'check': entry['check'], 'status': entry['status']}), flush=True)

    cli_check('cli-dry-run-s100p', 'efficientnet', ['--dry-run', '--target', 's100p'])

    entry = {'check': 'cli-target-mismatch-s100-on-s100p'}
    run_sh = base / 'samples' / 'vision' / 'efficientnet' / 'runtime' / 'python' / 'run.sh'
    proc = subprocess.run(
        ['bash', str(run_sh), '--target', 's100',
         '--asset-id', 's:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm',
         '--model-path', str(models / 's100' / 'efficientnet_lite0_224x224_nv12.hbm')],
        capture_output=True, text=True, cwd=str(base))
    entry.update({
        'rc': proc.returncode,
        'stderr_tail': proc.stderr.strip()[-300:],
        'stdout_tail': proc.stdout.strip()[-300:],
    })
    if proc.returncode == 2 and 'Target mismatch' in proc.stderr:
        entry['status'] = 'pass'
    else:
        entry['status'] = 'fail'
    record['entries'].append(entry)
    print(json.dumps({'check': entry['check'], 'rc': entry['rc'], 'status': entry['status']}), flush=True)
else:
    raise ValueError('unknown target slot ' + TARGET)

record['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
record['ok'] = bool(record['entries']) and all(e.get('status') == 'pass' for e in record['entries'])
print('HARNESS_OK' if record['ok'] else 'HARNESS_INCOMPLETE', flush=True)
print('RECORD_JSON_BEGIN', flush=True)
print(json.dumps(record), flush=True)
print('RECORD_JSON_END', flush=True)
sys.exit(0 if record['ok'] else 1)
