"""B2-R1-E evidence capture: default-entry runs with full traceability.

Runs on the board from /tmp/rdk-b2.  Records, for the EfficientNet
omitted-variant default entry on this S target: board identity, UTC
timestamps, SHA-256 of the four deployed fix files (computed ON the board
against the exact bytes that execute), SHA-256 of the artifact and input
image used, and the COMPLETE stdout/stderr + exit code of each command
with its exact argv and cwd.  Prints one JSON record.
"""

import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

TARGET = sys.argv[1]
B = pathlib.Path('/tmp/rdk-b2/bundle-d54d1cf531cd')
MAIN = B / 'samples/vision/efficientnet/runtime/python/main.py'
RECORD = {
    'purpose': 'B2-R1-E: B2 remediation re-review evidence for the S default '
               'entry (omitted --variant/--asset-id). Fresh full-output capture; '
               'the earlier re-verification runs were not saved with complete '
               'output, so this record is a new capture, not a backdated one.',
    'target_slot': TARGET,
    'capture_started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'bundle_base_sha256_of_original_tar': 'd54d1cf531cd8f8ce63aaf8685cde74c34a5c0b1443116c36cd4dee8d61a3845',
    'deployment_note': 'the four fix files below are overlaid onto that bundle; '
                       'their hashes (computed on this board) bind the executed '
                       'code; the original tar hash alone does not.',
}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ''


RECORD['identity'] = {
    'soc_name': read_text('/sys/class/boardinfo/soc_name'),
    'hostname': subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
    'python': sys.version.split()[0],
    'memtotal_kb': (read_text('/proc/meminfo').split() or ['MemTotal:', ''])[1],
}

DEPLOYED = {
    'samples/_shared/cls_binding.py': None,
    'samples/vision/efficientnet/runtime/python/model_binding.py': None,
    'samples/vision/efficientnet/runtime/python/main.py': None,
    'samples/vision/efficientnet/model/download.py': None,
}
RECORD['deployed_file_sha256'] = {
    rel: sha256(B / rel) for rel in DEPLOYED
}

artifact = B / 'models' / TARGET / 'efficientnet_lite0_224x224_nv12.hbm'
staged = B / 'samples/vision/efficientnet/model' / TARGET / 'efficientnet_lite0_224x224_nv12.hbm'
image = B / 'samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG'
labels = B / 'datasets/imagenet/imagenet_classes.names'
RECORD['inputs'] = {
    'model_artifact_sha256': sha256(artifact),
    'staged_model_sha256': sha256(staged),
    'staged_equals_bundle_artifact': sha256(artifact) == sha256(staged),
    'input_image_sha256': sha256(image),
    'labels_sha256': sha256(labels),
    'digest_check_note': 'artifact digest matches models.sha256 in the '
                         'original bundle (verified 19/19 during the original '
                         'smoke; lite0 digest also equals the recorded value '
                         'in the five-board records)',
}

RECORD['runs'] = []


def run_case(name, argv):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proc = subprocess.run(
        [sys.executable, str(MAIN)] + argv,
        capture_output=True, text=True, cwd=str(B))
    entry = {
        'check': name,
        'argv': ['python3', str(MAIN)] + argv,
        'cwd': str(B),
        'started_utc': started,
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'rc': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }
    RECORD['runs'].append(entry)
    print(json.dumps({'check': name, 'rc': entry['rc']}), flush=True)


run_case('dry-run-default-entry', ['--dry-run', '--target', TARGET])
run_case('default-entry-full-inference', ['--target', TARGET])
if TARGET == 's600':
    run_case('explicit-variant-control-lite2', [
        '--target', 's600',
        '--variant', 'lite2',
        '--asset-id', 's:efficientnet:s600/efficientnet_lite2_260x260_nv12.hbm',
        '--model-path', str(B / 'models/s600/efficientnet_lite2_260x260_nv12.hbm'),
    ])

RECORD['capture_finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print('R1E_CAPTURE_BEGIN', flush=True)
print(json.dumps(RECORD), flush=True)
print('R1E_CAPTURE_END', flush=True)
