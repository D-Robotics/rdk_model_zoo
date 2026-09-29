"""Compare real calibration arrays with source outputs and persisted frontend inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
from samples._shared.assets import sha256_file
from samples.speech.paraformer.conversion.workspace import verify_prepared
from samples.speech.paraformer.conversion.calibration import CALIBRATION

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--legacy-dir', type=Path, required=True)
args = parser.parse_args()
report, digest, configs = verify_prepared(args.workspace)
assert len(report['records']) == 2
result = {'preparation_sha256': digest, 'cases': [], 'compiler': 'not-run', 'board': 'not-run',
          'source_script_sha256': sha256_file(ROOT/'platforms/s/samples/speech/paraformer/conversion/10_gen_real_calib.py')}
features = ROOT/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/run-5_sjkdzc/features/feats'
for index, record in enumerate(report['records']):
    key = Path(record['audio_path']).stem
    checks = {}
    for name in CALIBRATION:
        actual = args.workspace/'calibration'/name/record['filename']
        reference = features/f'{key}.npy' if name == 'speech' else args.legacy_dir/name/f'{index:03d}.npy'
        a, b = np.load(actual, allow_pickle=False), np.load(reference, allow_pickle=False)
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
        checks[name] = {'shape': list(a.shape), 'dtype': str(a.dtype),
            'actual_sha256': sha256_file(actual), 'reference_sha256': sha256_file(reference),
            'array_bytes_equal': a.tobytes() == b.tobytes()}
    result['cases'].append({'audio': key, 'checks': checks})
result['oe_tools'] = {tool: shutil.which(tool) for tool in ('hb_compile', 'docker')}
if result['oe_tools']['hb_compile'] is None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)/'unavailable'
        command = [sys.executable, str(ROOT/'samples/speech/paraformer/conversion/compile.py'),
                   '--workspace', str(args.workspace.resolve()), '--output-dir', str(output)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert run.returncode == 2 and not output.exists() and 'unavailable' in run.stderr
        result['missing_oe_check'] = {'argv': command, 'rc': run.returncode,
            'stdout': run.stdout, 'stderr': run.stderr, 'output_created': output.exists()}
Path(__file__).with_name('summary.json').write_text(json.dumps(result, indent=2)+'\n')
Path(__file__).with_name('preparation.json').write_text(json.dumps(report, indent=2)+'\n')
for stage in configs:
    shutil.copyfile(args.workspace/f'configs/{stage}.yaml', Path(__file__).with_name(f'{stage}.yaml'))
print(json.dumps(result, indent=2))
