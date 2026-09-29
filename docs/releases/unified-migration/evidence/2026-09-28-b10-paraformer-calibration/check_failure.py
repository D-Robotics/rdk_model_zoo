"""Actual CLI rejects a selected 8 kHz WAV and retains failed preparation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT.parent/'.coordination/paraformer-calibration-wrong-rate'
EXPORT = ROOT.parent/'.coordination/paraformer-export-v4'
assert not OUTPUT.exists(), 'Use a fresh failure output path for a rerun'
with tempfile.TemporaryDirectory() as directory:
    sf.write(str(Path(directory)/'bad.wav'), np.zeros(8000, np.float32), 8000)
    command = [sys.executable, str(ROOT/'samples/speech/paraformer/conversion/prepare.py'),
               '--export-dir', str(EXPORT), '--wav-dir', directory, '--output-dir', str(OUTPUT)]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    record = json.loads((OUTPUT/'preparation.json').read_text())
    assert run.returncode == 2 and record['status'] == 'preparation_failed'
    assert not record['records'] and '16000' in record['error']
    assert not (OUTPUT/'configs').exists()
    result = {'argv': command, 'returncode': run.returncode, 'stdout': run.stdout,
              'stderr': run.stderr, 'report': record, 'configs_created': False}
    Path(__file__).with_name('wrong-rate.json').write_text(json.dumps(result, indent=2)+'\n')
    print('8 kHz rejected; failure retained; no configs and no skipped-success record')
