"""Exercise actual CLI + FunASR without SDK models; retain fresh run artifacts."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
run_root=Path(tempfile.mkdtemp(prefix='run-',dir=HERE))
script=ROOT/'samples/speech/paraformer/runtime/python/main.py'
sample=ROOT/'samples/speech/paraformer'
manifest=sample/'test_data/manifest.json'
original=manifest.read_bytes()
records=[]
def run(label,args,expected):
    command=[sys.executable,str(script),*args]
    start=datetime.now(timezone.utc).isoformat()
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    end=datetime.now(timezone.utc).isoformat()
    record={'label':label,'argv':command,'cwd':str(ROOT),'started_utc':start,'ended_utc':end,'rc':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    (run_root/f'{label}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    assert process.returncode==expected,(label,process.stderr)
    records.append({'label':label,'record':str((run_root/f'{label}.json').relative_to(HERE)),'rc':process.returncode})
    return process
run('help',['--help'],0)
run('list',['--list-models'],0)
run('dry',['--target','s100','--dry-run'],0)
run('s100p-rejected',['--target','s100p','--dry-run'],2)
out=run_root/'features'
run('prepare',['--preprocess-only','--output-dir',str(out)],0)
assert manifest.read_bytes()==original
prepared=json.loads((out/'prepared-manifest.json').read_text())
report=json.loads((out/'result.json').read_text())
assert report['status']=='completed' and not report['inference_executed'] and not report['inference_attempted']
assert [x['feat_length'] for x in prepared]==[71,78]
for item in prepared:
    feature=np.load(out/item['feature_file'],allow_pickle=False)
    assert feature.shape==(1,400,560) and feature.dtype==np.float32
    assert not feature[0,item['feat_length']:].any()
    assert hashlib.sha256((out/item['feature_file']).read_bytes()).hexdigest()==item['feature_sha256']
run('output-reuse-rejected',['--preprocess-only','--output-dir',str(out)],2)
run('single',['--preprocess-only','--audio-file',str(sample/'test_data/audio/BAC009S0724W0168.wav'),'--output-dir',str(run_root/'single')],0)
run('limit',['--preprocess-only','--max-utts','1','--output-dir',str(run_root/'limit')],0)
assert len(json.loads((run_root/'limit/result.json').read_text())['utterances'])==1
# Bad sample rate enters the genuine frontend path and must leave failed.json.
bad_audio=run_root/'bad-rate.wav'
sf.write(bad_audio,np.zeros(800,np.float32),8000,subtype='PCM_16')
run('bad-rate',['--preprocess-only','--audio-file',str(bad_audio),'--output-dir',str(run_root/'bad')],2)
assert json.loads((run_root/'bad/failed.json').read_text())['status']=='failed'
assert not (run_root/'bad/result.json').exists()
# Full inference must fail the real local identity gate; no SDK/model fabrication.
run('host-inference-rejected',['--target','s100','--output-dir',str(run_root/'inference')],2)
assert not (run_root/'inference').exists()
assert manifest.read_bytes()==original
summary={'scope':'real CPU frontend and CLI; no HBM or board inference',
 'run_root':str(run_root.relative_to(HERE)), 'records':records,
 'manifest_unchanged':True,'manifest_sha256':hashlib.sha256(original).hexdigest(),
 'versions':{p:importlib.metadata.version(p) for p in ('torch','torchaudio','funasr','numpy','soundfile')},
 'python':sys.version}
(HERE/'real-cli-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
