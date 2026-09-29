"""Pinned-source inventory and minimal CPU-only reproductions for B10."""
from pathlib import Path
import ast, hashlib, importlib.util, json, subprocess, wave
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
pins={'x5':'ac115717197920355fc390bb04299b20e6436864','s':'380e1a2bf42041af54be6f34935e50197cfadff9'}
paths={'himloco':('x5','samples/robotics/himloco'),'asr':('s','samples/speech/asr'),'kws':('s','samples/speech/kws'),'paraformer':('s','samples/speech/paraformer')}
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
records={}
for sample,(platform,path) in paths.items():
 files=git('ls-tree','-r','--name-only',pins[platform],path).decode().splitlines();items=[]
 for name in files:
  original=git('show',pins[platform]+':'+name)
  local=ROOT/'platforms'/platform/name
  items.append({'source':name,'sha256':sha(original),'bytes':len(original),'snapshot_exists':local.is_file(),'snapshot_matches':local.is_file() and sha(local.read_bytes())==sha(original)})
 records[sample]={'pin':pins[platform],'files':items}
# Verify pinned held-out inputs named by the already-preserved manifest; no writes.
manifest_path=ROOT/'platforms/x5/samples/robotics/himloco/test_data/runtime-input-manifest.json'
manifest=json.loads(manifest_path.read_text());verified=[]
for item in manifest['records']:
 relative='samples/robotics/himloco/test_data/'+item['file']
 raw=git('show',pins['x5']+':'+relative)
 assert len(raw)==item['bytes']==1080 and sha(raw)==item['sha256']
 assert np.isfinite(np.frombuffer(raw,dtype='<f4')).all()
 target=manifest_path.parent/item['file']
 assert target.is_file() and target.read_bytes()==raw
 verified.append({'path':str(target.relative_to(ROOT)),'source':pins['x5']+':'+relative,'sha256':sha(raw),'bytes':len(raw)})
# Actual source CIF with all-zero alphas: isolate the pure standalone module.
p=ROOT/'platforms/s/samples/speech/paraformer/conversion/cif_numpy.py'
spec=importlib.util.spec_from_file_location('source_cif',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
 module.cif_numpy(np.zeros((1,401),np.float32),np.zeros((1,401,512),np.float32),400)
 cif={'raised':False}
except Exception as error:cif={'raised':True,'type':type(error).__name__,'message':str(error)}
assert cif.get('type')=='IndexError',cif
# Extract unchanged ASR method only; do not import SDK or any audio package.
p=ROOT/'platforms/s/samples/speech/asr/runtime/python/asr.py';tree=ast.parse(p.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ASR')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='post_process')
method.returns=None
for arg in method.args.args:arg.annotation=None
scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(p),'exec'),scope)
class Dummy:model_name='asr';output_name='out'
logits=np.eye(3,dtype=np.float32)[np.array([[1,1,0,1,2,2]])]
text=scope['post_process'](Dummy(),{'asr':{'out':logits}},{0:'<pad>',1:'a',2:'b'})
assert text=='aaabb',text
# Capture real sample headers without importing runtime or executing inference.
audio={}
for sample,(platform,path) in paths.items():
 for wav in (ROOT/'platforms'/platform/path).rglob('*.wav'):
  with wave.open(str(wav),'rb') as stream:
   audio[str(wav.relative_to(ROOT))]={'rate':stream.getframerate(),'channels':stream.getnchannels(),'frames':stream.getnframes(),'sample_bytes':stream.getsampwidth(),'sha256':sha(wav.read_bytes())}
result={'utc':datetime.now(timezone.utc).isoformat(),'inventory':records,'verified_himloco':verified,'paraformer_zero_fire':cif,'asr_source_decode':{'token_ids':[1,1,0,1,2,2],'result':text,'standard_ctc_expected':'aab'},'audio':audio,'kws_fixed_duration_at_16khz_seconds':60000/16000,'board_or_sdk_executed':False}
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'files':{k:len(v['files']) for k,v in records.items()},'missing_before':{k:[i['source'] for i in v['files'] if not i['snapshot_exists']] for k,v in records.items()},'verified_himloco':len(verified),'cif':cif,'asr':text,'audio':audio},indent=2))
