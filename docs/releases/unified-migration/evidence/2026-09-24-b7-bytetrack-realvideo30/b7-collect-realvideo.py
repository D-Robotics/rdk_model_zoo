import pathlib,json,subprocess,hashlib,tarfile,io,concurrent.futures,shlex,math
import numpy as np
out=pathlib.Path('docs/releases/unified-migration/evidence/2026-09-24-b7-bytetrack-realvideo30')
unique={};refs=[]
for target in ('s100','s600'):
 for side in ('legacy','unified'):
  d=json.loads((out/f'{target}-{side}-capture.json').read_text()); assert d['return_code']==0 and len(d['frames'])==30
  for frame in d['frames']:
   entries=[frame['image'],*frame['inputs'].values(),*frame['outputs'].values()]
   assert len(entries)==6
   for e in entries:
    ref={'target':target,'side':side,**e};refs.append(ref)
    if e['sha256'] not in unique:unique[e['sha256']]=ref
    else:
     old=unique[e['sha256']];assert old['shape']==e['shape'] and old['dtype']==e['dtype']
jobs=[]
for target in ('s100','s600'):
 for side in ('legacy','unified'):
  batch=[];size=0
  for r in unique.values():
   if r['target']!=target or r['side']!=side:continue
   n=math.prod(r['shape'])*np.dtype(r['dtype']).itemsize+1024
   if batch and size+n>40_000_000:jobs.append((target,side,batch));batch=[];size=0
   batch.append(r);size+=n
  if batch:jobs.append((target,side,batch))
def collect(item):
 index,(target,side,batch)=item;host='s100' if target=='s100' else 's600-64g';remote=f'/tmp/rdk-b7-{target}-realvideo30-4d45f9a/{side}'
 name=f'payloads-{index:02d}-{target}-{side}.tar.gz';path=out/name
 command='tar -czf - -C '+shlex.quote(remote)+' -- '+' '.join(shlex.quote(r['file']) for r in batch)
 with path.open('wb') as f:
  p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,command],stdout=f,stderr=subprocess.PIPE)
 assert p.returncode==0,p.stderr
 entries=[]
 with tarfile.open(path) as t:
  assert sorted(m.name for m in t.getmembers())==sorted(r['file'] for r in batch)
  for r in batch:
   raw=t.extractfile(r['file']).read();assert hashlib.sha256(raw).hexdigest()==r['sha256'];a=np.load(io.BytesIO(raw),allow_pickle=False);assert list(a.shape)==r['shape'] and str(a.dtype)==r['dtype'];assert np.isfinite(a).all();entries.append({'sha256':r['sha256'],'member':r['file']})
 result={'archive':name,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'entries':entries};assert result['bytes']<50_000_000
 (out/(name+'.verification.json')).write_text(json.dumps(result,indent=2)+'\n'); print(name,len(entries),result['bytes'],flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:archives=list(pool.map(collect,enumerate(jobs)))
assert len(refs)==720;assert sum(len(a['entries']) for a in archives)==len(unique)
(out/'archive-verification.json').write_text(json.dumps({'note':'Every captured array retained by SHA-256. Identical bytes shared across records; each reference maps to one verified archive member. This is storage deduplication, not a replacement for either independent run.','captured_array_references':refs,'reference_count':len(refs),'unique_payload_count':len(unique),'archives':archives},indent=2)+'\n')
print('COMPLETE',len(refs),len(unique),flush=True)
