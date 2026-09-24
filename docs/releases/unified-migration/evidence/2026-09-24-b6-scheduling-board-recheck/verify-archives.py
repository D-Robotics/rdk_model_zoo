import pathlib,json,subprocess,tarfile,hashlib,io,concurrent.futures
import numpy as np
root=pathlib.Path.cwd();out=root/'docs/releases/unified-migration/evidence/2026-09-24-b6-scheduling-board-recheck';out.mkdir(exist_ok=True)
sha='a72f92b41d4265e36b9fc7c5a85d0b15800cee8d'
def collect(item):
 host,target,sample,priority=item;key=target+'-'+sample+('-p7' if priority==7 else '');remote=f'/tmp/rdk-b6-{target}-{sample}-a72f92b'+('-p7' if priority==7 else '');archive=out/(key+'.tar.gz')
 with archive.open('wb') as f:p=subprocess.run(['ssh','-o','BatchMode=yes',host,'tar -czf - -C '+remote+' .'],stdout=f,stderr=subprocess.PIPE)
 assert p.returncode==0,p.stderr
 with tarfile.open(archive) as t:
  files={m.name.removeprefix('./'):m for m in t.getmembers() if m.isfile()};assert all(n=='comparison.json' or n.endswith('.npy') for n in files)
  read=lambda n:t.extractfile(files[n]).read();d=json.loads(read('comparison.json'));(out/(key+'-comparison.json')).write_text(json.dumps(d,indent=2)+'\n');count=0
  for name,record in d['arrays'].items():
   data=read(name);assert hashlib.sha256(data).hexdigest()==record['sha256'];a=np.load(io.BytesIO(data),allow_pickle=False);assert list(a.shape)==record['shape'] and str(a.dtype)==record['dtype'] and np.isfinite(a).all();count+=1
  assert count==sum(n.endswith('.npy') for n in files)
  for name,digest in d['code_sha256'].items():
   data=subprocess.run(['git','show',sha+':'+name],capture_output=True,check=True).stdout;assert hashlib.sha256(data).hexdigest()==digest,(key,name)
  assert d['passed'] and d['return_code']==0 and all(v is True for v in d['checks'].values()) and count==(14 if sample=='efficient_sam' else 16)
  schedule=d['scheduling'];assert schedule['requested']['priority']==priority
  for side in ('legacy','unified'):
   for stage in ('encoder','decoder'):
    calls=[c for c in schedule['native_calls'][side] if c['stage']==stage]
    applied=[c for c in calls if c.get('applied')]
    assert len(applied)==(2 if target=='s100' and side=='legacy' else 1)
    assert applied[-1]['args']==schedule['explicit_control'][side+'.'+stage]['native_args']
    assert list(applied[-1]['args']['priority'].values())==[priority]
    if target=='x5' and side=='legacy':assert len(calls)==2 and 'TypeError' in calls[0]['error'] and not calls[0].get('applied')

 result={'case':key,'passed':d['passed'],'return_code':d['return_code'],'arrays_verified':count,'code_hashes_verified':len(d['code_sha256']),'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()};assert result['archive_bytes']<50_000_000;print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(collect,[(h,t,s,p) for h,t in [('x5-8g','x5'),('s100','s100')] for s in ('efficient_sam','mobile_sam') for p in ((0,7) if t=='x5' else (0,))]))
for target in ('x5','s100'):(out/(target+'-execution.json')).write_bytes((root.parent/'.coordination'/f'b6-{target}-scheduling-recheck.json').read_bytes())
(out/'archive-verification.json').write_text(json.dumps({'commit':sha,'cases':results},indent=2)+'\n')
