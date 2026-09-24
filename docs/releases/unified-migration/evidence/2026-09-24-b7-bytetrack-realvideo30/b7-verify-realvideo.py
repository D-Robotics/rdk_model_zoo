import pathlib,json,tarfile,hashlib,io
import numpy as np
out=pathlib.Path('docs/releases/unified-migration/evidence/2026-09-24-b7-bytetrack-realvideo30');index=json.loads((out/'archive-verification.json').read_text());payloads={};size=0
for archive in index['archives']:
 p=out/archive['archive'];assert hashlib.sha256(p.read_bytes()).hexdigest()==archive['archive_sha256'];assert p.stat().st_size==archive['bytes'];size+=p.stat().st_size
 with tarfile.open(p) as t:
  assert len(t.getmembers())==len(archive['entries'])
  for e in archive['entries']:
   b=t.extractfile(e['member']).read();assert hashlib.sha256(b).hexdigest()==e['sha256'];a=np.load(io.BytesIO(b),allow_pickle=False);payloads[e['sha256']]={'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all())}
assert len(payloads)==270
for r in index['captured_array_references']:
 p=payloads[r['sha256']];assert p['shape']==r['shape'] and p['dtype']==r['dtype'] and p['finite']
boards=[]
for target in ('s100','s600'):
 a,b=[json.loads((out/f'{target}-{side}-capture.json').read_text()) for side in ('legacy','unified')];summary=json.loads((out/f'{target}-comparison.json').read_text());assert summary['passed'] and summary['return_code']==0 and all(x is True for x in summary['checks'].values());assert all(r['return_code']==0 for r in summary['runs'])
 assert a['video_sha256']==b['video_sha256']=='4bbe5bf11fe8967b28a900fd2add4949aba89b62076eaa03d0c55cdf7dd41397';assert a['model_sha256']==b['model_sha256'];assert len(a['frames'])==len(b['frames'])==30
 tracks=0
 for fa,fb in zip(a['frames'],b['frames']):
  assert fa['frame']==fb['frame'];assert fa['image']==fb['image'];assert fa['inputs']==fb['inputs'] and fa['outputs']==fb['outputs']; assert len(fa['tracks'])==len(fb['tracks'])
  for ta,tb in zip(fa['tracks'],fb['tracks']):
   assert ta['track_id']==tb['track_id'];np.testing.assert_allclose(ta['tlbr'],tb['tlbr'],atol=1e-4,rtol=0);assert abs(ta['score']-tb['score'])<=1e-5;tracks+=1
 boards.append({'target':target,'frames_per_side':30,'track_records_compared':tracks,'checks':len(summary['checks']),'model_sha256':a['model_sha256'],'arrays_between_source_unified':'same bytes by complete sha256 for every image/input/output','passed':True})
(out/'independent-verification.json').write_text(json.dumps({'array_references':len(index['captured_array_references']),'unique_payloads':len(payloads),'archive_count':len(index['archives']),'archive_bytes':size,'boards':boards,'scope':'First 30 frames of fixed-source track_test.mp4; neither full video nor labeled MOT accuracy/latency.'},indent=2)+'\n')
print(json.dumps({'boards':boards,'bytes':size,'unique':len(payloads)}))
