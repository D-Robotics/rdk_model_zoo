"""Exercise the real native CLI application with an explicitly marked transport double."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,copy
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
EXE=ROOT.parent/'.coordination/paraformer-native-io/paraformer_cli_fixture'
VOCAB=ROOT/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json'
MANIFEST=ROOT/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/run-5_sjkdzc/features/prepared-manifest.json'
records=[]
def run(label,argv,expected):
 r=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
 record={'case':label,'argv':argv,'cwd':str(ROOT),'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'expected':expected};records.append(record)
 assert r.returncode==expected,(label,r.stdout,r.stderr)
 return record
with tempfile.TemporaryDirectory(prefix='paraformer-cli-') as tmp:
 tmp=Path(tmp);models={}
 for stage in ('encoder','predictor','decoder'):
  path=tmp/(stage+'.hbm');path.write_text('fixture');models[stage]=path
 def command(output,manifest=MANIFEST):
  argv=[str(EXE),'--target','s100','--manifest',str(manifest),'--vocab-file',str(VOCAB),'--output-dir',str(output)]
  for stage,path in models.items():
   geom='400x560' if stage=='encoder' else '400x512'
   argv += [f'--{stage}-model-path',str(path),f'--{stage}-asset-id',f's:paraformer:s100/paraformer_large_{stage}_{geom}_s100.hbm',f'--{stage}-sha256',hashlib.sha256(path.read_bytes()).hexdigest()]
  return argv
 r=run('help',[str(EXE),'--help'],0);assert 'Backend: host-fixture' in r['stdout']
 run('missing-arguments',[str(EXE)],2)
 out=tmp/'complete';r=run('two-real-prepared-inputs',command(out),0);report=json.loads((out/'result.json').read_text());r['report']=report
 assert report['execution_backend']=='host-fixture' and report['status']=='completed'
 assert [x['text'] for x in report['records']]==['andand','andand']
 assert [x['valid_frames'] for x in report['records']]==[71,78]
 assert all(x['token_ids']==[3,3] and x['decoder_executed'] is True for x in report['records'])
 assert report['inference_executed'] is True
 run('existing-output-rejected',command(out),2)
 for label,extra in [('negative-limit',['--max-utts','-1']),('duplicate-target',['--target','s100']),('unknown-option',['--unknown','x'])]:run(label,command(tmp/label)+extra,2)
 r=run('prefix-one',command(tmp/'limit')+['--max-utts','1'],0);r['report']=json.loads((tmp/'limit/result.json').read_text());assert len(r['report']['records'])==1
 args=command(tmp/'wrong-sha');args[args.index('--decoder-sha256')+1]='0'*64
 run('decoder-mismatch-before-output',args,2);assert not (tmp/'wrong-sha').exists()
 models['predictor'].write_text('zero');out=tmp/'zero';r=run('zero-CIF-bypasses-decoder',command(out),0);r['report']=json.loads((out/'result.json').read_text())
 assert all(x['text']=='' and x['token_count']==0 and x['decoder_executed'] is False and x['timings']['decoder_ms'] is None for x in r['report']['records'])
 models['predictor'].write_text('fixture')
 for stage in ('encoder','decoder'):
  models[stage].write_text('fail-infer');out=tmp/('fail-'+stage);r=run('failure-'+stage,command(out),2);r['report']=json.loads((out/'failed.json').read_text());assert r['report']['status']=='failed' and r['report']['inference_attempted'] is True and r['report']['inference_executed'] is None
  assert not (out/'result.json').exists();models[stage].write_text('fixture')
 models['predictor'].write_text('fail-load');out=tmp/'fail-load';r=run('load-failure-report',command(out),2);r['report']=json.loads((out/'failed.json').read_text());assert r['report']['inference_attempted'] is False;models['predictor'].write_text('fixture')
 entries=json.loads(MANIFEST.read_text())
 for e in entries:e['feature_file']=str(MANIFEST.parent/e['feature_file'])
 entries[1]['feature_sha256']='0'*64
 bad=tmp/'bad.json';bad.write_text(json.dumps(entries));out=tmp/'partial';r=run('second-feature-failure-retains-first',command(out,bad),2);r['report']=json.loads((out/'failed.json').read_text());assert len(r['report']['records'])==1 and r['report']['inference_executed'] is True and r['report']['current_utt_id']==entries[1]['utt_id']
(HERE/'summary.json').write_text(json.dumps({'scope':'host-fixture; actual CLI/CIF/feature/JSON but no vendor SDK or HBM inference','runs':records},ensure_ascii=False,indent=2)+'\n')
print(f'{len(records)} full native CLI host cases passed')
