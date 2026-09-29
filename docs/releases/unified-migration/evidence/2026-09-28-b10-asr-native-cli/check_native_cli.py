"""Real native audio/CLI with explicitly marked fixture transport; never SDK."""
from pathlib import Path
from datetime import datetime,timezone
import contextlib,hashlib,io,json,os,subprocess,sys
from unittest.mock import patch
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from samples.speech.asr.runtime.cpp import launcher
BINARY=ROOT.parent/'.coordination/asr-native-library/tests/asr_cli_fixture'
RUNS=OUT/'fixture-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');RUNS.mkdir(parents=True)
model=RUNS/'fixture-model.txt';model.write_text('host-only model placeholder; no compiled network\n')
sha=hashlib.sha256(model.read_bytes()).hexdigest();vocab=ROOT/'samples/speech/asr/test_data/vocab.json';audio=ROOT/'samples/speech/asr/test_data/chi_sound.wav'
empty=RUNS/'empty.wav';sf.write(empty,np.array([],np.float32),16000)
nan=RUNS/'nan.wav';sf.write(nan,np.array([np.nan],np.float32),16000,subtype='FLOAT')
results=[]
def run(name,updates=None,extra=(),environment=None,expected=0):
 options={'--target':'s100','--asset-id':'s:asr:s100/asr.hbm','--model-path':str(model),'--model-sha256':sha,'--audio-file':str(audio),'--vocab-file':str(vocab),'--output-dir':str(RUNS/name),'--decode-mode':'ctc'}
 options.update(updates or {})
 argv=[str(BINARY)]+[value for pair in options.items() for value in pair]+list(extra)
 env=dict(os.environ);env.pop('ASR_FIXTURE_FAIL_AFTER',None);env['ASR_FIXTURE_TARGET']='s100';env.update(environment or {})
 started=datetime.now(timezone.utc).isoformat();p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True)
 (RUNS/(name+'.stdout.log')).write_bytes(p.stdout);(RUNS/(name+'.stderr.log')).write_bytes(p.stderr)
 record=dict(name=name,argv=argv,cwd=str(ROOT),started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),rc=p.returncode,expected_rc=expected,fixture_environment=environment or {})
 results.append(record);assert p.returncode==expected,(name,p.stderr)
 return options
for name,target,mode in [('s100-ctc','s100','ctc'),('s100-legacy','s100','legacy'),('s600-ctc','s600','ctc')]:
 run(name,{'--target':target,'--asset-id':f's:asr:{target}/asr.hbm','--decode-mode':mode},environment={'ASR_FIXTURE_TARGET':target})
 report=json.loads((RUNS/name/'result.json').read_text())
 assert report['execution_backend']=='host-fixture' and report['text']==('AAAAAA' if mode=='ctc' else 'AAAAAAAAA')
 assert [r['source_frames'] for r in report['chunks']]==[30000,30000,13440]
 assert [r['valid_target_samples'] for r in report['chunks']]==[30000,30000,13440]
run('target-mismatch',environment={'ASR_FIXTURE_TARGET':'s600'},expected=2)
run('unsupported-target',{'--target':'s100p'},expected=2)
run('asset-mismatch',{'--asset-id':'s:asr:s600/asr.hbm'},expected=2)
run('digest-mismatch',{'--model-sha256':'0'*64},expected=2)
run('vocabulary-mismatch',{'--vocab-file':str(model)},expected=2)
run('missing-model',{'--model-path':str(RUNS/'missing')},expected=2)
run('empty-audio',{'--audio-file':str(empty)},expected=2)
run('nonfinite-audio',{'--audio-file':str(nan)},expected=2)
(RUNS/'existing').mkdir();marker=RUNS/'existing/marker';marker.write_text('preserve')
run('existing',expected=2);assert marker.read_text()=='preserve'
run('duplicate-option',extra=['--target','s600'],expected=2)
run('unknown-option',extra=['--allow-unknown'],expected=2)
run('late-failure',environment={'ASR_FIXTURE_FAIL_AFTER':'1'},expected=2)
failed=json.loads((RUNS/'late-failure/failed.json').read_text());assert len(failed['chunks'])==1 and failed['status']=='failed';assert not (RUNS/'late-failure/result.json').exists()
# The public launcher must reject a successful host-fixture report, retaining
# raw process output. Only board identity is patched for this host test.
args=['--target','s100','--asset-id','s:asr:s100/asr.hbm','--model-path',str(model),'--binary',str(BINARY),'--output-dir',str(RUNS/'launcher-rejects-fixture')]
stdout=io.StringIO();stderr=io.StringIO()
with patch.object(launcher,'require_execution_target',return_value='s100'),contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):rc=launcher.main(args)
assert rc==2
record=json.loads((RUNS/'launcher-rejects-fixture/launch-report.json').read_text());assert record['status']=='failed' and record['executed'] and not record['runtime_metadata_verified']
(RUNS/'launcher.stdout.log').write_text(stdout.getvalue());(RUNS/'launcher.stderr.log').write_text(stderr.getvalue())
results.append(dict(name='launcher-rejects-fixture',argv=args,rc=rc,expected_rc=2,board_identity='patched; host fixture only'))
(OUT/'native-cli-results.json').write_text(json.dumps(dict(runs=results,evidence_directory=str(RUNS.relative_to(ROOT)),binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),json_header_sha256=hashlib.sha256((ROOT.parent/'.coordination/asr-json/include/nlohmann/json.hpp').read_bytes()).hexdigest(),json_version='3.11.3',sdk_or_board_executed=False),indent=2)+'\n')
print(len(results),'native/launcher cases passed; all inference is host-fixture')
