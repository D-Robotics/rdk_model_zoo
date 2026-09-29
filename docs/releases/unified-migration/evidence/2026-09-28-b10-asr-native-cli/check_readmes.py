"""Execute bilingual host shell and real-frontend API examples; check local links."""
from pathlib import Path
from urllib.parse import unquote
import contextlib,hashlib,io,json,os,re,subprocess,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent;SAMPLE=ROOT/'samples/speech/asr'
sys.path.insert(0,str(ROOT))
from samples.speech.asr.tests.test_runtime import FakeRuntime
from samples.speech.asr.runtime.python import model_runner
original=model_runner.RuntimeModelRunner
records=[];links=[]
for en in sorted(SAMPLE.rglob('README.md')):
    zh=en.with_name('README_cn.md');assert zh.exists()
    texts=[p.read_text() for p in (en,zh)]
    shell=[re.findall(r'```bash\n(.*?)```',s,re.S) for s in texts];assert shell[0]==shell[1],en
    for i,command in enumerate(shell[0]):
        env=dict(os.environ);env['PYTHON']=sys.executable;env['ASR_AUDIO_PREFIX']=str(ROOT.parent/'.coordination/asr-audio-deps/install');env['ASR_JSON_INCLUDE']=str(ROOT.parent/'.coordination/asr-json/include');env['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(ROOT.parent/'.coordination/native-build-tools/cmake/data/bin')+os.pathsep+env['PATH']
        result=subprocess.run(['bash','-e','-c',command],cwd=ROOT,env=env,capture_output=True)
        name=str(en.parent.relative_to(SAMPLE)).replace('/','-').replace('.','root')+f'-shell-{i}'
        (OUT/(name+'.stdout.log')).write_bytes(result.stdout);(OUT/(name+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode==0,(en,result.stderr)
        records.append({'page':str(en.relative_to(ROOT)),'kind':'host-shell','command':command,'rc':result.returncode,'stdout':name+'.stdout.log','stderr':name+'.stderr.log'})
    blocks=[re.findall(r'```python\n(.*?)```',s,re.S) for s in texts];assert blocks[0]==blocks[1]
    for code in blocks[0]:
        stream=io.StringIO()
        with patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda selection:original(selection,runtime=FakeRuntime())),contextlib.redirect_stdout(stream):exec(compile(code,str(en),'exec'),{})
        assert stream.getvalue().strip() == 'AAAAAA', stream.getvalue()
        records.append({'page':str(en.relative_to(ROOT)),'kind':'real-frontend-api-fake-sdk','stdout':stream.getvalue()})
    for page,text in zip((en,zh),texts):
        text=re.sub(r"```.*?```", "", text, flags=re.S)
        for link in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text):
            if link.startswith(('https:','http:','mailto:')):continue
            raw,_,anchor=link.partition('#');target=(page.parent/unquote(raw)).resolve() if raw else page
            assert target.exists(),(page,link)
            if anchor:assert 'id="'+anchor+'"' in target.read_text(),(page,link)
            links.append({'page':str(page.relative_to(ROOT)),'link':link})
(OUT/'readme-results.json').write_text(json.dumps({'commands':records,'local_links':links,'board_or_sdk_executed':False},indent=2)+'\n')
print(len(records),'executed bilingual examples;',len(links),'local links')
