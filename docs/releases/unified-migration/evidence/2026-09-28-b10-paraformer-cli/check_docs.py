"""Run new CLI README commands in fresh cwd; never run board/download blocks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
records=[]
seen={}
for folder in ('','runtime/python','model','test_data'):
    anchors=[]
    for filename in ('README.md','README_cn.md'):
        path=ROOT/'samples/speech/paraformer'/folder/filename
        content=path.read_text()
        anchors.append(set(re.findall(r'<a id="([^"]+)"',content)))
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if target.startswith(('https://','http://')): continue
            file,_,anchor=target.partition('#')
            destination=(path.parent/file).resolve() if file else path
            assert destination.exists(),target
            if anchor:
                assert f'<a id="{anchor}">' in destination.read_text(),target
        for index,block in enumerate(re.findall(r'```bash\n(.*?)```',content,re.S)):
            if 'runtime/python/main.py' not in block and 'model/download_model.sh' not in block:
                continue
            if not any(mode in block for mode in ('--preprocess-only','--dry-run','--list-models','--help')):
                records.append({'path':str(path.relative_to(ROOT)),'block':index,'status':'not-run-board-or-download'})
                continue
            key=hashlib.sha256(block.encode()).hexdigest()
            if key not in seen:
                with tempfile.TemporaryDirectory() as directory:
                    cwd=Path(directory)
                    (cwd/'samples').symlink_to(ROOT/'samples',target_is_directory=True)
                    env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+':'+env['PATH'];env['PYTHON']=sys.executable
                    p=subprocess.run(['bash','-e','-c',block],cwd=cwd,env=env,capture_output=True,text=True)
                    log=f'doc-{len(seen)}.log'
                    (HERE/log).write_text(p.stdout+p.stderr)
                    assert p.returncode==0,p.stderr
                    result_files=list((cwd/'outputs').glob('*/result.json')) if (cwd/'outputs').exists() else []
                    for result_path in result_files:
                        result=json.loads(result_path.read_text())
                        assert result['status']=='completed' and result['inference_executed'] is False
                        assert result['utterances']
                    seen[key]={'rc':p.returncode,'log':log,'result_files':len(result_files)}
            records.append({'path':str(path.relative_to(ROOT)),'block':index,**seen[key]})
    assert anchors[0]==anchors[1],folder
# All parser options must be documented in both runtime guides.
sys.path.insert(0,str(ROOT))
from samples.speech.paraformer.runtime.python.main import build_parser
options=[s for action in build_parser()._actions for s in action.option_strings if s.startswith('--')]
for filename in ('README.md','README_cn.md'):
    content=(ROOT/'samples/speech/paraformer/runtime/python'/filename).read_text()
    assert all(option in content for option in options)
summary={'unique_executed_blocks':len(seen),'records':records,'parser_options_documented':options,'matching_bilingual_anchors':True,'python':sys.version}
(HERE/'docs-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(f'{len(seen)} distinct CLI blocks passed; local links, anchors and all parser options checked')
