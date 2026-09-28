import subprocess,json,re,hashlib
from pathlib import Path
from urllib.parse import unquote
r=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration');out=r/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout'
changed=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()
files=[p for p in changed if p.endswith('.md') and (p.startswith('samples/') or p=='CLAUDE.md')]
files += ['samples/vision/ultralytics_yolo/test_data/README.md','samples/vision/ultralytics_yolo/test_data/README_cn.md']
result={'files':{},'failures':[],'scope':'changed customer documentation; local file paths and existing command preservation, not remote-link validation'}
def fences(s):return re.findall(r'^```(?:bash|sh|shell|python|cpp|c\+\+|console)\s*\n(.*?)^```',s,re.M|re.S)
for rel in files:
 p=r/rel;s=p.read_text();old=subprocess.run(['git','show','HEAD:'+rel],cwd=r,capture_output=True,text=True)
 row={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'links':0,'old_command_blocks_unchanged':old.returncode!=0 or fences(s)==fences(old.stdout)}
 # Gemma intentionally changes API examples; documents are read separately.
 if 'gemma4-e2b' not in rel and not row['old_command_blocks_unchanged']:result['failures'].append([rel,'existing fenced blocks changed'])
 text=re.sub(r'^```.*?^```[^\n]*','',s,flags=re.M|re.S)
 for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
  target=target.strip().split(' "')[0].strip('<>');part=unquote(target.split('#')[0])
  if not part or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',part):continue
  row['links']+=1
  if not (p.parent/part).exists():result['failures'].append([rel,'missing path',target])
 result['files'][rel]=row
pairs=[]
for rel in files:
 if not rel.endswith('README.md'):continue
 p=r/rel;cn=p.with_name('README_cn.md')
 if not cn.exists():continue
 en=p.read_text();zh=cn.read_text();a=re.findall(r'<a id="([^"]+)"',en);b=re.findall(r'<a id="([^"]+)"',zh)
 pairs.append({'en':rel,'anchors_match':a==b})
 if a!=b:result['failures'].append([rel,'anchor parity'])
result['pairs']=pairs
assets=[]
base=r/'samples/vision/ultralytics_yolo/test_data'
for p in sorted(base.iterdir()):
 if p.name.startswith('README'):continue
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 for doc in ['README.md','README_cn.md']:
  if sha not in (base/doc).read_text():result['failures'].append([str(p),'missing documented hash',doc])
 old=subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(r))],cwd=r)
 assets.append({'path':str(p.relative_to(r)),'sha256':sha,'unchanged':old==p.read_bytes()})
for family in ['11','26']:
 p=r/f'samples/vision/yoloe/test_data/source_s{family}_result_figure.jpg';source=r/f'platforms/s/samples/vision/yoloe{family}_seg/test_data/result.jpg'
 assets.append({'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':str(source.relative_to(r)),'unchanged':p.read_bytes()==source.read_bytes()})
result['assets']=assets
assert all(a['unchanged'] for a in assets)
(out/'document-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('files',len(files),'links',sum(x['links'] for x in result['files'].values()),'pairs',len(pairs),'assets',len(assets),'failures',result['failures'])
