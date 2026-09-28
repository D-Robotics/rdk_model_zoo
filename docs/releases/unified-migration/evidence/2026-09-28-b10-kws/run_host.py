"""Host-only KWS and shared regression, never board/SDK execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
ENV=dict(os.environ);ENV['PATH']='/opt/homebrew/opt/node@22/bin'+os.pathsep+ENV['PATH']
commands=[(name,[sys.executable,'-m','unittest','discover','-s',path]) for name,path in {'kws':'samples/speech/kws/tests','shared':'samples/_shared/tests','resnet':'samples/vision/resnet/tests','ultralytics':'samples/vision/ultralytics_yolo/tests','ocr':'samples/vision/paddle_ocr/tests','checker':'tools/sample_contract/tests'}.items()]
commands+=[('publisher',['npm','--prefix','tools/catalog-publisher','test']),('catalog-check',['npm','--prefix','tools/catalog-publisher','run','catalog:check']),('contracts',[sys.executable,'tools/sample_contract/check.py','--scope','migration','--parser-mode','import','--report',str(OUT/'contracts.json')])]
def run(item):
 name,argv=item;start=datetime.now(timezone.utc).isoformat()
 with (OUT/(name+'.log')).open('w') as stream:result=subprocess.run(argv,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
 return {'name':name,'argv':argv,'cwd':str(ROOT),'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'rc':result.returncode,'log':name+'.log'}
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,commands))
links=[]
for page in [ROOT/'README.md',ROOT/'README_cn.md',ROOT/'samples/README.md',ROOT/'samples/README_cn.md']:
 for href in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',page.read_text()):
  if href.startswith(('http:','https:','mailto:')):continue
  raw,_,anchor=href.partition('#');dest=(page.parent/unquote(raw)).resolve() if raw else page
  assert dest.exists(),(page,href)
  links.append({'page':str(page.relative_to(ROOT)),'href':href})
(OUT/'host-results.json').write_text(json.dumps({'runs':results,'entry_links':links,'board_or_sdk_executed':False},indent=2)+'\n')
files=[p for p in (ROOT/'samples/speech/kws').rglob('*') if p.is_file() and '__pycache__' not in str(p)]+[ROOT/'docs/release/s/models.yaml']
(OUT/'implementation-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
print([(r['name'],r['rc']) for r in results]);sys.exit(any(r['rc'] for r in results))
