import re,json,hashlib,datetime
from pathlib import Path
from urllib.parse import unquote
r=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration');d=r/'docs/releases/unified-migration/evidence';old=json.loads((d/'2026-09-28-navigation-independent-review/readme-links-corrected.json').read_text());oldmap={x['file']:x for x in old['records']}
files=sorted(set(oldmap)|{'samples/vision/ultralytics_yolo/test_data/README.md','samples/vision/ultralytics_yolo/test_data/README_cn.md'})
records=[];missing=[];anchors=[];cache={}
def unfence(s):return re.sub(r'^```.*?^```[^\n]*','',s,flags=re.M|re.S)
def ids(p):
 if p not in cache:
  s=unfence(p.read_text());found=set(re.findall(r'<a\s+(?:id|name)=[\"\x27]([^\"\x27]+)',s));seen={}
  for t in re.findall(r'^#{1,6}\s+(.+?)\s*#*$',s,re.M):
   t=re.sub(r'<[^>]*>','',t);t=re.sub(r'[^\w\-\s]','',t.lower());t=re.sub(r'\s','-',t)
   n=seen.get(t,0);seen[t]=n+1;found.add(t+(f'-{n}' if n else ''))
  cache[p]=found
 return cache[p]
for rel in files:
 p=r/rel
 if not p.is_file():missing.append([rel,'document absent']);continue
 s=p.read_text();refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',unfence(s));count=0
 for target in refs:
  target=target.strip().split(' "')[0].strip('<>')
  if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('//'):continue
  path,_,frag=target.partition('#');q=(p.parent/unquote(path)).resolve() if path else p;count+=1
  if not q.exists():missing.append([rel,target]);continue
  if frag and q.suffix=='.md' and unquote(frag) not in ids(q):anchors.append([rel,target])
 h=hashlib.sha256(p.read_bytes()).hexdigest();records.append({'file':rel,'sha256':h,'local_references':count,'changed_since_prior':oldmap.get(rel,{}).get('sha256')!=h})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'missing_files':missing,'unresolved_anchors':anchors,'limits':'inline Markdown outside fenced code, no remote URLs, reference-style links or submodule interiors; heading heuristic flags require manual review'}
(d/'2026-09-29-independent-closeout/navigation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('files',len(records),'local refs',sum(x['local_references'] for x in records),'changed',sum(x['changed_since_prior'] for x in records),'missing',missing,'anchors',anchors)
