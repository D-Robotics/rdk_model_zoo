from pathlib import Path
import json, hashlib, shutil, collections
b=Path(__file__).resolve().parent
r=b.parent/'rdk-b7-board-integration'
out=r/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout/agent-behavior'
out.mkdir(exist_ok=True)
source=b/'20260928-skills-behavior'
for n in ['environment.json','snapshot-hashes.json']:
    shutil.copyfile(source/n,out/n)
shutil.copytree(source/'pack-snapshot',out/'pack-snapshot',dirs_exist_ok=True)
index=[]
for kind,root in [('initial',source),('prompted-correction',b/'20260929-behavior-corrections')]:
    for p in sorted(root.glob('AG*')):
        dest=out/kind/p.name
        dest.mkdir(parents=True,exist_ok=True)
        for n in ['prompt.md','fixture.json','execution.json','gitlinks.txt']:
            if (p/n).exists(): shutil.copyfile(p/n,dest/n)
        if (p/'customer/README.md').exists():
            shutil.copyfile(p/'customer/README.md',dest/'customer-README-after.md')
        events=[]; calls=collections.Counter(); final=None
        for line in (p/'trace.jsonl').read_text().splitlines():
            x=json.loads(line); typ=x.get('type'); blocks=x.get('message',{}).get('content',[])
            if typ in ['assistant','user'] and isinstance(blocks,list):
                blocks=[y for y in blocks if y.get('type') in ['text','tool_use','tool_result']]
                for y in blocks:
                    if y.get('type')=='tool_use': calls[y['name']]+=1
                if blocks: events.append({'type':typ,'content':blocks})
            if typ=='result':
                final=x.get('result','')
                events.append({'type':'result','result':final,'is_error':x.get('is_error'),'duration_ms':x.get('duration_ms')})
        assert final is not None,p
        (dest/'observable-trace.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in events))
        (dest/'final.md').write_text(final+'\n')
        execution=json.loads((p/'execution.json').read_text())
        index.append({'case':p.name,'kind':kind,'raw_trace_sha256':hashlib.sha256((p/'trace.jsonl').read_bytes()).hexdigest(),'tool_calls':dict(calls),'execution_exit_code':execution.get('exit_code',execution.get('rc'))})
        if kind=='prompted-correction':
            shutil.copytree(p/'skill',dest/'skill-snapshot',dirs_exist_ok=True)
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
h=json.loads((source/'snapshot-hashes.json').read_text())
drift=[n for n,v in h.items() if hashlib.sha256((r/'skills'/n).read_bytes()).hexdigest()!=v]
(out/'source-reconciliation.json').write_text(json.dumps({'initial_snapshot_files':len(h),'changed_since_initial':drift,'meaning':'Initial snapshot predates release-contract factual correction only. AG13 prompted correction uses corrected release skill; original failures preserved.'},indent=2)+'\n')
print(json.dumps(index,indent=2));print('drift',drift)
