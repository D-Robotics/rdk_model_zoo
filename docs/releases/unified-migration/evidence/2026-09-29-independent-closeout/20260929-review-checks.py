import subprocess,json,hashlib,datetime
from pathlib import Path
repo=Path('/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration')
out=repo/'docs/releases/unified-migration/evidence/2026-09-29-independent-closeout'
py=str(repo.parent/'rdk_model_zoo/.venv/bin/python')
cmake=str(repo.parent/'.coordination/native-build-tools/cmake/data/bin/cmake')
ctest=str(Path(cmake).with_name('ctest'))
build=str(repo.parent/'.coordination/gemma-vision-sanitized')
checks=[('gemma-host',[py,'-m','unittest','discover','-s','samples/llm/gemma4-e2b/tests','-v']),('gemma-build',[cmake,'--build',build,'-j','4']),('gemma-sanitizers',[ctest,'--test-dir',build,'--output-on-failure']),('skills-host',[py,'-m','unittest','discover','-s','skills/tests','-v']),('skills-pack',[py,'skills/tools/validate_pack.py','--pack-root','skills']),('skills-references',[py,'skills/tools/sync_references.py'])]
records=[]
paths=[p for root in ['samples/llm/gemma4-e2b','skills'] for p in (repo/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc']]
hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for name,argv in checks:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(argv,cwd=repo,text=True,capture_output=True)
 (out/(name+'.stdout.log')).write_text(p.stdout);(out/(name+'.stderr.log')).write_text(p.stderr)
 records.append(dict(id=name,argv=argv,cwd=str(repo),started_at=start,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=p.returncode))
 (out/'checks.json').write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),checks=records,hashes=hashes),indent=2)+'\n')
 print(name,p.returncode,flush=True)
print('changed_during_checks',[str(p.relative_to(repo)) for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p.relative_to(repo))]],flush=True)
