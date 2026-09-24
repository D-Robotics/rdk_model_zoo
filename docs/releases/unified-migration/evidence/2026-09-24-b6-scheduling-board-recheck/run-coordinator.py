import subprocess,json,pathlib,datetime,concurrent.futures
coord=pathlib.Path('../.coordination');sha='a72f92b41d4265e36b9fc7c5a85d0b15800cee8d';repo='/tmp/rdk-b7-bindings-73a6de1'
def run(item):
 host,target=item;records=[]
 def execute(label,cmd):
  r={'host':host,'label':label,'command':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=30',host,cmd],capture_output=True,text=True);r.update(rc=p.returncode,stdout=p.stdout,stderr=p.stderr,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());records.append(r);(coord/f'b6-{target}-scheduling-recheck.json').write_text(json.dumps({'commit':sha,'records':records},indent=2)+'\n');print(host,label,p.returncode,p.stdout[-450:],p.stderr[-650:],flush=True);return p.returncode
 check=f'git -C {repo} diff --exit-code && git -C {repo} diff --cached --exit-code && git -C {repo} fetch --depth=1 origin {sha} && git -C {repo} checkout --detach {sha} && git -C {repo} rev-parse HEAD'
 if execute('github-checkout',check):return
 for sample in ('efficient_sam','mobile_sam'):
  execute(sample+'-compare',f'cd {repo} && python3 samples/vision/{sample}/evaluator/compare.py --target {target} --output-dir /tmp/rdk-b6-{target}-{sample}-a72f92b')
  if target == 'x5':
   execute(sample+'-priority7-compare',f'cd {repo} && python3 samples/vision/{sample}/evaluator/compare.py --target {target} --priority 7 --output-dir /tmp/rdk-b6-{target}-{sample}-a72f92b-p7')
with concurrent.futures.ThreadPoolExecutor() as pool:list(pool.map(run,[('x5-8g','x5'),('s100','s100')]))
