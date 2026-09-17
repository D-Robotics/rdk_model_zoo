REMOTE='/tmp/rdk-model-zoo-20260916-contract-pilot'
import pathlib,subprocess,sys,hashlib,json
path=pathlib.Path(REMOTE)/'python-deps'
command=[sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','--disable-pip-version-check','--target',str(path),'--index-url','https://pypi.org/simple','pyclipper==1.4.0']
print('COMMAND '+json.dumps(command),flush=True)
subprocess.run(command,check=True,timeout=120)
sys.path.insert(0,str(path))
import pyclipper
print('VERSION '+pyclipper.__version__,flush=True)
for f in sorted(path.rglob('*.so')):print('NATIVE_FILE '+json.dumps({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}),flush=True)
