"""Run at repository root; real source inputs, no SDK, quantization or robot calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

root=Path.cwd()
source=root/'platforms/x5/samples/robotics/himloco'
source_pin='ac115717197920355fc390bb04299b20e6436864'
paths=subprocess.check_output(['git','ls-files',str(source.relative_to(root))],text=True).splitlines()
inventory=[]
for name in paths:
    local=Path(name)
    original=subprocess.check_output(['git','show',f'{source_pin}:{str(local).removeprefix("platforms/x5/")}'])
    assert original==local.read_bytes(),name
    inventory.append({'path':name,'sha256':hashlib.sha256(original).hexdigest()})
path=source/'runtime/python/himloco.py'
spec=importlib.util.spec_from_file_location('himloco_source_contract',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
legacy=module.HimLoco.__new__(module.HimLoco)
legacy.model_name='source-fixture';legacy.input_name='obs_history';legacy.input_shape=(1,270)
legacy.output_name='actions';legacy._last_latency_ms=3.5
sys.path.insert(0,str(root))
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

def runner(feed):
    # Deterministic raw fixture, not model or SDK output.
    return {'actions':np.arange(12,dtype=np.float32).reshape(1,12)}

task=HimLocoTask(runner)
manifest=json.loads((source/'test_data/runtime-input-manifest.json').read_text())
records=[]
for entry in manifest['records']:
    data=(source/'test_data'/entry['file']).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    assert digest==entry['sha256']
    values=np.frombuffer(data,dtype='<f4').reshape(1,270)
    old=legacy.pre_process(values)[legacy.model_name]['obs_history']
    new=task.pre_process(values).tensors['obs_history']
    np.testing.assert_array_equal(old,new)
    assert not np.shares_memory(new,values)
    records.append({'source_index':entry['source_index'],'sha256':digest,'preprocessing_equal':True,'owns_independent_storage':True})
actions=runner(None)['actions']
old=legacy.post_process({legacy.model_name:{'actions':actions}}).actions
new=task.predict(np.zeros(270,np.float32)).actions
np.testing.assert_array_equal(old,new)
out=Path(__file__).parent/'source-comparison.json'
out.write_text(json.dumps({'source_pin':source_pin,'source_files':inventory,'real_observation_comparisons':records,'postprocess':'same action values for explicit synthetic runner output; not actual policy inference','sdk_executed':False,'quantization_executed':False,'robot_control_executed':False},indent=2)+'\n')
print(f'{len(inventory)} source files match pin; {len(records)} real inputs preserve preprocessing; synthetic action fixture matches source')
