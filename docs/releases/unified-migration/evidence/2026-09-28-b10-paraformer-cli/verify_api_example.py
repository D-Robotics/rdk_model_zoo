"""Execute the documented API with real frontend and explicitly synthetic SDKs."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT))
from samples.speech.paraformer.runtime.python import runtime
from samples.speech.paraformer.tests.test_binding import metadata

original=runtime.load_runtime
calls=[]
schedules=[]
def factory(path):
    stage=next(s for s in ('encoder','predictor','decoder') if s in Path(path).name)
    meta=metadata(stage)
    sdk=SimpleNamespace(model_names=[stage])
    for name in ('input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes'):
        setattr(sdk,name,{stage:getattr(meta,name)})
    sdk.set_scheduling_params=lambda **kwargs:schedules.append((stage,kwargs))
    def run(inputs):
        calls.append(stage)
        values={n:np.zeros(shape,dtype=meta.output_dtypes[n]) for n,shape in meta.output_shapes.items()}
        if stage=='predictor':values['/predictor/Add_output_0'][0,:2]=1
        if stage=='decoder':
            values['logits'][0,:2,3]=1
            values['token_num'][:]=2
        return {stage:values}
    sdk.run=run
    return sdk
examples=[]
for name in ('README.md','README_cn.md'):
    text=(ROOT/'samples/speech/paraformer/runtime/python'/name).read_text()
    example=next(block for block in re.findall(r'```python\n(.*?)```',text,re.S) if 'bundle = load_runtime' in block)
    examples.append(example)
assert examples[0]==examples[1]
with tempfile.TemporaryDirectory() as directory:
    cwd=Path(directory)
    sample=cwd/'samples/speech/paraformer'
    (sample/'model/s100').mkdir(parents=True)
    (sample/'test_data/audio').mkdir(parents=True)
    shutil.copyfile(ROOT/'samples/speech/paraformer/model/am.mvn',sample/'model/am.mvn')
    shutil.copyfile(ROOT/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json',sample/'model/s100/tokens.json')
    shutil.copyfile(ROOT/'samples/speech/paraformer/test_data/audio/BAC009S0724W0121.wav',sample/'test_data/audio/BAC009S0724W0121.wav')
    previous=Path.cwd()
    output=io.StringIO()
    try:
        os.chdir(cwd)
        with patch.object(runtime,'load_runtime',side_effect=lambda selections,vocabulary:original(selections,vocabulary,runtime_factory=factory)), contextlib.redirect_stdout(output):
            exec(compile(example,'README API example','exec'),{})
    finally:os.chdir(previous)
assert calls==['encoder','predictor','decoder']
assert [s for s,_ in schedules]==['encoder','predictor','decoder']
assert 'andand False' in output.getvalue()
summary={'scope':'documented complete API; real FunASR frontend, synthetic SDK/model outputs only',
 'matching_bilingual_example':True,'model_calls':calls,'schedules':schedules,
 'stdout':output.getvalue(),'expected_fixture_text':'andand','board_inference':'not-run'}
(HERE/'api-example.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print('Complete bilingual API example passed with real frontend and explicit SDK doubles')
