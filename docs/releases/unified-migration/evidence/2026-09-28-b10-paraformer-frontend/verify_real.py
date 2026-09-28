"""Mandatory real FunASR source comparison; no SDK, HBM or board execution."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Union

import numpy as np
import soundfile as sound_file
import torch
from funasr.frontends.wav_frontend import WavFrontend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend

source_path = 'platforms/s/samples/speech/paraformer/runtime/python/paraformer.py'
source = (ROOT / source_path).read_bytes()
pinned = subprocess.check_output(['git','show','380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/runtime/python/paraformer.py'],cwd=ROOT)
assert source == pinned
classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in ('ParaformerFrontendConfig', 'ParaformerFrontend')]
namespace = dict(dataclass=dataclass,Path=Path,Union=Union,np=np,sound_file=sound_file,torch=torch,WavFrontend=WavFrontend)
exec(compile(ast.Module(body=classes,type_ignores=[]), str(ROOT/source_path), 'exec'),namespace)
cmvn = ROOT/'samples/speech/paraformer/model/am.mvn'
legacy = namespace['ParaformerFrontend'](namespace['ParaformerFrontendConfig'](cmvn_path=str(cmvn)))
unified = ParaformerFrontend(cmvn)
source_audio = ROOT/'samples/speech/paraformer/test_data/audio'
inputs=[]
for path in sorted(source_audio.glob('*.wav')):
    original_audio = subprocess.check_output(['git','show','380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/test_data/audio/' + path.name], cwd=ROOT)
    assert path.read_bytes() == original_audio
    audio,rate=sound_file.read(path,dtype='float32')
    inputs.append((path.stem,audio,rate,hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(inputs)==2
first=inputs[0][1]
inputs.extend([
 ('stereo',np.stack([first,first*.5],axis=1),16000,None),
 ('silence',np.zeros(8000,np.float32),16000,None),
 ('long',np.resize(first,16000*30).astype(np.float32),16000,None),
 ('one_window',np.zeros(400,np.float32),16000,None),
 ('short_window',np.zeros(160,np.float32),16000,None),
])
records=[]
with tempfile.TemporaryDirectory() as directory:
    for label,audio,rate,source_hash in inputs:
        path=Path(directory)/f'{label}.wav'
        sound_file.write(path,audio,rate,subtype='FLOAT')
        # The source owns WAV loading; use identical loaded bytes for the new
        # numerical interface, including stereo samples and temporary WAV type.
        loaded,loaded_rate=sound_file.read(path,dtype='float32')
        torch.manual_seed(123456)
        original_state=torch.get_rng_state().clone()
        expected,expected_length=legacy.pre_process(path)
        source_changes_rng=not torch.equal(torch.get_rng_state(),original_state)
        assert source_changes_rng
        torch.set_rng_state(original_state)
        actual=unified.pre_process(loaded,loaded_rate)
        assert torch.equal(torch.get_rng_state(),original_state), label
        assert actual.valid_frames==expected_length
        assert actual.tensor.tobytes()==expected.tobytes(), (label,float(np.max(np.abs(actual.tensor-expected))))
        repeated=unified.pre_process(loaded,loaded_rate)
        assert repeated.tensor.tobytes()==actual.tensor.tobytes()
        assert torch.equal(torch.get_rng_state(),original_state)
        assert actual.truncated==(label=='long')
        assert not actual.tensor[0,actual.valid_frames:].any()
        records.append({'case':label,'samples':len(loaded),'source_wav_sha256':source_hash,
          'valid_frames':actual.valid_frames,'original_frames':actual.original_frames,
          'truncated':actual.truncated,'byte_equal':True,'max_abs_diff':0.0,
          'source_changed_rng':True,'unified_preserved_rng':True,'repeat_byte_equal':True,
          'features_sha256':hashlib.sha256(actual.tensor.tobytes()).hexdigest()})
# Failure injection verifies finally restoration after RNG has actually advanced.
original_backend=unified._frontend
def fail_after_random(*args):
    torch.rand(12)
    raise RuntimeError('intentional frontend failure after RNG use')
unified._frontend=fail_after_random
before=torch.get_rng_state().clone()
try:
    unified.pre_process(first,16000)
except RuntimeError as error:
    assert 'intentional' in str(error)
else:
    raise AssertionError('Expected injected failure')
assert torch.equal(before,torch.get_rng_state())
unified._frontend=original_backend
versions={name:importlib.metadata.version(name) for name in ('torch','torchaudio','funasr','numpy','soundfile','protobuf')}
summary={'scope':'real CPU FunASR features only; no HBM/SDK/board inference',
 'source_commit':'380e1a2bf42041af54be6f34935e50197cfadff9','source_matches_git':True,
 'source_sha256':hashlib.sha256(source).hexdigest(),'versions':versions,'python':sys.version,
 'cases':records,'exception_restores_rng':True,
 'concurrency_limit':'adapter calls serialize; external threads using global Torch RNG need caller coordination'}
(HERE/'real-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
