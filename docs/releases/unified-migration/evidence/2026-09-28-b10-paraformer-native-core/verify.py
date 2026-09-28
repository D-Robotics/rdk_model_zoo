"""Compile actual legacy/new native CIF and compare with unified Python."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT))
from samples.speech.paraformer.runtime.python.cif import cif_numpy
from samples.speech.paraformer.runtime.python.decoding import decode_logits
SOURCE='platforms/s/samples/speech/paraformer/runtime/cpp/src/paraformer.cpp'
source=(ROOT/SOURCE).read_bytes()
pinned=subprocess.check_output(['git','show','380e1a2bf42041af54be6f34935e50197cfadff9:samples/speech/paraformer/runtime/cpp/src/paraformer.cpp'],cwd=ROOT)
assert source==pinned
text=source.decode()
start=text.index('constexpr int MAX_LABEL_LEN = 100;')
end=text.index('// ==== simple char-level',start)
header=text[start:end]
(HERE/'source_cif.h').write_text(header)
cpp=ROOT/'samples/speech/paraformer/runtime/cpp'
records=[]
with tempfile.TemporaryDirectory() as directory:
    work=Path(directory)
    common=['clang++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(HERE/'driver.cc')]
    commands=[common+['-DLEGACY_CIF', '-I'+str(HERE),'-o',str(work/'legacy')],
              common+['-I'+str(cpp/'inc'),str(cpp/'src/contract.cc'),'-o',str(work/'unified')]]
    for index,command in enumerate(commands):
        result=subprocess.run(command,capture_output=True,text=True)
        (HERE/f'driver-build-{index}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
    cases=[]
    for seed in range(6):
        rng=np.random.default_rng(seed)
        weights=rng.uniform(0,.8, (1,401)).astype(np.float32)
        hidden=rng.normal(size=(1,401,512)).astype(np.float32)
        for length in (0,1,37,400):cases.append((f'random-{seed}-{length}',weights,hidden,length))
    cases.append(('zero',np.zeros((1,401),np.float32),np.zeros((1,401,512),np.float32),400))
    cases.append(('cap',np.ones((1,401),np.float32),np.broadcast_to(np.arange(401,dtype=np.float32)[None,:,None],(1,401,512)).copy(),400))
    fractional=np.zeros((1,401),np.float32);fractional[0,:3]=[.75,.75,.5]
    hidden=np.zeros((1,401,512),np.float32);hidden[0,:3]=np.array([2,6,10],np.float32)[:,None]
    cases.append(('fractional',fractional,hidden,3))
    for label,weights,hidden,length in cases:
        binary=weights.tobytes()+hidden.tobytes()
        (work/'input.bin').write_bytes(binary)
        acoustic,count=cif_numpy(weights,hidden,real_T=length)
        expected=count.tobytes()+acoustic.tobytes()
        outputs=[]
        for kind in ('legacy','unified'):
            command=[str(work/kind),'cif',str(work/'input.bin'),str(work/f'{kind}.bin'),str(length)]
            result=subprocess.run(command,capture_output=True,text=True)
            assert result.returncode==0,(label,kind,result.stderr)
            outputs.append((work/f'{kind}.bin').read_bytes())
        assert outputs[0]==outputs[1]==expected,label
        records.append({'case':label,'valid_frames':length,'token_count':int(count[0]),'input_sha256':hashlib.sha256(binary).hexdigest(),'output_sha256':hashlib.sha256(expected).hexdigest(),'native_source_equal':True,'python_equal':True})
    vocabulary=json.loads((ROOT/'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json').read_text())
    assert all('\n' not in t and '\r' not in t for t in vocabulary)
    (work/'vocabulary.txt').write_text('\n'.join(vocabulary)+'\n')
    text_records=[]
    for seed in range(5):
        logits=np.random.default_rng(seed).normal(size=(1,100,8404)).astype(np.float32)
        (work/'logits.bin').write_bytes(logits.tobytes())
        for count in (0,1,37,100):
            result=subprocess.run([str(work/'unified'),'decode',str(work/'logits.bin'),str(work/'text'),str(count),str(work/'vocabulary.txt')],capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            expected,_=decode_logits(logits,count,vocabulary)
            actual=(work/'text').read_bytes()
            assert actual==expected.encode(),(seed,count)
            text_records.append({'seed':seed,'count':count,'text_sha256':hashlib.sha256(actual).hexdigest(),'python_equal':True})
summary={'scope':'host numerical kernel only; no native SDK or model inference',
 'source_commit':'380e1a2bf42041af54be6f34935e50197cfadff9','source_matches_git':True,
 'source_sha256':hashlib.sha256(source).hexdigest(),'extracted_cif_sha256':hashlib.sha256(header.encode()).hexdigest(),
 'cif_cases':records,'text_cases':text_records,'python':sys.version,'numpy':np.__version__,
 'platform':platform.platform(),'compiler':subprocess.check_output(['clang++','--version'],text=True),
 'byte_order':sys.byteorder,'compiler_flags':['-O2','-ffp-contract=off','-fsanitize=address,undefined']}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(f'{len(records)} native/source/Python CIF byte comparisons; {len(text_records)} native/Python text comparisons passed')
