"""Execute archived Python frontend and native normalization, no SDK or boards."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,tempfile
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from scipy.signal import resample
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from samples.speech.asr.runtime.python.audio_io import read_chunks
from samples.speech.asr.runtime.python.frontend import Config,prepare_chunk
source=ROOT/'platforms/s/samples/speech/asr/runtime/python/asr.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ASR')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pre_process')
method.returns=None
for arg in method.args.args:arg.annotation=None
helper=ROOT/'platforms/s/utils/py_utils/nn_math.py'
normal=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='zscore_normalize_lastdim')
space={'np':np,'sf':sf};exec(compile(ast.fix_missing_locations(ast.Module(body=[normal],type_ignores=[])),str(helper),'exec'),space)
space['nn_math']=SimpleNamespace(zscore_normalize_lastdim=space['zscore_normalize_lastdim']);exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),space)
probe=ROOT.parent/'.coordination/asr-normalize-probe'
command=['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I',str(ROOT/'samples/speech/asr/runtime/cpp/inc'),str(ROOT/'samples/speech/asr/runtime/cpp/tests/normalize_probe.cc'),'-o',str(probe)]
subprocess.run(command,check=True,cwd=ROOT)
records=[]
with tempfile.TemporaryDirectory(prefix='asr-frontend-') as tmp:
    tmp=Path(tmp);bundled=ROOT/'samples/speech/asr/test_data/chi_sound.wav'
    wave,rate=sf.read(bundled,dtype='float32')
    stereo=np.stack([resample(wave,round(len(wave)*44100/16000)),resample(wave,round(len(wave)*44100/16000))*.25],axis=1).astype(np.float32)
    sf.write(tmp/'stereo.wav',stereo,44100,subtype='FLOAT')
    sf.write(tmp/'short.wav',np.full(40,.5,np.float32),8000,subtype='FLOAT')
    for name,path in [('bundled',bundled),('stereo-44100',tmp/'stereo.wav'),('constant-8000',tmp/'short.wav')]:
        dummy=SimpleNamespace(cfg=Config(),model_name='asr',input_names=['audio'])
        original=list(space['pre_process'](dummy,str(path)))
        chunks=list(read_chunks(path));assert len(chunks)==len(original)
        for index,(chunk,old) in enumerate(zip(chunks,original)):
            prepared=prepare_chunk(chunk.waveform,chunk.sample_rate)
            expected=old['asr']['audio'];np.testing.assert_array_equal(prepared.tensor,expected)
            mono=chunk.waveform.mean(axis=1) if chunk.waveform.ndim==2 else chunk.waveform
            if chunk.sample_rate!=16000:mono=resample(mono,round(len(mono)*16000/chunk.sample_rate))
            input_path=tmp/'mono.bin';output_path=tmp/'normalized.bin';np.asarray(mono,dtype=np.float32).tofile(input_path)
            run=subprocess.run([str(probe),str(input_path),str(output_path)],capture_output=True,check=True)
            native=np.fromfile(output_path,dtype=np.float32).reshape(1,30000)
            diff=float(np.max(np.abs(native-prepared.tensor)));assert diff<2e-6,diff
            records.append({'case':name,'chunk':index,'source_frames':len(chunk.waveform),'source_rate':chunk.sample_rate,'valid_samples':prepared.valid_samples,'source_python_maxdiff':float(np.max(np.abs(prepared.tensor-expected))),'native_normalize_maxdiff':diff,'prepared_sha256':hashlib.sha256(prepared.tensor.tobytes()).hexdigest(),'native_rc':run.returncode})
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'normalization_source_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'probe_build':command,'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'records':records,'native_resampling_compared':False,'board_or_sdk_executed':False}
(OUT/'frontend-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
