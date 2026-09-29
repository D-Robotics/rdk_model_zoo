"""Actual PaddleAudio frontend comparison, CPU only; no model/SDK inference."""
from pathlib import Path
import ast,hashlib,inspect,json,sys
import numpy as np
import paddle,paddleaudio
from paddleaudio.compliance.kaldi import fbank
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from samples.speech.kws.runtime.python.audio_io import load_audio
from samples.speech.kws.runtime.python.frontend import Config,prepare_waveform,paddle_fbank
source=ROOT/'platforms/s/samples/speech/kws/runtime/python/kws.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KWS');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='audio_trunc');method.decorator_list=[]
scope={'paddle':paddle,'THRES':60000};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
path=ROOT/'samples/speech/kws/test_data/sample.wav'
wave,rate=load_audio(path);source_wave,source_rate=paddleaudio.load(str(path));source_np=source_wave.numpy() if hasattr(source_wave,'numpy') else source_wave
assert source_rate==rate==16000
np.testing.assert_array_equal(source_np[0],wave)
cases={'sample':wave,'silence':np.zeros(16000,np.float32),'long':np.tile(wave,2)}
records=[]
for name,wav in cases.items():
    original=scope['audio_trunc'](paddle.to_tensor(wav[None]),60000)
    prepared=prepare_waveform(wav,16000,Config())
    np.testing.assert_array_equal(original.numpy(),prepared)
    before=fbank(waveform=original,sr=16000,frame_shift=10,frame_length=25,n_mels=80).numpy()
    after=paddle_fbank(prepared,Config())
    np.testing.assert_array_equal(before,after)
    assert after.shape==(373,80) and after.dtype==np.float32 and np.isfinite(after).all()
    records.append({'case':name,'source_samples':len(wav),'shape':list(after.shape),'dtype':str(after.dtype),'max_abs_difference':float(np.max(np.abs(before-after))),'feature_sha256':hashlib.sha256(after.tobytes()).hexdigest()})
result={'paddle':paddle.__version__,'paddleaudio':paddleaudio.__version__,'paddleaudio_load':str(inspect.signature(paddleaudio.load)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'audio_decode_equal':True,'cases':records,'sdk_or_board_executed':False}
(OUT/'frontend-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
