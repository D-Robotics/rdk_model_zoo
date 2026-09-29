"""Actual libraries and unchanged source resampling; no SDK/model execution."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,sys
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from samples.speech.asr.runtime.python.frontend import prepare_chunk
from samples.speech.asr.runtime.python.audio_io import read_chunks
from samples.speech.asr.runtime.python.frontend import Config
SOURCE=ROOT/'platforms/s/samples/speech/asr/runtime/cpp'
NATIVE=ROOT/'samples/speech/asr/runtime/cpp'
PREFIX=ROOT.parent/'.coordination/asr-audio-deps/install'
records=[];builds=[]
with tempfile.TemporaryDirectory() as temp:
 tmp=Path(temp)
 original=(SOURCE/'src/audio_chunk_reader.cpp').read_text()
 # Keep the actual source read/mix/resample operations; expose their raw result
 # before the old normalization, which is deliberately corrected in migration.
 prefix,separator,_=original.partition('    // Apply z-score normalization')
 assert separator
 extracted=prefix+'    out_chunk = std::move(resampled);\n    return true;\n}\n'
 (tmp/'source_resampler.cpp').write_text(extracted)
 (tmp/'source_probe.cc').write_text('''#include "audio_chunk_reader.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char **argv) {
 if(argc!=3) return 2;
 AudioChunkReader reader(argv[1],30000,16000); std::vector<float> samples;size_t i=0;
 while(reader.next(samples)) {
  std::ofstream out(std::string(argv[2])+std::to_string(i++)+".bin",std::ios::binary);
  out.write(reinterpret_cast<const char *>(samples.data()),samples.size()*sizeof(float));
  if(!out) return 2;
  std::cout<<samples.size()<<'\\n';
 }
}
''')
 common=['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I',str(PREFIX/'include')]
 libs=[str(PREFIX/'lib/libsndfile.a'),str(PREFIX/'lib/libsamplerate.a')]
 for name,files,include in [('source',[tmp/'source_probe.cc',tmp/'source_resampler.cpp'],SOURCE/'inc'),('canonical',[NATIVE/'tests/audio_probe.cc',NATIVE/'src/audio_io.cc',NATIVE/'src/frontend.cc'],NATIVE/'inc')]:
  args=common+['-I',str(include)]+list(map(str,files))+libs+['-o',str(tmp/name)]
  result=subprocess.run(args,capture_output=True)
  (OUT/(name+'-build.log')).write_bytes(result.stdout+result.stderr)
  builds.append(dict(argv=args,rc=result.returncode));result.check_returncode()
 bundled=ROOT/'samples/speech/asr/test_data/chi_sound.wav'
 stereo=tmp/'stereo.wav';n=np.arange(190000,dtype=np.float32)
 sf.write(stereo,np.stack((.2*np.sin(n*.07),.3*np.cos(n*.09)),axis=1),44100,subtype='FLOAT')
 constant=tmp/'constant.wav';sf.write(constant,np.full(40,.25,np.float32),8000,subtype='FLOAT')
 for name,path in [('bundled',bundled),('stereo-44100',stereo),('constant-8000',constant)]:
  runs={}
  for executable in ['source','canonical']:
   args=[str(tmp/executable),str(path),str(tmp/(name+'-'+executable+'-'))]
   result=subprocess.run(args,capture_output=True)
   (OUT/(name+'-'+executable+'.stdout.log')).write_bytes(result.stdout)
   (OUT/(name+'-'+executable+'.stderr.log')).write_bytes(result.stderr)
   result.check_returncode();runs[executable]=dict(argv=args,rc=result.returncode,stdout=result.stdout.decode())
  geometry=[list(map(int,line.split())) for line in runs['canonical']['stdout'].splitlines()]
  chunks=list(read_chunks(path,Config()));assert len(geometry)==len(chunks)
  for chunk,row in zip(chunks,geometry):
   raw=np.fromfile(tmp/f'{name}-source-{chunk.index}.bin',np.float32)
   actual=np.fromfile(tmp/f'{name}-canonical-{chunk.index}.bin',np.float32)
   expected=np.zeros(30000,np.float32);a=raw.astype(np.float64)
   count=min(len(a),30000);expected[:count]=((a-a.mean())/np.sqrt(a.var()+1e-5))[:count]
   diff=float(np.max(np.abs(actual-expected)));assert diff<2e-6,(name,row,diff)
   assert row==[chunk.index,chunk.source_start,len(chunk.waveform),count],row
   python=prepare_chunk(chunk.waveform,chunk.sample_rate).tensor[0]
   records.append(dict(case=name,chunk=chunk.index,geometry=row,source_resample_then_corrected_normalization_maxdiff=diff,python_fourier_maxdiff=float(np.max(np.abs(actual-python))),prepared_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
  records.append(dict(case=name,runs=runs,audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 (OUT/'audio-results.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),extracted_source_sha256=hashlib.sha256(extracted.encode()).hexdigest(),extraction='unchanged source read/mix/resample; remove old normalization/padding only',builds=builds,records=records,board_or_sdk_executed=False),indent=2)+'\n')
 print('Actual native audio source comparisons passed:',sum('chunk' in r for r in records),'chunks')
