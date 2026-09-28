from pathlib import Path
import json,re,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
CPP=ROOT/'samples/speech/asr/runtime/cpp';PREFIX=ROOT.parent/'.coordination/asr-audio-deps/install'
blocks=[re.findall(r'```cpp\n(.*?)```',(CPP/name).read_text(),re.S) for name in ['README.md','README_cn.md']]
assert blocks[0]==blocks[1] and len(blocks[0])==1
with tempfile.TemporaryDirectory() as tmp:
 source=Path(tmp)/'example.cc';source.write_text(blocks[0][0]);exe=Path(tmp)/'example'
 args=['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-I',str(CPP/'inc'),'-I',str(PREFIX/'include'),str(source),str(CPP/'src/frontend.cc'),str(PREFIX/'lib/libsamplerate.a'),'-o',str(exe)]
 build=subprocess.run(args,capture_output=True);(OUT/'cpp-example-build.log').write_bytes(build.stdout+build.stderr);build.check_returncode()
 run=subprocess.run([str(exe)],capture_output=True);(OUT/'cpp-example-run.log').write_bytes(run.stdout+run.stderr);run.check_returncode();assert run.stdout==b'AA\n'
 (OUT/'cpp-example.json').write_text(json.dumps(dict(argv=args,build_rc=build.returncode,run_rc=run.returncode,stdout=run.stdout.decode(),sdk_or_model_executed=False),indent=2)+'\n')
print('Bilingual complete C++ example compiled and ran with fixture transport')
