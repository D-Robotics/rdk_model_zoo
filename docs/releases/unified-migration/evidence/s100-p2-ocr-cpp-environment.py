import json,pathlib,shutil,subprocess
report={'tools':{name:shutil.which(name) for name in ('cmake','g++','pkg-config')}}
report['headers']={path:pathlib.Path(path).is_file() for path in ('/usr/include/gflags/gflags.h','/usr/include/polyclipping/clipper.hpp','/usr/include/hobot/dnn/hb_dnn.h','/usr/hobot/include/dnn/hb_dnn.h')}
for args in (['cmake','--version'],['g++','--version'],['pkg-config','--modversion','opencv4'],['pkg-config','--modversion','freetype2']):
 if shutil.which(args[0]):
  p=subprocess.run(args,capture_output=True,text=True);report[' '.join(args)]={'status':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(report),flush=True)
