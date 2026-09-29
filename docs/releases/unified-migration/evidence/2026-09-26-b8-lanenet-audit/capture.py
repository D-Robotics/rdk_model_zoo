"""Source byte identities and LaneNet display counterexamples; host only."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[5]
PIN='380e1a2bf42041af54be6f34935e50197cfadff9'
REL=Path('samples/vision/lanenet');base=ROOT/'platforms/s'/REL
files=[]
for path in sorted(base.rglob('*')):
 if not path.is_file():continue
 rel=path.relative_to(ROOT/'platforms/s')
 original=subprocess.check_output(['git','show',f'{PIN}:{rel}'],cwd=ROOT)
 assert original==path.read_bytes(),path
 files.append({'path':str(rel),'sha256':hashlib.sha256(original).hexdigest()})
x=np.array([[-.1,.1,.5,1.,1.2]],np.float32)
python=(x*255).astype(np.uint8)
cpp=cv2.convertScaleAbs(np.clip(x,0,1)*255) # nonnegative: same saturating rounding
assert not np.array_equal(python,cpp)
binary=np.array([[0,1,2,-1]],np.int64)
py_binary=(binary*255).astype(np.uint8)
cpp_binary=np.where(binary!=0,255,0).astype(np.uint8)
assert not np.array_equal(py_binary,cpp_binary)
config=(base/'conversion/config.yaml').read_text()
assert "input_type_rt: 'featuremap'" in config and "lanenet256x512_nv12" in config
missing=['conversion/test.py','conversion/get_calibration_data.py','test_data/readme_img/result.jpg','conversion/README_cn.md']
assert all(not (base/p).exists() for p in missing)
print(json.dumps({'pin':PIN,'files':files,'embedding_values':x.tolist(),'source_python_display':python.tolist(),'source_cpp_display':cpp.tolist(),'binary_values':binary.tolist(),'source_python_binary':py_binary.tolist(),'source_cpp_binary':cpp_binary.tolist(),'missing_source_references':missing,'runtime_consumed_output_names':['instance_seg_logits','binary_seg_pred'],'source_conversion_claimed_output_count':3,'config_input':'NCHW featuremap 1x3x256x512','config_misleading_prefix':'lanenet256x512_nv12','board':'not-run'},indent=2))
