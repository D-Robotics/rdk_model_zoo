"""Pinned source audit, with host-only numerical counterexamples."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[5]
PIN='380e1a2bf42041af54be6f34935e50197cfadff9'
REL=Path('samples/vision/depth_anything_v2')
source=ROOT/'platforms/s'/REL
files=[]
for path in sorted(source.rglob('*')):
 if not path.is_file():continue
 rel=path.relative_to(ROOT/'platforms/s')
 original=subprocess.check_output(['git','show',f'{PIN}:{rel}'],cwd=ROOT)
 assert path.read_bytes()==original,path
 files.append({'path':str(rel),'sha256':hashlib.sha256(original).hexdigest()})
helper=ROOT/'platforms/s/utils/py_utils/nn_math.py'
node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='zscore_normalize_lastdim')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),namespace)
pixel=np.array([[[10,100,200]]],np.uint8)
actual=namespace[node.name](pixel)
imagenet=(pixel.astype(np.float32)/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
assert not np.allclose(actual,imagenet)
constant=np.ones((2,3),np.float32)
with np.errstate(invalid='ignore',divide='ignore'):
 old=(constant-constant.min())/(constant.max()-constant.min())*255
assert np.isnan(old).all()
# Letterbox padding must be cropped before original-size restoration.
raw=np.array([[0,0],[1,2],[3,4],[0,0]],np.float32)
old_stretch=cv2.resize(raw,(2,2),interpolation=cv2.INTER_LINEAR)
correct_crop=cv2.resize(raw[1:3],(2,2),interpolation=cv2.INTER_LINEAR)
assert not np.array_equal(old_stretch,correct_crop)
print(json.dumps({'source_pin':PIN,'files':files,'pixel_rgb':[10,100,200],
 'source_zscore':actual.tolist(),'imagenet':imagenet.tolist(),
 'constant_source_all_nan':bool(np.isnan(old).all()),
 'letterbox_source_stretch':old_stretch.tolist(),'letterbox_crop':correct_crop.tolist(),
 'artifact':'s:depth_anything_v2:s100/depth_any.hbm','board':'not-run'},indent=2))
