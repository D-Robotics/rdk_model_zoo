"""Verify YOLO local links and run both documented stage examples with a fake SDK."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import re
import sys
from urllib.parse import unquote

import numpy as np

ROOT=Path(__file__).resolve().parents[5]
SAMPLE=ROOT/'samples/vision/ultralytics_yolo'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(SAMPLE/'runtime/python'))
import yolo_detect
from samples.vision.ultralytics_yolo.runtime.python import yolo_seg
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner

links=[]
for page in [*SAMPLE.rglob('README*.md'), SAMPLE/'DETECTION_CONTRACT.md']:
    for href in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',page.read_text()):
        if href.startswith(('http:','https:','mailto:')):continue
        raw,_,anchor=href.partition('#')
        target=(page.parent/unquote(raw)).resolve() if raw else page
        assert target.exists(),(page,href)
        if anchor and target.suffix=='.md':
            content=target.read_text()
            slugs={re.sub(r'[^\w\-\s]','',h.lower()).strip().replace(' ','-')
                   for h in re.findall(r'^#+\s+(.+)$',content,re.M)}
            assert f'id="{anchor}"' in content or anchor in slugs,(page,href)
        links.append({'page':str(page.relative_to(ROOT)),'href':href})

class Runtime:
    def __init__(self):
        self.calls=0
        self.model_names=['m']
        self.input_names={'m':['y','uv']}
        self.input_shapes={'m':{'y':(1,64,64,1),'uv':(1,32,32,2)}}
        self.input_dtypes={'m':{'y':'U8','uv':'U8'}}
        self.arrays={}
        for stride in (8,16,32):
            cls=np.full((1,64//stride,64//stride,80),-20,np.float32)
            boxes=np.full((1,64//stride,64//stride,64),-24,np.int8)
            boxes[...,[1,17,33,49]]=16
            if stride==8:cls[0,3,3,7]=6
            self.arrays[f'cls_{stride}']=cls
            self.arrays[f'box_{stride}']=boxes
        self.output_names={'m':list(self.arrays)}
        self.output_shapes={'m':{n:a.shape for n,a in self.arrays.items()}}
        self.output_dtypes={'m':{n:a.dtype for n,a in self.arrays.items()}}
        self.output_quants={'m':{n:SimpleNamespace(quant_type='NONE' if n.startswith('cls') else 'SCALE',
                             scale=np.array([.25],np.float32),zero_point=np.array([0],np.int32),axis=3)
                            for n in self.arrays}}
    def run(self,tensors):
        assert tensors['m']['y'].shape==(1,64,64,1)
        assert tensors['m']['uv'].shape==(1,32,32,2)
        self.calls+=1
        return {'m':self.arrays}

runs=[]
examples=[]
for suffix in ('','_cn'):
    page=SAMPLE/'runtime/python'/f'README{suffix}.md'
    snippet=re.findall(r'```python\n(.*?)```',page.read_text(),re.S)
    assert len(snippet)==2
    examples.append(snippet[0])
    runtime=Runtime()
    sdk=SimpleNamespace(HB_HBMRuntime=lambda _:runtime)
    with patch.object(yolo_detect,'build_runner',side_effect=lambda config:build_runner(config,runtime_loader=lambda:sdk)),redirect_stdout(io.StringIO()) as out:
        context={}
        exec(compile(snippet[0],str(page),'exec'),context)
    assert runtime.calls==2
    assert context['raw']['box_8'] is runtime.arrays['box_8']
    assert context['prepared'].transform.original_size==context['bgr_image'].shape[:2]
    assert len(context['result'].scores)>0
    runs.append({'page':str(page.relative_to(ROOT)),'sdk_calls':runtime.calls,'stdout':out.getvalue(),
                 'results':len(context['result'].scores),'raw_box_dtype':str(context['raw']['box_8'].dtype)})
assert examples[0]==examples[1]
seg_examples=[]
for suffix in ('','_cn'):
    page=SAMPLE/'runtime/python'/f'README{suffix}.md'
    snippet=re.findall(r'```python\n(.*?)```',page.read_text(),re.S)[1]
    seg_examples.append(snippet)
    runtime=Runtime()
    for stride in (8,16,32):
        runtime.arrays[f'mces_{stride}']=np.ones((1,64//stride,64//stride,32),np.float32)
    runtime.arrays['protos']=np.ones((1,32,16,16),np.int8)
    runtime.output_names={'m':list(reversed(runtime.arrays))}
    runtime.output_shapes={'m':{n:a.shape for n,a in runtime.arrays.items()}}
    runtime.output_dtypes={'m':{n:a.dtype for n,a in runtime.arrays.items()}}
    runtime.output_quants['m']['protos']=SimpleNamespace(quant_type='SCALE',scale=np.full(32,.25,np.float32),zero_point=np.array([0],np.int32),axis=1)
    sdk=SimpleNamespace(HB_HBMRuntime=lambda _:runtime)
    with patch.object(yolo_seg,'build_runner',side_effect=lambda selection:build_runner(selection,runtime_loader=lambda:sdk)),redirect_stdout(io.StringIO()) as out:
        context={}
        exec(compile(snippet,str(page),'exec'),context)
    assert runtime.calls==2
    assert context['raw']['protos'] is runtime.arrays['protos']
    assert len(context['masks'])==1
    runs.append({'page':str(page.relative_to(ROOT)), 'task':'segmentation','sdk_calls':runtime.calls,'stdout':out.getvalue(),'masks':len(context['masks']),'raw_prototype_dtype':'int8','raw_prototype_shape':[1,32,16,16]})
assert seg_examples[0]==seg_examples[1]
print(json.dumps({'local_links':links,'api_fixture_runs':runs,'real_sdk':'not-run','board':'not-run'},indent=2))
