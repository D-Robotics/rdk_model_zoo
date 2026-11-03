"""CLIP delivery keeps the CPU text graph and tokenizer in the model bundle."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from component_package import validate_components, validate_bundle_files


def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return json.dumps(value).encode()


class ClipPackageTests(unittest.TestCase):
    def setUp(self):
        self.files={'image.hbm':b'image fixture','text.onnx':b'text fixture',
            'bpe.txt.gz':b'BPE fixture','simple_tokenizer.py':b'tokenizer fixture','LICENSE.clip':b'MIT fixture'}
        self.io={
            'image':{'inputs':[{'name':'image','shape':[1,3,224,224],'dtype':'float32'}],
                     'outputs':[{'name':'image_features','shape':[1,512],'dtype':'float32'}]},
            'text':{'inputs':[{'name':'tokens','shape':['prompts',77],'dtype':'int32'}],
                    'outputs':[{'name':'text_features','shape':['prompts',512],'dtype':'float32'}]}}
        self.files['text-runtime.json']=encoded({'bpe_sha256':sha(self.files['bpe.txt.gz']),
            'tokenizer_sha256':sha(self.files['simple_tokenizer.py'])})
        components=[{'role':role,'filename':name,'platform':'s100','format':fmt,'execution':device,
            'url':'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s100/clip/'+name,
            'sha256':sha(self.files[name]),'size_bytes':len(self.files[name])}
            for role,name,fmt,device in [('image','image.hbm','HBM','BPU'),('text','text.onnx','ONNX','CPU')]]
        prov={r:{'onnx_sha256':('a'*64 if r=='image' else components[1]['sha256']),
                 'config_sha256':('b'*64 if r=='image' else sha(self.files['text-runtime.json']))}
              for r in ('image','text')}
        self.files['host-smoke.json']=encoded({'status':'passed','checkpoint':{'sha256':'c'*64},
            'graphs':{r:{'sha256':prov[r]['onnx_sha256']} for r in prov},
            'atol':1e-4,'rtol':1e-3,'tokenizer_matches_official':True,'bpe':{'sha256':sha(self.files['bpe.txt.gz'])},
            'cases':[{'name':n,'status':'passed','image_max_abs':0.0,'text_max_abs':0.0,'cosine_max_abs':0.0}
                     for n in ('first.jpg','second.jpg','third.jpg','random-seed0')]})
        conversion={};oe_components={}
        for c in components:
            role=c['role'];prefix='oe' if role=='image' else 'host'
            kind='native-bpu-compile' if role=='image' else 'cpu-onnx-host-check'
            self.files[role+'.html']=encoded('<!doctype html><html>'+role+'</html>')
            self.files[role+'.json']=encoded({'kind':kind,'target':{'platform':'s100'},'io':self.io[role],
                'provenance':dict(prov[role],artifact_sha256=c['sha256']),'board_measurement':None})
            conversion[role]=dict(self.io[role],status='passed',kind=kind,artifact_sha256=c['sha256'],**{
                prefix+'_report_filename':role+'.html',prefix+'_report_sha256':sha(self.files[role+'.html']),
                prefix+'_data_filename':role+'.json',prefix+'_data_sha256':sha(self.files[role+'.json'])})
            oe_components[role]=dict(prov[role],artifact_sha256=c['sha256'],**{
                prefix+'_report_sha256':sha(self.files[role+'.html']),prefix+'_data_sha256':sha(self.files[role+'.json'])})
        self.files['deployment.json']=encoded({'schema_version':1,'target':'s100','kind':'clip-image-text','components':components})
        members=[{'filename':n,'sha256':sha(d),'size_bytes':len(d)} for n,d in self.files.items() if n not in ('image.hbm','text.onnx')]
        self.row={'model':{'source':'clip'},'target':{'platform':'s100'},
            'artifact':{'filename':'clip.tar.gz','sha256':'f'*64,'components':components,'bundle':{
                'kind':'clip-image-text','descriptor':'deployment.json','members':members,'bpe_filename':'bpe.txt.gz',
                'tokenizer_filename':'simple_tokenizer.py','text_config_filename':'text-runtime.json'}},
            'provenance':{'source_checkpoint_sha256':'c'*64,'components':prov},
            'conversion':{'components':conversion,'outputs':self.io['image']['outputs']+self.io['text']['outputs']},
            'validation':{'host_pipeline':{'status':'passed','stages':['image','text'],
                'evidence_file':'host-smoke.json','evidence_sha256':sha(self.files['host-smoke.json']),
                'component_sha256':{c['role']:c['sha256'] for c in components}}}}
        self.oe={'provenance':{'artifact_sha256':'f'*64},'components':oe_components,
                 'aggregate':{'latency_ms':None},'board_measurement':None}
        self.html=('<!doctype html><script id="component-reports" type="application/json">'+json.dumps({
            r:base64.b64encode(self.files[r+'.html']).decode() for r in ('image','text')})+'</script>').encode()

    def archive(self,files=None):
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode='w:gz') as archive:
            for name,data in (self.files if files is None else files).items():
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        return output.getvalue()

    def test_complete_mixed_device_pair_is_accepted(self):
        validate_bundle_files(self.row,self.archive(),self.html,self.oe)

    def test_missing_or_corrupt_cpu_graph_and_bpe_are_rejected(self):
        for name in ('text.onnx','bpe.txt.gz','simple_tokenizer.py'):
            for mutation in ('missing','corrupt'):
                files=dict(self.files)
                if mutation=='missing':files.pop(name)
                else:files[name]+=b'corrupt'
                with self.subTest(name=name,mutation=mutation),self.assertRaises(ValueError):
                    validate_bundle_files(self.row,self.archive(files),self.html,self.oe)

    def test_cpu_graph_is_not_accepted_as_an_hbm_or_compiler_result(self):
        for field,value in (('format','HBM'),('execution','BPU')):
            row=copy.deepcopy(self.row);row['artifact']['components'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_components(row,platform='s100',source='clip')
        row=copy.deepcopy(self.row);row['conversion']['components']['text']['kind']='native-bpu-compile'
        with self.assertRaises(ValueError):validate_components(row,platform='s100',source='clip')

    def test_component_speed_does_not_become_full_matching_speed(self):
        for field in ('latency_ms','fps_derived'):
            oe=copy.deepcopy(self.oe);oe['aggregate'][field]=1.0
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_bundle_files(self.row,self.archive(),self.html,oe)
