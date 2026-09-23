"""SigLIP host acceptance: real legacy numerics, packed submodels and identity."""
from pathlib import Path
import contextlib
import importlib.util
import io
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SAMPLE=ROOT/'samples/vision/siglip'
FACTS={
 'base-patch16-224':(224,768,196), 'base-patch16-384':(384,768,576),
 'base-patch16-512':(512,768,1024), 'large-patch16-256':(256,1024,256),
 'large-patch16-384':(384,1024,576), 'so400m-patch14-224':(224,1152,256),
 'so400m-patch14-384':(384,1152,729), 'so400m-patch16-256-i18n':(256,1152,256),
}
SUBMODELS=('pooler_output','last_hidden_state')


def meta(v='base-patch16-224',sub='pooler_output',dtype='F32'):
 size,d,n=FACTS[v]
 return dict(model_names=SUBMODELS,model_name=sub,input_names=['_input_0'],input_shapes={'_input_0':(1,3,size,size)},input_dtypes={'_input_0':'F32'},output_names=['_output_0'],output_shapes={'_output_0':(1,1,d) if sub=='pooler_output' else (1,n,d)},output_dtypes={'_output_0':dtype})


class FakeRuntime:
    def __init__(self,v='base-patch16-224',dtype='F32'):
        self.model_names=list(SUBMODELS);self.calls=[];self.scheduling={}
        for field in ('input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes'):
            setattr(self,field,{s:meta(v,s,dtype)[field] for s in SUBMODELS})
        npdtype={'F32':np.float32,'I16':np.int16}[dtype]
        self.raw={s:{'_output_0':np.ones(self.output_shapes[s]['_output_0'],dtype=npdtype)} for s in SUBMODELS}
    def run(self,inputs):
        self.calls.append(inputs)
        return {k:self.raw[k] for k in inputs}
    def set_scheduling_params(self,**kwargs):self.scheduling=kwargs


class BindingTests(unittest.TestCase):
    def test_exact_eight_assets_and_explicit_both_target_support(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection,list_available_assets
        for target in ('s100','s100p'):
            assets=list_available_assets(target)
            self.assertEqual(len(assets),8)
            for v,(size,_,_) in FACTS.items():
                sel=resolve_selection(target,variant=v)
                self.assertEqual(sel.asset.reference,f's:siglip:s100/bpu-siglip-{v}.hbm')
                self.assertEqual(sel.target,target)
                self.assertEqual(sel.image_size,size)
        self.assertEqual(resolve_selection('s100p').variant,'base-patch16-224')
        self.assertEqual(resolve_selection('auto',soc_name='s100',board_type='s100p').target,'s100p')

    def test_rejects_target_override_variant_and_wrong_image_size(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection
        for kwargs in ({'target':'x5'},{'target':'s600'},{'target':'s100','model_path':'x.hbm'},{'target':'s100','variant':'not-real'},{'target':'s100','image_size':384},{'target':'s100','submodel':'bogus'},{'target':'s100','asset_id':'s:vit:s100/vit_cifar10_batch1_int8.hbm'}):
            with self.assertRaises(ValueError):resolve_selection(**kwargs)
        with self.assertRaises(ValueError):resolve_selection('s100',variant='base-patch16-384',asset_id='s:siglip:s100/bpu-siglip-base-patch16-224.hbm')

    def test_bind_all_variants_submodels_and_preserve_native_output_shape(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection,bind_model
        for v in FACTS:
            for s in SUBMODELS:
                binding=bind_model(resolve_selection('s100p',variant=v,submodel=s),meta(v,s))
                self.assertEqual(binding.output_shape,meta(v,s)['output_shapes']['_output_0'])
        facts=meta();facts['output_shapes']['_output_0']=(1,768)
        self.assertEqual(bind_model(resolve_selection('s100'),facts).output_shape,(1,768))

    def test_bad_metadata_rejected_including_pooler_token_count(self):
        from samples.vision.siglip.runtime.python.model_binding import bind_model,resolve_selection
        bad=[]
        for key,value in [('model_names',('pooler_output',)),('model_name','last_hidden_state'),('input_names',['wrong']),('input_shapes',{'_input_0':(1,3,384,384)}),('input_dtypes',{'_input_0':'U8'}),('output_shapes',{'_output_0':(1,2,768)}),('output_shapes',{'_output_0':(1,1,1152)}),('output_dtypes',{'_output_0':'unknown'})]:
            m=meta();m[key]=value;bad.append(m)
        for m in bad:
            with self.assertRaises(ValueError):bind_model(resolve_selection('s100'),m)

    def test_default_download_and_s100p_use_same_exact_asset_without_network(self):
        from samples.vision.siglip.model import download
        for target in ('s100','s100p'):
            for v in (None,*FACTS):
                observed=[]
                def fetch(asset,path):observed.append((asset.reference,Path(path).relative_to(download.DEFAULT_OUTPUT_DIR).as_posix()));return 'a'*64
                args=['--target',target]+(['--variant',v] if v else [])
                with patch.object(download,'download_asset',fetch),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(download.main(args),0)
                file=f's100/bpu-siglip-{v or "base-patch16-224"}.hbm'
                self.assertEqual(observed,[(f's:siglip:{file}',file)])


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('_siglip_source',ROOT/'platforms/s/samples/vision/siglip/runtime/python/siglip.py')
        mod=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{spec.name:mod,'hbm_runtime':types.ModuleType('hbm_runtime')}):spec.loader.exec_module(mod)
        cls.source=mod

    def task(self,v='base-patch16-224',s='pooler_output',dtype='F32'):
        from samples.vision.siglip.runtime.python.model_binding import bind_model,resolve_selection
        from samples.vision.siglip.runtime.python.embedding import SigLIPTask
        runtime=FakeRuntime(v,dtype);binding=bind_model(resolve_selection('s100',variant=v,submodel=s),meta(v,s,dtype))
        return SigLIPTask(lambda inputs:runtime.raw[s],binding),runtime.raw[s]

    def test_preprocess_matches_source_all_sizes_and_preserves_per_call_context(self):
        rng=np.random.default_rng(3)
        for v,(size,_,_) in FACTS.items():
            task,_=self.task(v)
            legacy=self.source.SigLIP.__new__(self.source.SigLIP);legacy.cfg=self.source.SigLIPConfig('not-loaded',size)
            saved=[]
            for shape in ((31,59,3),(92,33,3),(31,59,3)):
                image=rng.integers(0,256,shape,dtype=np.uint8)
                got=task.pre_process(image)
                expected=legacy.pre_process(image)['pooler_output']['_input_0']
                np.testing.assert_array_equal(got.tensors['_input_0'],expected)
                self.assertEqual(got.context.original_shape,shape[:2]);saved.append(got)
            self.assertEqual(saved[0].context,saved[2].context)
            self.assertNotEqual(saved[0].context,saved[1].context)

    def test_raw_forward_postprocess_and_predict_all_submodels(self):
        for v in FACTS:
            for s in SUBMODELS:
                task,raw=self.task(v,s)
                image=np.zeros((32,81,3),dtype=np.uint8)
                self.assertIs(task.forward(task.pre_process(image).tensors),raw)
                actual=task.post_process(raw)
                np.testing.assert_array_equal(actual,raw['_output_0'])
                self.assertFalse(np.shares_memory(actual,raw['_output_0']))
                np.testing.assert_array_equal(task.predict(image),task.post_process(task.forward(task.pre_process(image).tensors)))

    def test_integer_outputs_remain_native_no_unrequested_softmax_or_dequant(self):
        task,raw=self.task(dtype='I16')
        raw['_output_0'][0,0,:3]=[10,-20,30]
        out=task.post_process(raw)
        self.assertEqual(out.dtype,np.int16)
        np.testing.assert_array_equal(out,raw['_output_0'])

    def test_postprocess_matches_original_without_dtype_or_shape_changes(self):
        for s in SUBMODELS:
            for dtype in ('F32','I16'):
                task,raw=self.task(s=s,dtype=dtype)
                legacy=self.source.SigLIP.__new__(self.source.SigLIP)
                legacy.cfg=self.source.SigLIPConfig('not-loaded',224,s)
                expected=legacy.post_process({s:raw})
                actual=task.post_process(raw)
                self.assertEqual(actual.dtype,expected.dtype)
                np.testing.assert_array_equal(actual,expected)

    def test_bad_inputs_output_nonfinite_shape_and_dtype_rejected(self):
        task,raw=self.task()
        for image in (np.zeros((0,10,3),np.uint8),np.zeros((10,10),np.uint8),np.zeros((10,10,3),np.float32)):
            with self.assertRaises(ValueError):task.pre_process(image)
        for bad in (np.full((1,1,768),np.nan,np.float32),np.zeros((1,768),np.float32),np.zeros((1,1,768),np.int16)):
            with self.assertRaises(ValueError):task.post_process({'_output_0':bad})


class ReadmeTests(unittest.TestCase):
    def test_runtime_api_examples_execute_with_real_binding_and_runner(self):
        import re
        from samples.vision.siglip.runtime.python import model_runner
        original=model_runner.RuntimeModelRunner
        for fn in ('README.md','README_cn.md'):
            text=(SAMPLE/'runtime/python'/fn).read_text()
            snippets=re.findall(r'```python\n(.*?)```',text,re.S)
            self.assertEqual(len(snippets),1,fn)
            runtime=FakeRuntime()
            def runner(selection):return original(selection,runtime=runtime)
            scope={}
            with patch.object(model_runner,'RuntimeModelRunner',runner),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0],fn,'exec'),scope)
            self.assertEqual(scope['composed_result'].shape,(1,1,768))
            self.assertEqual(len(runtime.calls),2)
            np.testing.assert_array_equal(scope['explicit_result'],scope['composed_result'])

    def test_native_readme_commands_parse_and_local_links_exist(self):
        import re,shlex
        from samples.vision.siglip.runtime.python.main import build_parser
        from samples.vision.siglip.model.download import build_parser as download_parser
        self.assertEqual(len(list(SAMPLE.rglob('README*.md'))),10)
        for p in SAMPLE.rglob('README*.md'):
            text=p.read_text().replace('\\\n',' ')
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' not in target:
                    self.assertTrue((p.parent/target.split('#')[0]).exists(),(p,target))
            for line in text.splitlines():
                if not line.startswith('python3 samples/'):continue
                args=shlex.split(line)
                if args[1].endswith('/runtime/python/main.py'):
                    options=build_parser().parse_args(args[2:])
                    self.assertTrue(Path(options.test_img).is_file())
                elif args[1].endswith('/model/download.py'):
                    download_parser().parse_args(args[2:])


class RunnerEntryTests(unittest.TestCase):
    def test_lazy_runner_explicit_submodel_metadata_and_raw_identity(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection
        from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
        for s in SUBMODELS:
            runtime=FakeRuntime();runner=RuntimeModelRunner(resolve_selection('s100p',submodel=s),runtime=runtime)
            self.assertFalse(runner.loaded)
            binding=runner.load();self.assertEqual(binding.model_name,s)
            runner.set_scheduling_params(priority=2,bpu_cores=[0])
            self.assertEqual(runtime.scheduling['priority'],dict.fromkeys(SUBMODELS,2))
            inputs={'_input_0':np.zeros((1,3,224,224),np.float32)}
            outputs=runner(inputs)
            self.assertIs(outputs['_output_0'],runtime.raw[s]['_output_0'])
            self.assertEqual(list(runtime.calls[-1]),[s])
            with self.assertRaises(ValueError):runner({'_input_0':np.zeros((1,3,384,384),np.float32)})

    def test_execution_gate_precedes_default_runtime_factory(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection
        from samples.vision.siglip.runtime.python import model_runner
        with patch('samples._shared.platforms.detect_target',return_value='s600'),patch.object(model_runner,'_default_runtime_factory') as factory:
            runner=model_runner.RuntimeModelRunner(resolve_selection('s100'))
            with self.assertRaises(ValueError):runner.load()
            factory.assert_not_called()

    def test_both_packed_models_validated_even_when_only_one_is_selected(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection
        from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
        r=FakeRuntime();r.input_shapes['last_hidden_state']['_input_0']=(1,3,512,512)
        runner=RuntimeModelRunner(resolve_selection('s100'),runtime=r)
        with self.assertRaises(ValueError):runner.load()
        self.assertFalse(runner.loaded)

    def test_cli_runs_actual_pipeline_and_saves_exact_path_native_tensor(self):
        import tempfile,json
        from samples.vision.siglip.runtime.python import main,model_runner
        real_runner=model_runner.RuntimeModelRunner
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model.hbm';model.write_bytes(b'host fixture only')
            output=Path(d)/'results/feature-no-extension'
            runtime=FakeRuntime('so400m-patch14-384')
            text=io.StringIO()
            with patch('samples._shared.platforms.detect_target',return_value='s100p'), patch.object(model_runner,'RuntimeModelRunner',lambda selection:real_runner(selection,runtime=runtime)),contextlib.redirect_stdout(text):
                rc=main.main(['--target','s100p','--asset-id','s:siglip:s100/bpu-siglip-so400m-patch14-384.hbm','--model-path',str(model),'--submodel','last_hidden_state','--output-file',str(output)])
            self.assertEqual(rc,0)
            result=np.load(output,allow_pickle=False)
            self.assertEqual(result.shape,(1,729,1152))
            np.testing.assert_array_equal(result,runtime.raw['last_hidden_state']['_output_0'])
            self.assertFalse(Path(str(output)+'.npy').exists())
            summary=json.loads(text.getvalue().split('Feature tensor saved:')[0])
            self.assertEqual(summary['submodel'],'last_hidden_state')
            self.assertEqual(summary['mean'],1.0)
            self.assertEqual(list(runtime.calls[0]),['last_hidden_state'])

    def test_cli_identity_and_image_size_errors_precede_inference(self):
        from samples.vision.siglip.runtime.python import main
        for argv,board in ((['--target','s100'],'s100p'),(['--target','s100','--image-size','384'],'s100'),(['--dry-run'],'s100')):
            with patch('samples._shared.platforms.detect_target',return_value=board),patch.object(main,'_run') as run,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(argv),2)
                run.assert_not_called()

    def test_runner_rejects_changed_output_metadata_and_invalid_input_values(self):
        from samples.vision.siglip.runtime.python.model_binding import resolve_selection
        from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
        runtime=FakeRuntime();runner=RuntimeModelRunner(resolve_selection('s100'),runtime=runtime)
        runner.load()
        for value in (np.full((1,3,224,224),2,np.float32),np.full((1,3,224,224),np.nan,np.float32)):
            with self.assertRaises(ValueError):runner({'_input_0':value})
        self.assertFalse(runtime.calls)
        for value in (np.zeros((1,768),np.float32),np.ones((1,1,768),np.int16)):
            runtime.raw['pooler_output']['_output_0']=value
            with self.assertRaises(ValueError):runner({'_input_0':np.zeros((1,3,224,224),np.float32)})

    def test_sdk_free_help_list_and_variant_dry_run_external_cwd(self):
        for args in (['--help'],['--list-models'],['--dry-run','--target','s100p','--variant','so400m-patch14-384','--submodel','last_hidden_state']):
            p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            if '--dry-run' in args:self.assertIn('729',p.stdout)
