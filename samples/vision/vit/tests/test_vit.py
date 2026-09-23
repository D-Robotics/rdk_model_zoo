"""ViT source-to-unified host regressions, without board SDK or downloads."""
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

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / 'samples/vision/vit'
VARIANTS = ('int8', 'int16')


def metadata(classes=10, dtype='F32', size=224):
    return dict(model_name='fixture', input_names=['input_y','input_uv'],
                input_shapes={'input_y':(1,size,size,1),'input_uv':(1,size//2,size//2,2)},
                input_dtypes={'input_y':'U8','input_uv':'U8'},output_names=['output'],
                output_shapes={'output':(1,classes)},output_dtypes={'output':dtype})


class EntryTests(unittest.TestCase):
    def test_sdk_free_entry_and_both_variants_from_external_cwd(self):
        for args in (['--help'],['--list-models'], *(['--dry-run','--target','s100','--variant',v] for v in VARIANTS)):
            p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            if '--dry-run' in args:
                self.assertIn('squeeze -> (10,)',p.stdout)
                self.assertIn('variant: '+args[-1],p.stdout)

    def test_published_set_defaults_and_input_contract(self):
        from samples.vision.vit.runtime.python.model_binding import list_available_assets,resolve_selection
        self.assertEqual({a.asset_id for a in list_available_assets()},
            {f's:vit:s100/vit_cifar10_batch1_{v}.hbm' for v in VARIANTS})
        self.assertEqual(resolve_selection('s100').variant,'int8')
        for v in VARIANTS:
            c=resolve_selection('s100',variant=v).contract
            self.assertEqual((c.class_count,c.input_height,c.input_width,c.resize_type),(10,224,224,0))
            self.assertEqual(c.resize_interpolation,'nearest')
            self.assertEqual(c.output_score_policy,'softmax')

    def test_unpublished_targets_identity_and_override_rejected(self):
        from samples.vision.vit.runtime.python.model_binding import resolve_selection,BindingError
        for t in ('x5','s100p','s600'):
            with self.assertRaises(BindingError):resolve_selection(t)
        with self.assertRaises(BindingError):resolve_selection('auto',soc_name='s100',board_type='s100p')
        with self.assertRaises(BindingError):resolve_selection('s100',model_path='/tmp/arbitrary.hbm')
        with self.assertRaises(BindingError):resolve_selection('s100',variant='int16',asset_id='s:vit:s100/vit_cifar10_batch1_int8.hbm')

    def test_execution_target_mismatch_stops_before_runtime(self):
        from samples.vision.vit.runtime.python import main
        with patch('samples._shared.platforms.detect_target',return_value='s100p'), patch.object(Path,'is_file',return_value=True), patch.object(main,'_run') as execute, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(['--target','s100']),2)
            execute.assert_not_called()

    def test_download_defaults_and_each_variant_use_exact_manifest(self):
        from samples.vision.vit.model import download
        for v in (None,*VARIANTS):
            seen=[]
            def fetch(asset,path):
                seen.append((asset.reference,Path(path).relative_to(download.DEFAULT_OUTPUT_DIR).as_posix()))
                return '0'*64
            with patch.object(download,'download_asset',fetch),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download.main([] if v is None else ['--variant',v]),0)
            file=f's100/vit_cifar10_batch1_{v or "int8"}.hbm'
            self.assertEqual(seen,[(f's:vit:{file}',file)])

    def test_parser_keeps_source_alias_and_bundled_cifar_labels(self):
        from samples.vision.vit.runtime.python.main import build_parser
        p=build_parser();a=p.parse_args(['--model-variant','int16'])
        self.assertEqual(a.variant,'int16')
        self.assertEqual(Path(a.test_img).name,'airplane_0000.png')
        from samples.vision.vit.runtime.python.labels import load_labels
        self.assertEqual(list(load_labels(Path(a.label_file)).values()),['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('_vit_legacy',ROOT/'platforms/s/samples/vision/vit/runtime/python/vit.py')
        mod=importlib.util.module_from_spec(spec)
        utils=types.ModuleType('utils');utils.__path__=[str(ROOT/'platforms/s/utils')]
        # Import only the two real helpers used by ViT, not package-wide SDK exports.
        pyutils=types.ModuleType('utils.py_utils');pyutils.__path__=[str(ROOT/'platforms/s/utils/py_utils')]
        oldpath=sys.path[:]
        with patch.dict(sys.modules,{'utils':utils,'utils.py_utils':pyutils,'hbm_runtime':types.ModuleType('hbm_runtime'),spec.name:mod}):
            try:spec.loader.exec_module(mod)
            finally:sys.path[:]=oldpath
        cls.old=mod.ViT.__new__(mod.ViT);cls.old.cfg=mod.ViTConfig('not-loaded')
        cls.old.model_name='fixture';cls.old.input_names=['input_y','input_uv'];cls.old.output_names=['output']
        cls.old.input_h=224;cls.old.input_w=224

    def task(self,variant='int8'):
        from samples.vision.vit.runtime.python.model_binding import bind_model,resolve_selection
        from samples.vision.vit.runtime.python.classification import ClassificationTask
        binding=bind_model(resolve_selection('s100',variant=variant),metadata())
        raw={'output':np.array([[-2.1,0.3,2.2,-1.9,5.1,3.1,1.5,-0.2,0.1,0.8]],dtype=np.float32)}
        return ClassificationTask(lambda _:raw,binding),raw

    def test_source_preprocess_planes_match_for_modes_shapes_and_variants(self):
        rng=np.random.default_rng(8)
        for v in VARIANTS:
            task,_=self.task(v)
            for shape in ((32,32,3),(111,219,3),(217,105,3)):
                image=rng.integers(0,256,shape,dtype=np.uint8)
                for mode in (0,1):
                    task.resize_type=mode
                    got=task.pre_process(image).tensors
                    expected=self.old.pre_process(image,mode)['fixture']
                    for key in expected:np.testing.assert_array_equal(got[key],expected[key])

    def test_raw_forward_and_postprocess_preserve_source_scores(self):
        for v in VARIANTS:
            task,raw=self.task(v)
            self.assertIs(task.forward({}),raw)
            expected=self.old.post_process({'fixture':raw})
            got=task.post_process(raw)
            np.testing.assert_array_equal(got.class_ids,[i for i,_ in expected])
            np.testing.assert_allclose(got.scores,[s for _,s in expected],rtol=0,atol=1e-7)

    def test_real_binding_predict_equals_stages_and_contexts_do_not_leak(self):
        for v in VARIANTS:
            task,_=self.task(v);task.resize_type=1
            a=np.full((40,80,3),127,np.uint8);b=np.full((91,37,3),64,np.uint8)
            first=task.pre_process(a);snapshot={k:x.copy() for k,x in first.tensors.items()}
            transform=first.transform;task.pre_process(b);last=task.pre_process(a)
            self.assertEqual(first.transform,transform);self.assertEqual(last.transform,transform)
            for key in snapshot:np.testing.assert_array_equal(first.tensors[key],snapshot[key])
            staged=task.post_process(task.forward(first));direct=task.predict(a)
            np.testing.assert_array_equal(direct.class_ids,staged.class_ids)
            np.testing.assert_array_equal(direct.scores,staged.scores)

    def test_rejects_imagenet_geometry_quantized_raw_and_invalid_topk(self):
        from samples.vision.vit.runtime.python.model_binding import bind_model,resolve_selection,MetadataMismatchError
        from samples.vision.vit.runtime.python.classification import ClassificationTask
        for bad in (metadata(classes=1000),metadata(size=256),metadata(dtype='I8')):
            with self.assertRaises(MetadataMismatchError):bind_model(resolve_selection('s100'),bad)
        task,_=self.task()
        for k in (0,11):
            with self.assertRaises(ValueError):ClassificationTask(task.runner,task.binding,top_k=k)

    def test_readme_integration_executes_with_real_binding_and_host_runner(self):
        import re
        from samples.vision.vit.runtime.python import model_runner
        task,raw=self.task()
        class HostRunner:
            def __init__(self,selection):
                if selection.variant != 'int8':raise AssertionError(selection)
            def load(self):return task.binding
            def set_scheduling_params(self,**kwargs):
                if kwargs != {'priority':0,'bpu_cores':[0]}:raise AssertionError(kwargs)
            def __call__(self,tensors):
                if tensors['input_y'].shape != (1,224,224,1):raise AssertionError(tensors)
                return raw
        for fn in ('README.md','README_cn.md'):
            text=(SAMPLE/'runtime/python'/fn).read_text()
            snippets=re.findall(r'```python\n(.*?)```',text,re.S)
            self.assertEqual(len(snippets),1)
            scope={}
            with patch.object(model_runner,'RuntimeModelRunner',HostRunner),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippets[0],str(SAMPLE/'runtime/python'/fn),'exec'),scope)
            self.assertEqual(scope['result'].labels[0],'deer')
            self.assertEqual(scope['result'].class_ids.tolist(),[4,5,2,6,9])

    def test_readme_commands_parse_and_local_links_resolve(self):
        import re,shlex
        from samples.vision.vit.runtime.python.main import build_parser
        from samples.vision.vit.model.download import build_parser as download_parser
        for p in SAMPLE.rglob('README*.md'):
            text=p.read_text().replace('\\\n',' ')
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' not in target:self.assertTrue((p.parent/target.split('#')[0]).exists(),(p,target))
            for line in text.splitlines():
                if not line.startswith('python3 samples/'):continue
                args=shlex.split(line)
                if args[1].endswith('/runtime/python/main.py'):
                    options=build_parser().parse_args(args[2:])
                    self.assertTrue(Path(options.test_img).is_file())
                elif args[1].endswith('/model/download.py'):download_parser().parse_args(args[2:])

    def test_preserves_conversion_log_recipe_and_all_test_resources(self):
        for folder in ('conversion','test_data'):
            source=ROOT/'platforms/s/samples/vision/vit'/folder
            for f in source.rglob('*'):
                if f.is_file() and not f.name.startswith('README'):
                    self.assertEqual((SAMPLE/folder/f.relative_to(source)).read_bytes(),f.read_bytes())
