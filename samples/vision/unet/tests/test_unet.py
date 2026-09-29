# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNet source geometry, stage semantics and metadata contract; host only."""
from pathlib import Path
import dataclasses
import subprocess
import sys
import types
import unittest
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[4]


def metadata(layout='nchw', dtype='float32', quant=None):
    shape = (1,21,512,512) if layout == 'nchw' else (1,512,512,21)
    return dict(model_name='unet', model_names=['unet'], input_names=['images'],
                input_shapes={'images': (1,3,512,512)}, input_dtypes={'images': 'NV12'},
                output_names=['logits'],output_shapes={'logits':shape},
                output_dtypes={'logits':dtype}, output_quants={'logits':quant} if quant else {})


def task(layout='nchw', dtype='float32', quant=None):
    from samples.vision.unet.runtime.python.model_binding import resolve_selection,bind_model
    from samples.vision.unet.runtime.python.unet import UNetTask
    binding=bind_model(resolve_selection('x5'),metadata(layout,dtype,quant))
    raw=np.zeros(binding.metadata.output_shapes['logits'],dtype=dtype)
    if layout=='nchw': raw[:,3]=5
    else: raw[:,:,:,3]=5
    return UNetTask(lambda x:raw,binding),raw


class UNetTests(unittest.TestCase):
    def test_packed_nv12_matches_independent_opencv_bytes(self):
        image=np.random.default_rng(32).integers(0,256,(17,25,3),dtype=np.uint8)
        resized=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
        planar=cv2.cvtColor(resized,cv2.COLOR_BGR2YUV_I420).reshape(-1)
        area=512*512
        expected=np.empty(area*3//2,np.uint8)
        expected[:area]=planar[:area]
        expected[area::2]=planar[area:area+area//4]
        expected[area+1::2]=planar[area+area//4:]
        t,_=task();prepared=t.pre_process(image)
        np.testing.assert_array_equal(prepared.tensors['images'].reshape(-1),expected)
        self.assertEqual(prepared.tensors['images'].shape,(1,768,512,1))

    def test_predict_explicit_stages_both_layouts(self):
        for layout in ['nchw','nhwc']:
            t,raw=task(layout); image=np.zeros((11,23,3),np.uint8)
            prepared=t.pre_process(image)
            np.testing.assert_array_equal(t.forward(prepared.tensors),raw)
            np.testing.assert_array_equal(t.predict(image),t.post_process(raw))
            self.assertEqual(t.predict(image).shape,(512,512))
            self.assertEqual(t.predict(image).dtype,np.uint8)
            self.assertTrue(np.all(t.predict(image)==3))

    def test_per_call_context_not_overwritten(self):
        t,_=task();a=t.pre_process(np.zeros((11,23,3),np.uint8))
        b=t.pre_process(np.zeros((29,13,3),np.uint8))
        c=t.pre_process(np.zeros((11,23,3),np.uint8))
        self.assertEqual(a.context,c.context);self.assertNotEqual(a.context,b.context)
        with self.assertRaises(dataclasses.FrozenInstanceError): a.context.original_height=99

    def test_integer_dequant_only_in_post(self):
        scales=np.ones(21,np.float32); scales[2]=10
        q=types.SimpleNamespace(quant_type='SCALE',scale=scales,zero_point=np.array([0]),axis=1)
        t,raw=task(dtype='int16',quant=q);raw[:,2]=2
        np.testing.assert_array_equal(t.forward(t.pre_process(np.zeros((4,6,3),np.uint8)).tensors),raw)
        self.assertTrue(np.all(t.post_process(raw)==2))

    def test_f32_vestigial_descriptor_is_not_applied(self):
        q=types.SimpleNamespace(quant_type='SCALE',scale=[-1],zero_point=[0],axis=1)
        t,raw=task(quant=q)
        self.assertTrue(np.all(t.post_process(raw)==3))

    def test_bad_inputs_and_outputs(self):
        t,_=task()
        for value in [None,np.empty((0,2,3),np.uint8),np.zeros((5,4,3),np.float32),np.zeros((5,4),np.uint8)]:
            with self.assertRaises(ValueError):t.pre_process(value)
        for value in [np.zeros((1,21,511,512),np.float32),np.zeros((1,21,512,512),np.int16),np.full((1,21,512,512),np.nan,np.float32)]:
            with self.assertRaises(ValueError):t.post_process(value)

    def test_all_five_assets_and_variant_mismatch(self):
        from samples.vision.unet.runtime.python.model_binding import resolve_selection,list_available_assets
        self.assertEqual(len(list_available_assets()),5)
        for variant in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            s=resolve_selection('x5',variant=variant)
            self.assertIn(variant,s.asset.filename);self.assertEqual(len(s.asset.sha256),64)
        asset=resolve_selection('x5',variant='resnet50').asset.reference
        self.assertEqual(resolve_selection('x5',asset_id=asset).variant,'resnet50')
        with self.assertRaises(ValueError):resolve_selection('x5',variant='resnet18',asset_id=asset)
        for target in ['s100','s100p','s600']:
            with self.assertRaises(ValueError):resolve_selection(target)

    def test_binding_rejects_wrong_geometry_and_dtype(self):
        from samples.vision.unet.runtime.python.model_binding import resolve_selection,bind_model
        for key,val in [('input_shapes',{'images':(1,3,256,256)}),('output_shapes',{'logits':(1,19,512,512)}),('output_dtypes',{'logits':'int16'}),('input_dtypes',{'images':'float32'})]:
            m=metadata();m[key]=val
            with self.assertRaises(ValueError):bind_model(resolve_selection(),m)

    def test_host_cli(self):
        path=ROOT/'samples/vision/unet/runtime/python/main.py'
        for args,rc in [(['--help'],0),(['--dry-run'],0),(['--list-models'],0),(['--dry-run','--target','s100'],2)]:
            p=subprocess.run([sys.executable,str(path),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(p.returncode,rc,p.stderr)


class EvaluatorTests(unittest.TestCase):
    def test_x5_evaluator_delegates_task_and_keeps_external_model_boundary(self):
        from unittest.mock import patch
        from samples.vision.unet.evaluator import eval_unet as ev
        from samples.vision.unet.runtime.python.model_runner import RuntimeModelRunner as Real
        class Fake:
            version='fixture'
            def __init__(self): self.__dict__.update(metadata())
            def run(self, inputs):
                assert inputs['unet']['images'].shape == (1,768,512,1)
                raw=np.zeros((1,21,512,512),np.float32);raw[:,7]=2
                return {'unet':{'logits':raw}}
        with patch.object(ev,'require_x5_runtime_environment',return_value='fixture'), \
             patch('samples._shared.platforms.require_execution_target'), \
             patch.dict(sys.modules,{'hbm_runtime':types.SimpleNamespace(HB_HBMRuntime=lambda path:Fake())}):
            run, info=ev.make_x5_runner(Path('/tmp/unet_resnet34_voc_512x512_nv12.bin'))
            pred=ev.prediction_from_output(run(np.zeros((512,512,3),np.uint8)))
            self.assertTrue(np.all(pred==7))
            self.assertEqual(info['artifact_source'],'caller-provided; publisher hash not asserted')

    def test_confusion_ignore_label_and_absent_class_average(self):
        from samples.vision.unet.evaluator.eval_unet import update_confusion,metrics_from_confusion
        matrix=np.zeros((21,21),np.int64)
        update_confusion(matrix,np.array([[0,1],[0,2]]),np.array([[0,1],[1,255]]))
        result=metrics_from_confusion(matrix)
        self.assertAlmostEqual(result['pixel_accuracy'],2/3)
        self.assertAlmostEqual(result['miou'],0.5)


class ConversionCLITests(unittest.TestCase):
    def test_exporter_help_does_not_need_torch(self):
        path=ROOT/'samples/vision/unet/conversion/onnx_export/export_unet.py'
        result=subprocess.run([sys.executable,str(path),'--help'],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class SourceAndDocsTests(unittest.TestCase):
    def test_source_preprocessing_and_postprocessing_match(self):
        import importlib.util
        from unittest.mock import patch
        pre_path=ROOT/'platforms/x5/utils/py_utils/preprocess.py'
        spec=importlib.util.spec_from_file_location('unet_source_pre',pre_path)
        pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)
        utils=types.ModuleType('utils');py=types.ModuleType('utils.py_utils')
        utils.py_utils=py;py.preprocess=pre
        source_path=ROOT/'platforms/x5/samples/vision/unet/runtime/python/unet.py'
        spec=importlib.util.spec_from_file_location('unet_source',source_path)
        source=importlib.util.module_from_spec(spec)
        original_path=list(sys.path)
        try:
            with patch.dict(sys.modules,{'unet_source':source,'utils':utils,'utils.py_utils':py,'utils.py_utils.preprocess':pre}):
                spec.loader.exec_module(source)
        finally:
            sys.path[:]=original_path
        old=source.UNet.__new__(source.UNet)
        old.config=source.UNetConfig();old.model_name='unet';old.input_name='images';old.output_name='logits'
        old.output_quant=types.SimpleNamespace(quant_type=types.SimpleNamespace(name='NONE'))
        t,raw=task()
        image=cv2.imread(str(ROOT/'samples/vision/unet/test_data/2007_000033.jpg'))
        np.testing.assert_array_equal(old.pre_process(image)['unet']['images'],t.pre_process(image).tensors['images'])
        np.testing.assert_array_equal(old.post_process({'unet':{'logits':raw}}),t.post_process(raw))

    def test_runtime_readme_examples_with_real_runner_fixture(self):
        import re
        from unittest.mock import patch
        from samples.vision.unet.runtime.python import model_runner
        class Fake:
            def __init__(self):self.__dict__.update(metadata())
            def run(self,inputs):
                raw=np.zeros((1,21,512,512),np.float32);raw[:,4]=1
                return {'unet':{'logits':raw}}
        real=model_runner.RuntimeModelRunner
        with patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda s:real(s,runtime=Fake())):
            for filename in ['README.md','README_cn.md']:
                text=(ROOT/'samples/vision/unet/runtime/python'/filename).read_text()
                code=re.findall(r'```python\n(.*?)```',text,re.S)[0]
                context={};exec(compile(code,filename,'exec'),context)
                np.testing.assert_array_equal(context['mask'],context['mask_again'])

    def test_runner_rejects_input_and_owns_output(self):
        from samples.vision.unet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.unet.runtime.python.model_binding import resolve_selection
        raw=np.zeros((1,21,512,512),np.float32)
        class Fake:
            def __init__(self):self.__dict__.update(metadata());self.calls=0
            def run(self,inputs):self.calls+=1;return {'unet':{'logits':raw}}
        fake=Fake();runner=RuntimeModelRunner(resolve_selection(),runtime=fake)
        out=runner({'images':np.zeros((1,768,512,1),np.uint8)})
        raw[:]=7
        self.assertTrue(np.all(out==0))
        with self.assertRaises(ValueError):runner({'images':np.zeros((1,3,512,512),np.float32)})
        self.assertEqual(fake.calls,1)

    def test_board_gate_precedes_sdk(self):
        from unittest.mock import patch
        from samples.vision.unet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.unet.runtime.python.model_binding import resolve_selection
        with patch('samples.vision.unet.runtime.python.model_runner.require_execution_target',side_effect=ValueError('wrong-board')):
            with self.assertRaisesRegex(ValueError,'wrong-board'):RuntimeModelRunner(resolve_selection()).load()


class ConversionPreparationTests(unittest.TestCase):
    def test_all_five_yaml_templates_keep_source_contract(self):
        from samples.vision.unet.conversion.mapper import BACKBONES,load_template,validate_template
        for backbone in BACKBONES:
            path, config=load_template(backbone)
            self.assertTrue(path.is_file())
            validate_template(backbone,config)

    def test_documented_calibration_recipe_passes_real_audit(self):
        import re,shutil,tempfile
        from samples.vision.unet.conversion.mapper import audit_calibration,SAMPLE_BYTES
        text=(ROOT/'samples/vision/unet/conversion/README.md').read_text()
        code=re.findall(r'```python\n(.*?)```',text,re.S)[0]
        with tempfile.TemporaryDirectory() as temp:
            images=Path(temp)/'images';images.mkdir()
            shutil.copyfile(ROOT/'samples/vision/unet/test_data/2007_000033.jpg',images/'demo.jpg')
            output=Path(temp)/'calibration'
            code=code.replace('/data/VOC2012/JPEGImages',str(images)).replace('/data/unet/calibration_data_rgb_f32_512',str(output))
            exec(compile(code,'calibration README','exec'),{})
            self.assertEqual((output/'demo.bin').stat().st_size,SAMPLE_BYTES)
            report=audit_calibration(output)
            self.assertTrue(report)
            bad=np.full(3*512*512,np.nan,dtype='<f4');bad.tofile(output/'bad.bin')
            with self.assertRaisesRegex(ValueError,'NaN'):audit_calibration(output)


class CLIResultTests(unittest.TestCase):
    def test_cli_preserves_runtime_version_and_writes_class_mask(self):
        import tempfile,json,contextlib,io
        from unittest.mock import patch
        from samples.vision.unet.runtime.python import main,model_runner
        class Fake:
            version='unet-sdk-fixture'
            def __init__(self):self.__dict__.update(metadata())
            def run(self,inputs):
                raw=np.zeros((1,21,512,512),np.float32);raw[:,6]=1
                return {'unet':{'logits':raw}}
        real=model_runner.RuntimeModelRunner
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda s:real(s,runtime=Fake())), \
             contextlib.redirect_stdout(io.StringIO()):
            folder=Path(temp)
            rc=main.main(['--mask-save-path',str(folder/'mask.png'),'--img-save-path',str(folder/'overlay.png'),'--report-path',str(folder/'report.json')])
            self.assertEqual(rc,0)
            report=json.loads((folder/'report.json').read_text())
            self.assertEqual(report['runtime_version'],'unet-sdk-fixture')
            mask=cv2.imread(str(folder/'mask.png'),cv2.IMREAD_UNCHANGED)
            self.assertEqual(mask.shape,(512,512));self.assertTrue(np.all(mask==6))


class DownloadTests(unittest.TestCase):
    def test_all_variants_use_published_hashes_and_exact_filenames(self):
        from unittest.mock import patch
        from samples.vision.unet.model import download
        with patch.object(download,'download_asset',return_value='fixture') as call:
            self.assertEqual(download.main(['--target','x5','--variant','all','--output-dir','/tmp/unet-fixture']),0)
        self.assertEqual(call.call_count,5)
        names=set()
        for args in call.call_args_list:
            asset,path=args.args
            self.assertEqual(path,Path('/tmp/unet-fixture')/asset.filename)
            self.assertEqual(len(asset.sha256),64)
            names.add(asset.filename)
        self.assertEqual(len(names),5)
