# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PP-LiteSeg class-map protocol; host fixtures are not board observations."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import subprocess
import sys
import unittest
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[4]


def metadata():
    return dict(model_name='pp',model_names=['pp'],input_names=['images'],
                input_shapes={'images':(1,3,512,1024)},input_dtypes={'images':'NV12'},
                output_names=['labels'],output_shapes={'labels':(1,512,1024,1)},
                output_dtypes={'labels':'int32'},output_quants={})


def task(raw=None):
    from samples.vision.pp_liteseg.runtime.python.model_binding import resolve_selection,bind_model
    from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegTask
    binding=bind_model(resolve_selection(),metadata())
    raw=np.arange(512*1024,dtype=np.int32).reshape(1,512,1024,1)%19 if raw is None else raw
    return PPLiteSegTask(lambda tensors:raw,binding),raw


class StageTests(unittest.TestCase):
    def source(self):
        path=ROOT/'platforms/x5/samples/vision/pp_liteseg/runtime/python/pp_liteseg.py'
        spec=importlib.util.spec_from_file_location('pp_source',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'hbm_runtime':SimpleNamespace(HB_HBMRuntime=None)}):
            spec.loader.exec_module(module)
        obj=module.PPLiteSeg.__new__(module.PPLiteSeg)
        obj.config=module.PPLiteSegConfig('/not-used.bin')
        return obj

    def test_source_tensor_and_mask_parity(self):
        t,raw=task();old=self.source()
        image=cv2.imread(str(ROOT/'platforms/x5/samples/vision/pp_liteseg/test_data/street.png'))
        prepared=t.pre_process(image)
        np.testing.assert_array_equal(prepared.tensors['images'],old.pre_process(image))
        self.assertEqual(prepared.tensors['images'].shape,(768,1024))
        np.testing.assert_array_equal(t.post_process(raw),old.post_process(raw))

    def test_raw_class_map_never_argmaxed_or_dequantized(self):
        t,raw=task();prepared=t.pre_process(np.zeros((15,21,3),np.uint8))
        np.testing.assert_array_equal(t.forward(prepared.tensors),raw)
        labels=t.post_process(raw)
        self.assertEqual(labels.shape,(512,1024));self.assertEqual(labels.dtype,np.int32)
        np.testing.assert_array_equal(labels,raw[0,:,:,0])
        raw[:]=18
        self.assertEqual(labels[0,0],0)  # owned, not a view of mutable SDK memory

    def test_predict_and_context_are_independent(self):
        t,raw=task();a=np.zeros((11,17,3),np.uint8);b=np.zeros((21,13,3),np.uint8)
        pa=t.pre_process(a);pb=t.pre_process(b);again=t.pre_process(a)
        self.assertEqual(pa.context,again.context);self.assertNotEqual(pa.context,pb.context)
        np.testing.assert_array_equal(t.predict(a),t.post_process(t.forward(pa.tensors)))

    def test_reject_bad_images_and_class_maps(self):
        t,_=task()
        for image in (None,np.zeros((2,2),np.uint8),np.zeros((2,2,3),np.float32),np.empty((0,2,3),np.uint8)):
            with self.assertRaises(ValueError):t.pre_process(image)
        for raw in (np.zeros((1,512,1024,19),np.float32),np.zeros((1,512,1024,1),np.float32),
                    np.full((1,512,1024,1),19,np.int32),np.full((1,512,1024,1),-1,np.int32)):
            with self.assertRaises(ValueError):t.post_process(raw)

    def test_visualization_matches_source_three_panels(self):
        from samples.vision.pp_liteseg.runtime.python.visualization import render_result
        t,raw=task();image=np.full((57,91,3),93,np.uint8);labels=t.post_process(raw)
        np.testing.assert_array_equal(render_result(image,labels,alpha=0.55),self.source().visualize(image,labels))


class BindingAndCLITests(unittest.TestCase):
    def test_exact_x5_asset_and_no_s_fallback(self):
        from samples.vision.pp_liteseg.runtime.python.model_binding import resolve_selection,ASSET_ID
        self.assertEqual(resolve_selection().asset.reference,ASSET_ID)
        for target in ['s100','s100p','s600']:
            with self.assertRaises(ValueError):resolve_selection(target)
        with self.assertRaises(ValueError):resolve_selection(model_path='/tmp/wrong.bin')
        with self.assertRaises(ValueError):resolve_selection(asset_id='x5:wrong:file.bin')

    def test_binding_rejects_logits_or_wrong_geometry(self):
        from samples.vision.pp_liteseg.runtime.python.model_binding import bind_model,resolve_selection
        for key,value in [('output_shapes',{'labels':(1,19,512,1024)}),('output_dtypes',{'labels':'float32'}),
                          ('input_shapes',{'images':(1,3,512,512)}),('input_dtypes',{'images':'float32'})]:
            m=metadata();m[key]=value
            with self.assertRaises(ValueError):bind_model(resolve_selection(),m)

    def test_host_entrypoints_and_real_default_image(self):
        from samples.vision.pp_liteseg.runtime.python.main import build_parser
        args=build_parser().parse_args([]);self.assertTrue(Path(args.test_img).is_file())
        main=ROOT/'samples/vision/pp_liteseg/runtime/python/main.py'
        for args,rc in [(['--help'],0),(['--list-models'],0),(['--dry-run'],0),(['--dry-run','--target','s100'],2)]:
            result=subprocess.run([sys.executable,str(main),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(result.returncode,rc,result.stderr)


class ConversionTests(unittest.TestCase):
    def calibration(self,src,out):
        command=[sys.executable,str(ROOT/'samples/vision/pp_liteseg/conversion/prepare_calibration.py'),
                 '--src',str(src),'--out',str(out),'--num','50']
        return subprocess.run(command,capture_output=True,text=True)

    def test_calibration_preserves_colliding_basenames_and_refuses_overwrite(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);src=folder/'images'
            for name,color in [('a',23),('b',181)]:
                (src/name).mkdir(parents=True)
                cv2.imwrite(str(src/name/'same.png'),np.full((13,21,3),color,np.uint8))
            out=folder/'cal'
            result=self.calibration(src,out)
            self.assertEqual(result.returncode,0,result.stderr)
            files=sorted(out.glob('*.rgbchw'))
            self.assertEqual(len(files),2,'nested same.png inputs must not collapse')
            self.assertEqual({float(np.fromfile(f,dtype='<f4')[0]) for f in files},{23.,181.})
            self.assertEqual({f.stat().st_size for f in files},{6291456})
            manifest=json.loads((folder/'cal.manifest.json').read_text())
            self.assertEqual(len(manifest['samples']),2)
            before=[f.read_bytes() for f in files]
            self.assertNotEqual(self.calibration(src,out).returncode,0)
            self.assertEqual(before,[f.read_bytes() for f in files])

    def test_build_fails_when_tool_claims_success_without_bin(self):
        import tempfile,os
        with tempfile.TemporaryDirectory() as temp:
            cwd=Path(temp);(cwd/'onnx').mkdir();(cwd/'onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx').touch()
            (cwd/'calibration_data_rgb_f32_1024x512').mkdir()
            fake=cwd/'bin';fake.mkdir()
            for name in ['hb_mapper','hb_perf']:
                p=fake/name;p.write_text('#!/bin/sh\nexit 0\n');p.chmod(0o755)
            env={**os.environ,'PATH':str(fake)+os.pathsep+os.environ['PATH']}
            result=subprocess.run(['bash',str(ROOT/'samples/vision/pp_liteseg/conversion/build_bin.sh')],cwd=cwd,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,'no BIN means failed conversion regardless of tool exit 0')

    def test_export_requires_checkpoint_before_external_commands(self):
        import tempfile,os
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'PaddleSeg').mkdir();(folder/'bin').mkdir()
            calls=folder/'calls'
            for name in ['python','paddle2onnx']:
                p=folder/'bin'/name;p.write_text('#!/bin/sh\nprintf called >> "$CALL_LOG"\nexit 0\n');p.chmod(0o755)
            env={**os.environ,'PATH':str(folder/'bin')+os.pathsep+os.environ['PATH'],
                 'PADDLESEG_DIR':str(folder/'PaddleSeg'),'CHECKPOINT':'','CALL_LOG':str(calls)}
            result=subprocess.run(['bash',str(ROOT/'samples/vision/pp_liteseg/conversion/onnx_export/export_pp_liteseg_stdc1_onnx.sh')],cwd=folder,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(calls.exists(),'missing checkpoint must fail before pip/export execution')


class IntegrationTests(unittest.TestCase):
    def fake(self):
        runtime=SimpleNamespace(**metadata(),version='fixture-only')
        runtime.run=lambda tensors:{'pp':{'labels':np.full((1,512,1024,1),7,np.int32)}}
        return runtime

    def test_cli_writes_mask_render_and_report(self):
        import tempfile,json,contextlib,io
        from samples.vision.pp_liteseg.runtime.python import main,model_runner
        real=model_runner.RuntimeModelRunner
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda s:real(s,runtime=self.fake())), \
             contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp)
            self.assertEqual(main.main(['--output',str(p/'result.png'),'--mask-save-path',str(p/'mask.npy'),
                                        '--report-path',str(p/'report.json')]),0)
            labels=np.load(p/'mask.npy',allow_pickle=False)
            self.assertEqual(labels.dtype,np.int32);self.assertTrue(np.all(labels==7))
            self.assertEqual(cv2.imread(str(p/'result.png')).shape,(548,3078,3))
            report=json.loads((p/'report.json').read_text())
            self.assertEqual(report['runtime_version'],'fixture-only')
            self.assertEqual(report['class_names'],['traffic sign'])

    def test_evaluator_delegates_without_duplicated_inference(self):
        from samples.vision.pp_liteseg.evaluator import infer_board
        with patch.object(infer_board,'runtime_main',return_value=2) as run:
            self.assertEqual(infer_board.main(['--model','/tmp/a.bin','--image','/tmp/a.png','--output','/tmp/out.png']),2)
            argv=run.call_args.args[0]
            self.assertEqual(argv[argv.index('--mask-save-path')+1],'/tmp/out.labels.npy')
            self.assertEqual(argv[argv.index('--report-path')+1],'/tmp/out.report.json')

    def test_readme_stage_examples_with_real_runner(self):
        import re
        from samples.vision.pp_liteseg.runtime.python import model_runner
        real=model_runner.RuntimeModelRunner
        with patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda s:real(s,runtime=self.fake())):
            for filename in ['README.md','README_cn.md']:
                text=(ROOT/'samples/vision/pp_liteseg/runtime/python'/filename).read_text()
                code=re.findall(r'```python\n(.*?)```',text,re.S)[0]
                env={};exec(compile(code,filename,'exec'),env)
                np.testing.assert_array_equal(env['mask'],env['mask_again'])

    def test_wrong_board_rejected_before_sdk_load(self):
        from samples.vision.pp_liteseg.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.pp_liteseg.runtime.python.model_binding import resolve_selection
        with patch('samples.vision.pp_liteseg.runtime.python.model_runner.require_execution_target',side_effect=ValueError('wrong-board')):
            with self.assertRaisesRegex(ValueError,'wrong-board'):RuntimeModelRunner(resolve_selection()).load()

    def test_download_exact_manifest_unknown_hash(self):
        from samples.vision.pp_liteseg.model import download
        with patch.object(download,'download_asset',return_value='fixture') as call:
            self.assertEqual(download.main(['--output-dir','/tmp/pp-fixture']),0)
        asset,destination=call.call_args.args
        self.assertEqual(asset.reference,download.ASSET_ID)
        self.assertIsNone(asset.sha256)
        self.assertEqual(destination,Path('/tmp/pp-fixture')/asset.filename)


class CalibrationBoundaryTests(unittest.TestCase):
    def test_rgb_values_and_manifest_digests_match_actual_files(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);src=folder/'src';src.mkdir()
            bgr=np.zeros((3,5,3),np.uint8);bgr[:]=[7,41,129]
            image=src/'rgb.png';cv2.imwrite(str(image),bgr)
            out=folder/'cal'
            result=ConversionTests().calibration(src,out)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((folder/'cal.manifest.json').read_text());entry=manifest['samples'][0]
            raw=(out/entry['tensor']).read_bytes()
            actual=np.frombuffer(raw,dtype='<f4').reshape(1,3,512,1024)
            np.testing.assert_array_equal(actual[0,:,0,0],[129.,41.,7.])
            self.assertEqual(entry['source_sha256'],hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(entry['tensor_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(list(out.iterdir())),1)  # no JSON fed to hb_mapper

    def test_unreadable_selected_image_leaves_no_published_dataset(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);src=folder/'src';src.mkdir();(src/'bad.png').write_bytes(b'not an image')
            out=folder/'cal'
            result=ConversionTests().calibration(src,out)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(out.exists())
            self.assertFalse((folder/'cal.manifest.json').exists())
