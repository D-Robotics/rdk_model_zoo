# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host protocol tests; synthetic metadata does not certify a deployed HBM."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import cv2
import numpy as np
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]


def metadata(dtype='int32', quant=None):
    if quant is None:
        quant = SimpleNamespace(quant_type=SimpleNamespace(name='NONE'))
    return dict(model_name='unet', model_names=['unet'],
                input_names=['y', 'uv'],
                input_shapes={'y': (1, 1024, 2048, 1), 'uv': (1, 512, 1024, 2)},
                input_dtypes={'y': 'uint8', 'uv': 'uint8'},
                output_names=['scores'], output_shapes={'scores': (1, 3, 5, 19)},
                output_dtypes={'scores': dtype}, output_quants={'scores': quant})


def make_task(raw=None, quant=None, dtype='int32'):
    from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
    from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetSegmenter, bind_model

    class StubRunner:
        """Host seam: callable transport with a prebuilt binding."""

        def __init__(self, binding, raw):
            self.binding, self.raw = binding, raw

        def load(self):
            return self.binding

        def __call__(self, tensors):
            return self.raw

    binding = bind_model(resolve_selection('s100'), metadata(dtype, quant))
    if raw is None:
        raw = np.zeros((1, 3, 5, 19), dtype=dtype)
        for y in range(3):
            for x in range(5):
                raw[0, y, x, (y*5+x)%19] = 7
    return UnetMobileNetSegmenter(resolve_selection('s100'),
                                  runner=StubRunner(binding, raw)), raw


class BindingTests(unittest.TestCase):
    def test_target_assets_and_paths_are_separate(self):
        from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
        for target in ['s100', 's600']:
            selection = resolve_selection(target)
            self.assertEqual(selection.asset.filename, f'{target}/unet_mobilenet_1024x2048_nv12.hbm')
            self.assertEqual(selection.model_path.parent.name, target)
        for target in ['s100p', 'x5']:
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with self.assertRaises(ValueError):
            resolve_selection('s600', asset_id='s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm')
        with self.assertRaises(ValueError):
            resolve_selection('s100', model_path='/tmp/unknown.hbm')

    def test_auto_uses_board_or_explicit_asset_never_s100_fallback(self):
        from samples.vision.unetmobilenet.runtime.python import cli as binding
        with patch.object(binding, 'resolve_target', return_value='s600'):
            self.assertEqual(binding.resolve_selection().target, 's600')
        with patch.object(binding, 'resolve_target', side_effect=ValueError('unknown-board')):
            with self.assertRaisesRegex(ValueError, 'unknown-board'):
                binding.resolve_selection()
            asset = 's:unetmobilenet:s600/unet_mobilenet_1024x2048_nv12.hbm'
            self.assertEqual(binding.resolve_selection(asset_id=asset).target, 's600')

    def test_invalid_split_layout_and_output_contract_rejected(self):
        from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
        from samples.vision.unetmobilenet.runtime.python.unetmobilenet import bind_model
        cases = [
            ('input_shapes', {'y': (1, 512, 1024, 1), 'uv': (1, 256, 512, 2)}),
            ('input_dtypes', {'y': 'float32', 'uv': 'uint8'}),
            ('output_shapes', {'scores': (1, 19, 3, 5)}),
            ('output_dtypes', {'scores': 'int16'}),
            ('output_quants', {}),
        ]
        for key, value in cases:
            m = metadata(); m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind_model(resolve_selection('s100'), m)


class StageTests(unittest.TestCase):
    def test_source_preprocessing_and_output_resize_parity(self):
        import importlib.util, sys, types
        modules = {}
        for name in ['preprocess', 'postprocess', 'visualize']:
            spec = importlib.util.spec_from_file_location('utils.py_utils.'+name, legacy_path(f's/utils/py_utils/{name}.py'))
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"hbm_runtime": SimpleNamespace(QuantParams=object)}):
                spec.loader.exec_module(module)
            modules['utils.py_utils.'+name] = module
        utils = types.ModuleType('utils'); py = types.ModuleType('utils.py_utils'); utils.py_utils = py
        for name in ['preprocess','postprocess','visualize']:
            setattr(py, name, modules['utils.py_utils.'+name])
        modules.update({'utils':utils, 'utils.py_utils':py, 'hbm_runtime':SimpleNamespace()})
        path = legacy_path('s/samples/vision/unetmobilenet/runtime/python/unetmobilenet.py')
        spec = importlib.util.spec_from_file_location('source_unetmobile', path)
        source = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(source)
        finally:
            sys.path[:] = original_path
        old = source.UnetMobileNet.__new__(source.UnetMobileNet)
        old.input_w = 2048; old.input_h = 1024; old.model_name = 'unet'
        old.input_names = ['y','uv']; old.output_names = ['scores']
        old.cfg = source.UnetMobileNetConfig('/not-used.hbm')
        task, raw = make_task()
        image = cv2.imread(str(legacy_path('s/samples/vision/unetmobilenet/test_data/segmentation.png')))
        image = cv2.resize(image, (37, 23), interpolation=cv2.INTER_AREA)
        prepared = task.pre_process(image)
        for name, array in old.pre_process(image)['unet'].items():
            np.testing.assert_array_equal(prepared.tensors[name], array)
        np.testing.assert_array_equal(task.post_process(raw, prepared.context),
                                      old.post_process({'unet':{'scores':raw}}, image.shape[1], image.shape[0]))

    def test_stage_context_is_per_call_and_predict_returns_mask(self):
        task, raw = make_task()
        a = np.zeros((7, 13, 3), np.uint8); b = np.zeros((11, 17, 3), np.uint8)
        pa = task.pre_process(a); pb = task.pre_process(b)
        first = task.post_process(task.forward(pa.tensors), pa.context)
        self.assertEqual(first.shape, (7, 13)); self.assertEqual(first.dtype, np.int32)
        self.assertEqual(task.post_process(raw, pb.context).shape, (11, 17))
        np.testing.assert_array_equal(first, task.predict(a))

    def test_channel_scales_change_argmax_without_mutating_raw(self):
        scales = [1., 100.] + [1.]*17
        quant = SimpleNamespace(quant_type=SimpleNamespace(name='SCALE'), axis=3,
                                scale=np.array(scales, np.float32), zero_point=np.array([0], np.int32))
        raw = np.zeros((1,3,5,19), np.int32); raw[:,:,:,0] = 10; raw[:,:,:,1] = 2
        task, _ = make_task(raw, quant)
        prepared = task.pre_process(np.zeros((7,13,3),np.uint8))
        before = raw.copy()
        self.assertTrue(np.all(task.post_process(raw, prepared.context) == 1))
        np.testing.assert_array_equal(raw, before)

    def test_large_raw_int32_order_and_lowest_id_ties(self):
        raw = np.zeros((1,3,5,19),np.int32)
        raw[:,:,:,0] = 2**25; raw[:,:,:,1] = 2**25+1
        task,_ = make_task(raw)
        context = task.pre_process(np.zeros((7,13,3),np.uint8)).context
        self.assertTrue(np.all(task.post_process(raw,context)==1))
        raw[:,:,:,0] = 2**25+1
        self.assertTrue(np.all(task.post_process(raw,context)==0))

    def test_invalid_images_and_logits_fail(self):
        task, raw = make_task()
        for image in [None,np.zeros((3,5),np.uint8),np.zeros((3,5,3),np.float32)]:
            with self.assertRaises(ValueError):task.pre_process(image)
        context=task.pre_process(np.zeros((3,5,3),np.uint8)).context
        for value in [raw.astype(np.float32),raw[:,:,:,0],np.zeros((1,3,5,20),np.int32)]:
            with self.assertRaises(ValueError):task.post_process(value,context)


class ReadableInterfaceTests(unittest.TestCase):
    """Canonical preprocess/infer/postprocess names drive predict."""
    def test_canonical_stages_exist_and_legacy_names_delegate(self):
        task, raw = make_task()
        image = np.zeros((7, 13, 3), np.uint8)
        for name in ['preprocess', 'infer', 'postprocess', 'predict']:
            self.assertTrue(callable(getattr(task, name, None)), name)
        pn = task.preprocess(image); po = task.pre_process(image)
        self.assertEqual(pn.context, po.context)
        for name in pn.tensors:
            np.testing.assert_array_equal(pn.tensors[name], po.tensors[name])
        np.testing.assert_array_equal(task.infer(pn.tensors), task.forward(po.tensors))
        np.testing.assert_array_equal(task.postprocess(raw, pn.context),
                                      task.post_process(raw, po.context))

    def test_predict_routes_through_canonical_stages_with_single_runner_call(self):
        task, raw = make_task(); routed = []; calls = []
        original_infer = task.infer
        def counting(tensors):
            calls.append(tensors); return original_infer(tensors)
        task.infer = counting
        for canonical in ['preprocess', 'postprocess']:
            attr = getattr(task, canonical)
            def spy(*args, _a=attr, _c=canonical, **kw):
                routed.append(_c); return _a(*args, **kw)
            setattr(task, canonical, spy)
        image = np.zeros((7, 13, 3), np.uint8)
        mask = task.predict(image)
        self.assertEqual(routed, ['preprocess', 'postprocess'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(mask.shape, (7, 13))

    def test_legacy_stage_names_route_through_the_same_canonical_stages(self):
        task, raw = make_task(); routed = []
        for canonical in ['preprocess', 'infer', 'postprocess']:
            attr = getattr(task, canonical)
            def spy(*args, _a=attr, _c=canonical, **kw):
                routed.append(_c); return _a(*args, **kw)
            setattr(task, canonical, spy)
        image = np.zeros((7, 13, 3), np.uint8)
        prepared = task.pre_process(image)
        task.post_process(task.forward(prepared.tensors), prepared.context)
        self.assertEqual(routed, ['preprocess', 'infer', 'postprocess'])

    def test_predict_equals_canonical_manual_composition(self):
        task, raw = make_task()
        for image in (np.zeros((7, 13, 3), np.uint8), np.zeros((11, 17, 3), np.uint8)):
            prepared = task.preprocess(image)
            manual = task.postprocess(task.infer(prepared.tensors), prepared.context)
            np.testing.assert_array_equal(task.predict(image), manual)


class RuntimeAndCLITests(unittest.TestCase):
    def fake(self):
        runtime = SimpleNamespace(**metadata(), version='host-fixture')
        raw = np.zeros((1,3,5,19),np.int32); raw[:,:,:,7] = 9
        runtime.run = lambda tensors: {'unet':{'scores':raw}}
        runtime.set_scheduling_params = lambda **kwargs: None
        return runtime

    def test_split_runner_and_cli_write_original_resolution(self):
        import contextlib,io,json,tempfile
        from samples.vision.unetmobilenet.runtime.python import main, unetmobilenet as task_module
        real = task_module.create_runner
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(task_module,'create_runner',side_effect=lambda s,**kw:real(s,runtime=self.fake())), \
             contextlib.redirect_stdout(io.StringIO()):
            folder=Path(temp)
            image=np.full((9,17,3),31,np.uint8);cv2.imwrite(str(folder/'input.png'),image)
            rc=main.main(['--target','s600','--test-img',str(folder/'input.png'),
                          '--img-save-path',str(folder/'out.png'),'--mask-save-path',str(folder/'mask.npy'),
                          '--report-path',str(folder/'report.json')])
            self.assertEqual(rc,0)
            labels=np.load(folder/'mask.npy',allow_pickle=False)
            self.assertEqual(labels.shape,(9,17));self.assertTrue(np.all(labels==7))
            self.assertEqual(cv2.imread(str(folder/'out.png')).shape,image.shape)
            report=json.loads((folder/'report.json').read_text())
            self.assertEqual(report['runtime_version'],'host-fixture')
            self.assertEqual(report['target'],'s600')

    def test_alpha_weights_original_and_source_palette_is_preserved(self):
        import importlib.util
        from samples.vision.unetmobilenet.runtime.python.cli import render_overlay,PALETTE_BGR
        spec=importlib.util.spec_from_file_location('source_vis',legacy_path('s/utils/py_utils/visualize.py'))
        source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        np.testing.assert_array_equal(np.asarray(PALETTE_BGR, np.uint8),np.asarray(source.rdk_colors,np.uint8))
        image=np.full((3,7,3),83,np.uint8);labels=np.full((3,7),5,np.int32)
        np.testing.assert_array_equal(render_overlay(image,labels,alpha_f=1),image)
        np.testing.assert_array_equal(render_overlay(image,labels,alpha_f=0),np.asarray(PALETTE_BGR, np.uint8)[labels])
        np.testing.assert_array_equal(render_overlay(image,labels),cv2.addWeighted(image,.75,np.asarray(PALETTE_BGR, np.uint8)[labels],.25,0))

    def test_host_inspection_and_s100p_rejection(self):
        import subprocess,sys
        path=ROOT/'samples/vision/unetmobilenet/runtime/python/main.py'
        for args,rc in [(['--help'],0),(['--list-models'],0),(['--dry-run','--target','s100'],0),
                        (['--dry-run','--target','s600'],0),(['--dry-run','--target','s100p'],2)]:
            result=subprocess.run([sys.executable,str(path),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(result.returncode,rc,result.stderr)

    def test_real_loading_checks_identity_before_sdk(self):
        from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
        from samples.vision.unetmobilenet.runtime.python import unetmobilenet as task_module
        with patch.object(task_module,'require_execution_target',side_effect=ValueError('wrong-board')):
            with self.assertRaisesRegex(ValueError,'wrong-board'):task_module.create_runner(resolve_selection('s100')).load()

    def test_each_download_keeps_target_subdirectory(self):
        from samples.vision.unetmobilenet.model import download
        for target in ['s100','s600']:
            with patch.object(download,'download_asset',return_value='fixture') as call:
                self.assertEqual(download.main(['--target',target,'--output-dir','/tmp/model']),0)
                asset,path=call.call_args.args
                self.assertEqual(path.parent,Path('/tmp/model')/target)
                self.assertIsNone(asset.sha256)


class NativeBoundaryTests(unittest.TestCase):
    def test_native_launcher_identity_precedes_build(self):
        import contextlib,io
        from samples.vision.unetmobilenet.runtime.cpp import launcher
        with patch.object(launcher,'require_execution_target',side_effect=ValueError('S100P refused')), \
             patch.object(launcher.subprocess,'run') as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main(['--target','s100','--build']),2)
            run.assert_not_called()

    def test_native_host_modes_and_unsupported_targets(self):
        import subprocess,sys
        path=ROOT/'samples/vision/unetmobilenet/runtime/cpp/launcher.py'
        for args,rc in [(['--help'],0),(['--list-models'],0),(['--dry-run','--target','s100'],0),
                        (['--dry-run','--target','s600'],0),(['--dry-run','--target','s100p'],2)]:
            result=subprocess.run([sys.executable,str(path),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(result.returncode,rc,result.stderr)

    def test_native_pure_decoder_compiles_and_runs(self):
        import subprocess,tempfile
        cpp=ROOT/'samples/vision/unetmobilenet/runtime/cpp'
        with tempfile.TemporaryDirectory() as temp:
            binary=Path(temp)/'test'
            build=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(cpp/'inc'),
                                  str(cpp/'tests/test_tensor_contract.cpp'),str(cpp/'src/tensor_contract.cpp'),
                                  '-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


class DocumentationAndResourceTests(unittest.TestCase):
    def test_runtime_readme_examples_use_real_runner_with_fixture(self):
        import re
        from samples.vision.unetmobilenet.runtime.python import unetmobilenet as task_module
        real=task_module.create_runner
        with patch.object(task_module,'create_runner',side_effect=lambda s,**kw:real(s,runtime=RuntimeAndCLITests().fake())):
            for name in ['README.md','README_cn.md']:
                text=(ROOT/'samples/vision/unetmobilenet/runtime/python'/name).read_text()
                snippet=re.findall(r'```python\n(.*?)```',text,re.S)[0]
                scope={};exec(compile(snippet,name,'exec'),scope)
                np.testing.assert_array_equal(scope['mask'],scope['mask_again'])
                self.assertEqual(scope['overlay'].shape,scope['image'].shape)

    def test_native_resource_cleanup_with_fake_interfaces(self):
        import subprocess,tempfile
        cpp=ROOT/'samples/vision/unetmobilenet/runtime/cpp'
        with tempfile.TemporaryDirectory() as temp:
            binary=Path(temp)/'resources'
            build=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',
                '-I',str(cpp/'tests/fixtures'),'-I',str(cpp/'inc'),str(cpp/'src/model_runner.cpp'),
                str(cpp/'src/tensor_contract.cpp'),str(cpp/'tests/test_resources.cpp'),'-o',str(binary)],
                capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_missing_integer_zero_point_field_is_rejected_during_binding(self):
        from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
        from samples.vision.unetmobilenet.runtime.python.unetmobilenet import bind_model
        quant=SimpleNamespace(quant_type=SimpleNamespace(name='SCALE'),scale=np.array([1.]),axis=3)
        with self.assertRaisesRegex(ValueError,'zero_point'):
            bind_model(resolve_selection('s100'),metadata(quant=quant))


class SimplifiedRuntimeTests(unittest.TestCase):
    """2026-10-08 runtime simplification boundary.

    Selection/catalog duties live in ``cli.py``; ``unetmobilenet.py`` owns the
    named model class that loads via the shared transport; the per-sample
    ``model_binding``/``model_runner`` forwarding modules are gone.
    """

    def _model(self):
        from samples.vision.unetmobilenet.runtime.python.cli import resolve_selection
        from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetSegmenter, create_runner
        runtime = SimpleNamespace(**metadata(), version='host-fixture')
        raw = np.zeros((1, 3, 5, 19), np.int32)
        raw[:, :, :, 4] = 9
        runtime.run = lambda tensors: {'unet': {'scores': raw}}
        selection = resolve_selection('s100')
        return UnetMobileNetSegmenter(
            selection, runner=create_runner(selection, runtime=runtime))

    def test_segmenter_constructs_from_selection_and_runs_stages(self):
        model = self._model()
        mask = model.predict(np.zeros((7, 13, 3), np.uint8))
        self.assertEqual(mask.shape, (7, 13))
        self.assertEqual(mask.dtype, np.int32)
        self.assertTrue(np.all(mask == 4))
        prepared = model.preprocess(np.zeros((7, 13, 3), np.uint8))
        manual = model.postprocess(model.infer(prepared.tensors), prepared.context)
        np.testing.assert_array_equal(mask, manual)

    def test_set_scheduling_params_reaches_runner(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model.set_scheduling_params(bpu_cores=[])

    def test_split_forwarding_modules_are_removed(self):
        base = ROOT / 'samples/vision/unetmobilenet/runtime/python'
        for name in ('model_binding.py', 'model_runner.py'):
            self.assertFalse((base / name).exists(), name)
