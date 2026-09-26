# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet source-parity and boundary tests; no board SDK."""
import dataclasses
import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def metadata(n=4, dtype='float32', quant=None):
    return dict(model_names=['pointnet'], model_name='pointnet',
                input_names=['point'], input_shapes={'point': [1, 3, n]},
                input_dtypes={'point': 'float32'}, output_names=['pred'],
                output_shapes={'pred': [1, n, 4]}, output_dtypes={'pred': dtype},
                output_quants={'pred': quant} if quant else {})


def bound(n=4, dtype='float32', quant=None):
    from samples.vision.pointnet.runtime.python.model_binding import bind_model, resolve_selection
    return bind_model(resolve_selection('s100'), metadata(n, dtype, quant))


class PointNetTests(unittest.TestCase):
    def task(self, raw=None, binding=None):
        from samples.vision.pointnet.runtime.python.pointnet import PointNetTask
        raw = np.eye(4, dtype=np.float32)[None] if raw is None else raw
        return PointNetTask(lambda tensors: raw, binding or bound())

    def points(self):
        return np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], np.float32)

    def test_source_cli_normalization_and_tensor_bytes(self):
        p = self.points()
        centered = p - np.mean(p, axis=0, keepdims=True)
        expected = centered / np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
        before = p.copy()
        prepared = self.task().pre_process(p)
        np.testing.assert_array_equal(prepared.tensors['point'], expected.T[None])
        np.testing.assert_array_equal(p, before)
        self.assertTrue(prepared.tensors['point'].flags.c_contiguous)

    def test_predict_equals_explicit_stages(self):
        task = self.task(); prepared = task.pre_process(self.points())
        explicit = task.post_process(task.forward(prepared.tensors))
        np.testing.assert_array_equal(task.predict(self.points()), explicit)
        np.testing.assert_array_equal(explicit, [0, 1, 2, 3])
        self.assertEqual(explicit.dtype, np.int32)

    def test_forward_keeps_raw_scores(self):
        raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4) - 8
        task = self.task(raw)
        np.testing.assert_array_equal(task.forward(task.pre_process(self.points()).tensors), raw)

    def test_context_is_per_call_and_frozen(self):
        task = self.task(); a = task.pre_process(self.points())
        b = task.pre_process(self.points() * 3 + 7)
        again = task.pre_process(self.points())
        self.assertEqual(a.context, again.context)
        self.assertNotEqual(a.context, b.context)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.context.radius = 42

    def test_rejects_invalid_or_wrong_count_points(self):
        for p in (np.zeros((4, 3)), np.ones((3, 3)), np.ones((4, 4)),
                  np.full((4, 3), np.nan), np.ones((4, 3), dtype=complex)):
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                self.task().pre_process(p)

    def test_post_rejects_shape_dtype_and_nonfinite(self):
        for raw in (np.zeros((2, 4, 4), np.float32), np.zeros((1, 5, 4), np.float32),
                    np.zeros((1, 4, 4), np.int8), np.full((1, 4, 4), np.nan, np.float32)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                self.task().post_process(raw)

    def test_integer_logits_dequantized_only_in_post(self):
        q = types.SimpleNamespace(quant_type=types.SimpleNamespace(name='SCALE'),
                                  scale=np.array([1, 3, 1, 1], np.float32),
                                  zero_point=np.array([7]), axis=2)
        raw = np.tile([2, 4, 0, 0], (1, 4, 1)).astype(np.int16)
        task = self.task(raw, bound(dtype='int16', quant=q))
        np.testing.assert_array_equal(task.forward(task.pre_process(self.points()).tensors), raw)
        np.testing.assert_array_equal(task.predict(self.points()), [0, 0, 0, 0])

    def test_raw_f32_ignores_vestigial_quantization(self):
        q = types.SimpleNamespace(quant_type='SCALE', scale=[99], zero_point=[3], axis=2)
        np.testing.assert_array_equal(self.task(binding=bound(quant=q)).predict(self.points()), [0, 1, 2, 3])


class BindingTests(unittest.TestCase):
    def test_exact_asset_and_default(self):
        from samples.vision.pointnet.runtime.python.model_binding import resolve_selection
        s = resolve_selection()
        self.assertEqual(s.target, 's100')
        self.assertEqual(s.asset.reference, 's:pointnet:s100/pointnet.hbm')
        self.assertTrue(str(s.model_path).endswith('/samples/vision/pointnet/model/s100/pointnet.hbm'))

    def test_targets_and_external_identity(self):
        from samples.vision.pointnet.runtime.python.model_binding import resolve_selection
        for target in ('x5', 's100p', 's600'):
            with self.assertRaises(ValueError): resolve_selection(target)
        with self.assertRaises(ValueError): resolve_selection('s100', model_path='/tmp/pointnet.hbm')
        with self.assertRaises(ValueError): resolve_selection('s100', asset_id='s:pointnet:s600/pointnet.hbm')

    def test_rejects_forged_selection(self):
        from samples.vision.pointnet.runtime.python.model_binding import resolve_selection, bind_model
        s = dataclasses.replace(resolve_selection(), model_path=Path('/tmp/forged.hbm'))
        with self.assertRaises(ValueError): bind_model(s, metadata())

    def test_bad_metadata(self):
        from samples.vision.pointnet.runtime.python.model_binding import bind_model, resolve_selection
        for field, value in [('input_shapes', {'point': (2, 3, 4)}),
                             ('input_shapes', {'point': (1, 4, 3)}),
                             ('output_shapes', {'pred': (1, 5, 4)}),
                             ('input_dtypes', {'point': 'int16'}),
                             ('output_dtypes', {'pred': 'int16'}),
                             ('output_names', ['pred', 'trans'])]:
            m = metadata(); m[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                bind_model(resolve_selection(), m)

    def test_invalid_integer_descriptor(self):
        for scale, zero, axis in [([], [], 2), ([1,2], [0], 2), ([1,2,3,4], [0], 7),
                                  ([1,2,3,4], [0,1], 2), ([float('nan')], [0], 2)]:
            q = types.SimpleNamespace(quant_type='SCALE', scale=scale, zero_point=zero, axis=axis)
            with self.subTest(scale=scale), self.assertRaises(ValueError): bound(dtype='int16', quant=q)


class RunnerAndCLITests(unittest.TestCase):
    def test_runtime_output_owned_and_input_checked(self):
        from samples.vision.pointnet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.pointnet.runtime.python.model_binding import resolve_selection
        raw = np.eye(4, dtype=np.float32)[None]
        class Fake:
            def __init__(self): self.__dict__.update(metadata()); self.calls = 0
            def run(self, inputs): self.calls += 1; return {'pointnet': {'pred': raw}}
        rt=Fake(); runner=RuntimeModelRunner(resolve_selection(), runtime=rt)
        result=runner({'point': np.ones((1,3,4), np.float32)})
        raw[:] = -1
        np.testing.assert_array_equal(result, np.eye(4, dtype=np.float32)[None])
        for v in (np.ones((1,3,5), np.float32), np.ones((1,3,4), np.float64)):
            with self.assertRaises(ValueError): runner({'point': v})
        self.assertEqual(rt.calls, 1)

    def test_board_identity_checked_before_sdk(self):
        from samples.vision.pointnet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.pointnet.runtime.python.model_binding import resolve_selection
        with patch('samples.vision.pointnet.runtime.python.model_runner.require_execution_target', side_effect=ValueError('mismatch')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                RuntimeModelRunner(resolve_selection()).load()

    def test_host_safe_entrypoints_and_target_rejection(self):
        main=ROOT/'samples/vision/pointnet/runtime/python/main.py'
        for args, code in [(['--help'],0),(['--list-models'],0),(['--dry-run'],0),
                           (['--dry-run','--target','s100p'],2)]:
            result=subprocess.run([sys.executable,str(main),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(result.returncode,code,result.stderr)
            self.assertNotIn('No module named',result.stderr)


class SourceAndEntrypointTests(unittest.TestCase):
    def source(self):
        name = 'pointnet_source_fixture'
        path = ROOT/'platforms/s/samples/vision/pointnet/runtime/python/pointnet.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: module, 'hbm_runtime': types.ModuleType('hbm_runtime')}):
            spec.loader.exec_module(module)
        return module

    def test_delivered_chair_matches_actual_source_loader_and_stages(self):
        from samples.vision.pointnet.runtime.python.pointnet import PointNetTask
        path = ROOT/'samples/vision/pointnet/test_data/chair.pts'
        source = self.source()
        normalized = source.PointNet.load_point_cloud(str(path))
        raw = np.arange(len(normalized)*4, dtype=np.float32).reshape(1,len(normalized),4)
        old = source.PointNet.__new__(source.PointNet)
        old.model_name='pointnet'; old.input_name='point'; old.output_name='pred'
        old.cfg=types.SimpleNamespace(num_parts=4)
        expected_tensor=old.pre_process(normalized)['pointnet']['point']
        task=PointNetTask(lambda x: raw, bound(len(normalized)))
        prepared=task.pre_process(np.loadtxt(path).astype(np.float32))
        np.testing.assert_array_equal(prepared.tensors['point'], expected_tensor)
        np.testing.assert_array_equal(task.post_process(raw), old.post_process({'pointnet':{'pred':raw}}))

    def test_readme_api_example_executes_with_real_runner_and_sdk_fixture(self):
        import re
        from samples.vision.pointnet.runtime.python import model_runner
        path=ROOT/'samples/vision/pointnet/test_data/chair.pts'
        n=len(np.loadtxt(path))
        class Fake:
            def __init__(self): self.__dict__.update(metadata(n))
            def run(self, inputs):
                self_shape=inputs['pointnet']['point'].shape
                assert self_shape == (1,3,n)
                return {'pointnet': {'pred':np.tile([0,1,2,3],(1,n,1)).astype(np.float32)}}
        real=model_runner.RuntimeModelRunner
        with patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda selection:real(selection,runtime=Fake())):
            for doc in ['README.md','README_cn.md']:
                text=(ROOT/'samples/vision/pointnet/runtime/python'/doc).read_text()
                code=re.findall(r'```python\n(.*?)```',text,re.S)[0]
                context={}
                exec(compile(code,doc,'exec'),context)
                np.testing.assert_array_equal(context['labels'], context['labels_again'])

    def test_cli_persists_labels_and_report_with_injected_runtime(self):
        import tempfile, json
        from samples.vision.pointnet.runtime.python.main import main
        from samples.vision.pointnet.runtime.python import model_runner
        n=len(np.loadtxt(ROOT/'samples/vision/pointnet/test_data/chair.pts'))
        class Fake:
            def __init__(self): self.__dict__.update(metadata(n))
            def run(self, inputs): return {'pointnet': {'pred':np.zeros((1,n,4),np.float32)}}
            def set_scheduling_params(self, **kwargs): pass
        real=model_runner.RuntimeModelRunner
        with tempfile.TemporaryDirectory() as tmp, patch.object(model_runner,'RuntimeModelRunner',side_effect=lambda s:real(s,runtime=Fake())):
            self.assertEqual(main(['--no-plot','--output-dir',tmp]),0)
            report=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(report['counts'],dict(back=n,seat=0,leg=0,arm=0))
            self.assertEqual(np.load(Path(tmp)/'labels.npy').shape,(n,))
            self.assertFalse((Path(tmp)/'result.png').exists())

    def test_download_uses_exact_manifest_and_separate_destination(self):
        from samples.vision.pointnet.model import download
        with patch.object(download,'download_asset',return_value='fixture-digest') as spy:
            self.assertEqual(download.main(['--target','s100','--output-dir','/tmp/pointnet-fixture']),0)
        asset,destination=spy.call_args.args
        self.assertEqual(asset.reference,'s:pointnet:s100/pointnet.hbm')
        self.assertEqual(destination,Path('/tmp/pointnet-fixture/s100/pointnet.hbm'))

    def test_inference_class_has_only_stage_methods(self):
        import ast
        tree=ast.parse((ROOT/'samples/vision/pointnet/runtime/python/pointnet.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PointNetTask')
        self.assertEqual({x.name for x in cls.body if isinstance(x,ast.FunctionDef)},
                         {'__init__','pre_process','forward','post_process','predict'})
