"""RepGhost migration acceptance on a host; no inference on hardware."""
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
SAMPLE = ROOT / 'samples/vision/repghost'
VARIANTS = ('100', '111', '130', '150', '200')


class EntryTests(unittest.TestCase):
    def test_entry_is_sdk_free_from_unrelated_directory(self):
        for args in (['--help'], ['--list-models'], ['--dry-run', '--target', 'x5']):
            run = subprocess.run([sys.executable, str(SAMPLE/'runtime/python/main.py'), *args], cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            if '--dry-run' in args:
                self.assertIn('variant: 100', run.stdout)

    def test_default_and_all_published_variants(self):
        from samples.vision.repghost.runtime.python.model_binding import resolve_selection, list_available_assets
        self.assertEqual(resolve_selection('x5').variant, '100')
        self.assertEqual({r.filename for r in list_available_assets()}, {f'RepGhost_{v}_224x224_nv12.bin' for v in VARIANTS})
        for v in VARIANTS:
            selected = resolve_selection('x5', variant=v)
            self.assertEqual(selected.asset_id, f'x5:repghost:RepGhost_{v}_224x224_nv12.bin')
            self.assertEqual((selected.contract.input_height, selected.contract.input_width), (224,224))

    def test_rejects_unpublished_target_and_mismatched_identity(self):
        from samples.vision.repghost.runtime.python.model_binding import resolve_selection, BindingError
        for target in ('s100','s100p','s600'):
            with self.assertRaises(BindingError):
                resolve_selection(target)
        with self.assertRaises(BindingError):
            resolve_selection('x5', variant='200', asset_id='x5:repghost:RepGhost_100_224x224_nv12.bin')

    def test_default_download_uses_published_100_and_explicit_variants(self):
        from samples.vision.repghost.model import download
        for v in (None, *VARIANTS):
            downloaded=[]
            def fetch(asset, destination):
                downloaded.append((asset.reference, Path(destination).name))
                return '0'*64
            argv=['--target','x5'] + (['--variant',v] if v else [])
            with patch.object(download,'download_asset',fetch), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download.main(argv),0)
            filename=f'RepGhost_{v or "100"}_224x224_nv12.bin'
            self.assertEqual(downloaded,[(f'x5:repghost:{filename}',filename)])


class SourceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only import the preserved source; SDK construction is never invoked.
        spec=importlib.util.spec_from_file_location('_repghost_source', ROOT/'platforms/x5/samples/vision/repghost/runtime/python/repghost.py')
        mod=importlib.util.module_from_spec(spec)
        old_path=sys.path[:]
        with patch.dict(sys.modules, {'hbm_runtime':types.ModuleType('hbm_runtime'),spec.name:mod}):
            try: spec.loader.exec_module(mod)
            finally: sys.path[:]=old_path
        cls.source=mod.RepGhost.__new__(mod.RepGhost)
        cls.source.cfg=mod.RepGhostConfig('not-loaded',resize_type=1,topk=5)
        cls.source.model_name='fixture';cls.source.input_names=['data'];cls.source.output_names=['prob']
        cls.source.input_h=224;cls.source.input_w=224;cls.source.labels={}

    def task(self, variant=None):
        from samples.vision.repghost.runtime.python.model_binding import bind_model,resolve_selection
        from samples.vision.repghost.runtime.python.classification import ClassificationTask
        selection = resolve_selection('x5', variant=variant)
        height, width = selection.contract.input_height, selection.contract.input_width
        class_count = selection.contract.class_count
        metadata={'model_name':'fixture','input_names':['data'],'input_shapes':{'data':(1,3,height,width)},'input_dtypes':{'data':'U8'},'output_names':['prob'],'output_shapes':{'prob':(1,class_count,1,1)},'output_dtypes':{'prob':'F32'}}
        binding=bind_model(selection,metadata)
        raw={'prob':np.linspace(-3,3,class_count,dtype=np.float32).reshape(1,class_count,1,1)}
        return ClassificationTask(lambda _:raw,binding),raw

    def test_preprocessing_matches_source_bytes_for_both_resize_modes(self):
        task,_=self.task()
        rng=np.random.default_rng(42)
        for shape in ((119,231,3),(225,91,3),(224,224,3)):
            image=rng.integers(0,256,shape,dtype=np.uint8)
            for resize in (0,1):
                task.resize_type=resize
                expected=self.source.pre_process(image,resize)['fixture']['data'].reshape(-1)
                actual=task.pre_process(image).tensors['data']
                np.testing.assert_array_equal(actual,expected)

    def test_forward_preserves_raw_output_and_postprocess_matches_source(self):
        task,raw=self.task()
        observed=task.forward({'data':np.zeros(75264,dtype=np.uint8)})
        self.assertIs(observed,raw)
        expected_ids,expected_scores,_=self.source.post_process(raw)
        result=task.post_process(raw)
        np.testing.assert_array_equal(result.class_ids,expected_ids)
        np.testing.assert_allclose(result.scores,expected_scores,rtol=0,atol=1e-7)

    def test_predict_equals_explicit_stages_and_interleaved_context(self):
        image_a = np.zeros((17, 31, 3), dtype=np.uint8)
        image_a[..., 0] = 23
        image_a[..., 1] = np.arange(31, dtype=np.uint8)
        image_b = np.full((29, 11, 3), 191, dtype=np.uint8)
        image_b[..., 2] = np.arange(29, dtype=np.uint8)[:, None]
        for variant in VARIANTS:
            task, _raw = self.task(variant)
            prepared_a = task.pre_process(image_a)
            saved_a = {name: tensor.copy() for name, tensor in prepared_a.tensors.items()}
            prepared_b = task.pre_process(image_b)
            self.assertEqual(prepared_a.transform.original_height, image_a.shape[0])
            self.assertEqual(prepared_a.transform.original_width, image_a.shape[1])
            self.assertEqual(prepared_b.transform.original_height, image_b.shape[0])
            self.assertEqual(prepared_b.transform.original_width, image_b.shape[1])
            self.assertNotEqual(prepared_a.transform, prepared_b.transform)
            for name, tensor in saved_a.items():
                np.testing.assert_array_equal(prepared_a.tensors[name], tensor)

            prepared_a_again = task.pre_process(image_a)
            self.assertEqual(prepared_a_again.transform, prepared_a.transform)
            for name, tensor in saved_a.items():
                np.testing.assert_array_equal(prepared_a_again.tensors[name], tensor)
            explicit = task.post_process(task.forward(prepared_a))
            chained = task.predict(image_a)
            np.testing.assert_array_equal(chained.class_ids, explicit.class_ids)
            np.testing.assert_array_equal(chained.scores, explicit.scores)
            self.assertEqual(chained.labels, explicit.labels)

    def test_conversion_recipes_remain_identical_to_source(self):
        for v in VARIANTS:
            filename=f'RepGhost_{v}.yaml'
            expected=(ROOT/'platforms/x5/samples/vision/repghost/conversion'/filename).read_bytes()
            self.assertEqual((SAMPLE/'conversion'/filename).read_bytes(),expected)
