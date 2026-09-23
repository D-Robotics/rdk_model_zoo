"""HGNetV2 migration acceptance on a host; no inference on hardware."""
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
SAMPLE = ROOT / 'samples/vision/hgnetv2'
FILENAMES = {'b0': 'hgnetv2_b0_224x224_nv12.bin', 'b1': 'hgnetv2_b1_224x224_nv12.bin', 'b2': 'hgnetv2_b2_224x224_nv12.bin', 'b3': 'hgnetv2_b3_224x224_nv12.bin', 'b4': 'hgnetv2_b4_224x224_nv12.bin'}
VARIANTS = tuple(FILENAMES)


class EntryTests(unittest.TestCase):
    def test_entry_is_sdk_free_from_unrelated_directory(self):
        for args in (['--help'], ['--list-models'], ['--dry-run', '--target', 'x5']):
            run = subprocess.run([sys.executable, str(SAMPLE/'runtime/python/main.py'), *args], cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            if '--dry-run' in args:
                self.assertIn('variant: b0', run.stdout)

    def test_default_and_all_published_variants(self):
        from samples.vision.hgnetv2.runtime.python.model_binding import resolve_selection, list_available_assets
        self.assertEqual(resolve_selection('x5').variant, 'b0')
        self.assertEqual({r.filename for r in list_available_assets()}, set(FILENAMES.values()))
        for v in VARIANTS:
            selected = resolve_selection('x5', variant=v)
            self.assertEqual(selected.asset_id, f'x5:hgnetv2:{FILENAMES[v]}')
            self.assertEqual((selected.contract.input_height, selected.contract.input_width), (224,224))

    def test_rejects_unpublished_target_and_mismatched_identity(self):
        from samples.vision.hgnetv2.runtime.python.model_binding import resolve_selection, BindingError
        for target in ('s100','s100p','s600'):
            with self.assertRaises(BindingError):
                resolve_selection(target)
        with self.assertRaises(BindingError):
            resolve_selection('x5', variant='b4', asset_id='x5:hgnetv2:hgnetv2_b0_224x224_nv12.bin')

    def test_default_download_uses_published_b0_and_explicit_variants(self):
        from samples.vision.hgnetv2.model import download
        for v in (None, *VARIANTS):
            downloaded=[]
            def fetch(asset, destination):
                downloaded.append((asset.reference, Path(destination).name))
                return '0'*64
            argv=['--target','x5'] + (['--variant',v] if v else [])
            with patch.object(download,'download_asset',fetch), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download.main(argv),0)
            filename=FILENAMES[v or 'b0']
            self.assertEqual(downloaded,[(f'x5:hgnetv2:{filename}',filename)])

    def test_readme_native_commands_parse_and_inputs_exist(self):
        """Catch stale README variants and test-image paths without execution."""
        import re
        import shlex
        from samples.vision.hgnetv2.runtime.python.main import build_parser
        from samples.vision.hgnetv2.model.download import build_parser as download_parser
        checked = 0
        for path in SAMPLE.rglob('README*.md'):
            text = path.read_text().replace('\\\n', ' ')
            for line in text.splitlines():
                tokens = shlex.split(line) if line.startswith('python3 samples/') else []
                if len(tokens) < 2:
                    continue
                if tokens[1].endswith('/runtime/python/main.py'):
                    args = build_parser().parse_args(tokens[2:])
                    self.assertTrue((ROOT / args.test_img).is_file(), (path, args.test_img))
                    checked += 1
                elif tokens[1].endswith('/model/download.py'):
                    download_parser().parse_args(tokens[2:])
                    checked += 1
        self.assertGreaterEqual(checked, 8)


class SourceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only import the preserved source; SDK construction is never invoked.
        spec=importlib.util.spec_from_file_location('_hgnetv2_source', ROOT/'platforms/x5/samples/vision/hgnetv2/runtime/python/hgnetv2.py')
        mod=importlib.util.module_from_spec(spec)
        old_path=sys.path[:]
        with patch.dict(sys.modules, {'hbm_runtime':types.ModuleType('hbm_runtime'),spec.name:mod}):
            try: spec.loader.exec_module(mod)
            finally: sys.path[:]=old_path
        cls.source=mod.HGNetV2.__new__(mod.HGNetV2)
        cls.source.cfg=mod.HGNetV2Config('not-loaded',resize_type=1,topk=5)
        cls.source.model_name='fixture';cls.source.input_names=['data'];cls.source.output_names=['prob']
        cls.source.input_h=224;cls.source.input_w=224;cls.source.labels={}

    def task(self, variant=None):
        from samples.vision.hgnetv2.runtime.python.model_binding import bind_model,resolve_selection
        from samples.vision.hgnetv2.runtime.python.classification import ClassificationTask
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
        source = ROOT / 'platforms/x5/samples/vision/hgnetv2/conversion'
        recipes = list(source.glob('*.yaml'))
        self.assertEqual(len(recipes), 5)
        for recipe in recipes:
            self.assertEqual((SAMPLE/'conversion'/recipe.name).read_bytes(), recipe.read_bytes())

    def test_rejects_incompatible_metadata(self):
        from samples.vision.hgnetv2.runtime.python.model_binding import bind_model, resolve_selection, MetadataMismatchError
        for dtype, size in [('I8',224),('F32',256)]:
            metadata={'model_name':'fixture','input_names':['data'],'input_shapes':{'data':(1,3,size,size)},'input_dtypes':{'data':'U8'},'output_names':['prob'],'output_shapes':{'prob':(1,1000)},'output_dtypes':{'prob':dtype}}
            with self.assertRaises(MetadataMismatchError):
                bind_model(resolve_selection('x5'),metadata)
