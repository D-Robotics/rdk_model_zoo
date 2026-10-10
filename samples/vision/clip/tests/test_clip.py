# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP source parity and host-only paired encoder boundary tests."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / 'samples/vision/clip'
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402
LEGACY = legacy_tree('x5/samples/vision/clip/runtime/python')


def source_module():
    spec = importlib.util.spec_from_file_location('_clip_source_tokenizer', LEGACY/'simple_tokenizer.py')
    tok = importlib.util.module_from_spec(spec); spec.loader.exec_module(tok)
    spec = importlib.util.spec_from_file_location('_clip_source', LEGACY/'clip_retrieval.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'hbm_runtime':types.ModuleType('hbm_runtime'), 'onnxruntime':types.ModuleType('onnxruntime'), 'simple_tokenizer':tok, spec.name:module}):
        spec.loader.exec_module(module)
    old = module.CLIPMatcher.__new__(module.CLIPMatcher)
    old.cfg = module.CLIPConfig('unused', 'unused')
    old.image_model_name='vision'; old.input_names=['pixels']; old.output_names=['embedding']
    old.tokenizer = tok.SimpleTokenizer(str(LEGACY/'bpe_simple_vocab_16e6.txt.gz'))
    return old


class ImageRuntime:
    model_names=['vision']
    input_names={'vision':['pixels']}; input_shapes={'vision':{'pixels':(1,3,224,224)}}
    input_dtypes={'vision':{'pixels':'F32'}}
    output_names={'vision':['embedding']}; output_shapes={'vision':{'embedding':(1,512)}}
    output_dtypes={'vision':{'embedding':'F32'}}
    def __init__(self): self.calls=[]; self.raw=np.arange(512,dtype=np.float32).reshape(1,512)
    def run(self,inputs): self.calls.append(inputs); return {'vision':{'embedding':self.raw}}
    def set_scheduling_params(self,**kw): self.scheduling=kw


class TextSession:
    def __init__(self): self.calls=[]; self.batch=None
    def get_inputs(self): return [types.SimpleNamespace(name='token_ids',type='tensor(int32)',shape=[self.batch,77])]
    def get_outputs(self): return [types.SimpleNamespace(name='text_embeddings',type='tensor(float)',shape=[self.batch,512])]
    def run(self,names,inputs):
        self.calls.append((names,inputs));n=inputs['token_ids'].shape[0]
        self.raw=np.arange(n*512,dtype=np.float32).reshape(n,512)
        return [self.raw]


def fixture():
    from samples.vision.clip.runtime.python.cli import resolve_selection
    from samples.vision.clip.runtime.python.matching import CLIPMatcher, RuntimeModelRunner
    from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer
    image=ImageRuntime(); text=TextSession()
    runner=RuntimeModelRunner(resolve_selection('x5'), image_runtime=image, text_session=text)
    return CLIPMatcher(runner.selection, tokenizer=PromptTokenizer(), runner=runner),runner,image,text


class ClipTests(unittest.TestCase):
    def test_paired_assets_and_explicit_path_identity(self):
        from samples.vision.clip.runtime.python.cli import resolve_selection,list_available_assets
        self.assertEqual({a.filename for a in list_available_assets('x5')},{'img_encoder.bin','text_encoder.onnx'})
        s=resolve_selection('x5');self.assertEqual(s.image_asset.reference,'x5:clip:img_encoder.bin');self.assertEqual(s.text_asset.reference,'x5:clip:text_encoder.onnx')
        for target in ['s100','s100p','s600']:
            with self.assertRaises(ValueError):resolve_selection(target)
        with self.assertRaises(ValueError):resolve_selection('x5',image_model_path='/tmp/custom.bin')
        with self.assertRaises(ValueError):resolve_selection('x5',text_model_path='/tmp/custom.onnx')
        with self.assertRaises(ValueError):resolve_selection('x5',image_asset_id='x5:clip:text_encoder.onnx')

    def test_actual_bpe_tokens_match_source_unicode_empty_and_truncation(self):
        from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer
        tokenizer=PromptTokenizer();old=source_module()
        prompts=['a diagram','a dog','机器人 café &amp; dog','', ' whitespace\t text ']
        np.testing.assert_array_equal(tokenizer(prompts),old.tokenize(prompts))
        with self.assertRaises(RuntimeError):tokenizer(['hello '*100])
        np.testing.assert_array_equal(tokenizer(['hello '*100],truncate=True),old.tokenize(['hello '*100],truncate=True))
        for invalid in [[], 'a dog', [1]]:
            with self.assertRaises((ValueError,TypeError)):tokenizer(invalid)

    def test_preprocess_actual_source_three_geometries_and_call_context(self):
        task,_,_,_=fixture();old=source_module();rng=np.random.default_rng(43)
        images=[rng.integers(0,256,s,dtype=np.uint8) for s in [(31,79,3),(81,29,3)]]
        a=task.pre_process(images[0],['a dog']);saved=a.tensors['image'].copy()
        b=task.pre_process(images[1],['a diagram','a dog']);again=task.pre_process(images[0],['a dog'])
        for image in [*images,np.zeros((224,224,3),np.uint8)]:
            prepared=task.pre_process(image,['a dog'])
            np.testing.assert_array_equal(prepared.tensors['image'],old.pre_process(image)['vision']['pixels'])
            self.assertEqual(prepared.tensors['texts'].dtype,np.int32)
        self.assertEqual(a.context,again.context);self.assertNotEqual(a.context,b.context)
        np.testing.assert_array_equal(a.tensors['image'],saved)

    def test_forward_raw_postprocess_source_cosine_and_predict(self):
        task,_,image,text=fixture();old=source_module();pic=np.zeros((31,72,3),np.uint8)
        prepared=task.pre_process(pic,['a diagram','a dog']);raw=task.forward(prepared.tensors)
        self.assertIs(raw['image_feature'],image.raw);self.assertIs(raw['text_features'],text.raw)
        result=task.post_process(raw);expected=old.post_process(image.raw.reshape(-1).astype(np.float32),text.raw.astype(np.float32))
        np.testing.assert_array_equal(result.scores,expected)
        np.testing.assert_array_equal(result.order,np.argsort(expected)[::-1])
        composed=task.predict(pic,['a diagram','a dog']);np.testing.assert_array_equal(composed.scores,result.scores)
        self.assertEqual(len(image.calls),2);self.assertEqual(len(text.calls),2)

    def test_metadata_and_input_rejection_precedes_encoder_calls(self):
        task,runner,image,text=fixture()
        with self.assertRaises(ValueError):runner({'image':np.zeros((1,3,224,224),np.float32),'texts':np.zeros((2,77),np.int64)})
        self.assertEqual(image.calls,[]);self.assertEqual(text.calls,[])
        from samples.vision.clip.runtime.python.cli import resolve_selection
        from samples.vision.clip.runtime.python.matching import RuntimeModelRunner
        bad=TextSession();bad.get_outputs=lambda:[types.SimpleNamespace(name='features',type='tensor(float)',shape=[None,768])]
        with self.assertRaises(ValueError):RuntimeModelRunner(resolve_selection('x5'),image_runtime=ImageRuntime(),text_session=bad).load()
        fixed=TextSession();fixed.batch=1
        r=RuntimeModelRunner(resolve_selection('x5'),image_runtime=ImageRuntime(),text_session=fixed);r.load()
        with self.assertRaises(ValueError):r({'image':np.zeros((1,3,224,224),np.float32),'texts':np.zeros((2,77),np.int32)})

    def test_singleton_sdk_feature_layouts_preserve_cosine_ranking(self):
        from samples.vision.clip.runtime.python.cli import resolve_selection
        from samples.vision.clip.runtime.python.matching import RuntimeModelRunner

        baseline = None
        for shape in ((1, 512), (1, 512, 1, 1), (1, 1, 1, 512)):
            with self.subTest(shape=shape):
                image, text = ImageRuntime(), TextSession()
                image.output_shapes = {'vision': {'embedding': shape}}
                image.raw = np.arange(512, dtype=np.float32).reshape(shape)
                runner = RuntimeModelRunner(resolve_selection('x5'),
                                            image_runtime=image, text_session=text)
                values = runner({'image': np.zeros((1, 3, 224, 224), np.float32),
                                 'texts': np.zeros((2, 77), np.int32)})
                np.testing.assert_array_equal(values['image_feature'],
                                              np.arange(512, dtype=np.float32)[None])
                task, _, _, _ = fixture()
                result = task.post_process(values)
                if baseline is None:
                    baseline = result.scores
                np.testing.assert_array_equal(result.scores, baseline)

        for shape in ((512,), (2, 256), (1, 2, 256, 1)):
            image = ImageRuntime()
            image.output_shapes = {'vision': {'embedding': shape}}
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                RuntimeModelRunner(resolve_selection('x5'), image_runtime=image,
                                   text_session=TextSession()).load()

    def test_sdk_runtime_feature_shape_and_finite_values_still_checked(self):
        from samples.vision.clip.runtime.python.cli import resolve_selection
        from samples.vision.clip.runtime.python.matching import RuntimeModelRunner

        for raw in (np.zeros((1, 512), np.float32),
                    np.full((1, 512, 1, 1), np.nan, np.float32),
                    np.zeros((1, 512, 1, 1), np.int8)):
            image, text = ImageRuntime(), TextSession()
            image.output_shapes = {'vision': {'embedding': (1, 512, 1, 1)}}
            image.raw = raw
            runner = RuntimeModelRunner(resolve_selection('x5'),
                                        image_runtime=image, text_session=text)
            with self.assertRaises(ValueError):
                runner({'image': np.zeros((1, 3, 224, 224), np.float32),
                        'texts': np.zeros((1, 77), np.int32)})
            self.assertEqual(text.calls, [])

    def test_sdk_free_entry_help_and_explicit_dryrun(self):
        for args in [('--help',),('--list-models',),('--dry-run','--target','x5')]:
            p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),'--dry-run'],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(p.returncode,2)

    def test_execution_identity_gate_precedes_default_sdk_factories(self):
        from samples.vision.clip.runtime.python.cli import resolve_selection
        from samples.vision.clip.runtime.python.matching import RuntimeModelRunner
        with patch('utils.py_utils.platforms.require_execution_target',side_effect=ValueError('wrong board')) as gate:
            with self.assertRaisesRegex(ValueError,'wrong board'):RuntimeModelRunner(resolve_selection('x5')).load()
            gate.assert_called_once_with('x5')


if __name__=='__main__':unittest.main()


class SimplifiedRuntimeTests(unittest.TestCase):
    """2026-10-08 runtime simplification boundary.

    Selection/catalog duties live in ``cli.py``; ``matching.py`` owns the
    named model class, the dual-runtime runner, and multimodal input
    preparation; the per-sample ``model_binding``/``model_runner``/
    ``tensor_io`` forwarding modules are gone. ``simple_tokenizer.py`` stays:
    the preserved BPE tokenizer algorithm and its 77-token prompt bridge.
    """

    def test_matcher_constructs_from_selection_and_runs_stages(self):
        from samples.vision.clip.runtime.python.cli import resolve_selection
        from samples.vision.clip.runtime.python.matching import CLIPMatcher, RuntimeModelRunner
        from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer
        image, text = ImageRuntime(), TextSession()
        model = CLIPMatcher(resolve_selection('x5'), tokenizer=PromptTokenizer(),
                            runner=RuntimeModelRunner(resolve_selection('x5'),
                                                      image_runtime=image, text_session=text))
        result = model.predict(np.zeros((31, 47, 3), np.uint8), ['a dog', 'a cat'])
        self.assertEqual(result.scores.shape, (2,))
        prepared = model.preprocess(np.zeros((31, 47, 3), np.uint8), ['a dog', 'a cat'])
        manual = model.postprocess(model.infer(prepared.tensors))
        np.testing.assert_allclose(result.scores, manual.scores)
        np.testing.assert_array_equal(result.order, manual.order)

    def test_split_forwarding_modules_are_removed(self):
        base = Path(__file__).resolve().parents[1] / 'runtime/python'
        for name in ('model_binding.py', 'model_runner.py', 'tensor_io.py'):
            self.assertFalse((base / name).exists(), name)
