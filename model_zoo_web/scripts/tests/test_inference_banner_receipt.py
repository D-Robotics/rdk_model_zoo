"""New visualizations retain their source input, every output head and render."""
import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('banner_catalog', Path(__file__).parents[1]/'build_catalog.py')
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class InferenceBannerReceiptTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.asset = 'model_zoo_web/release/assets/pending-heroes/reference.webp'
        path = self.root/self.asset
        path.parent.mkdir(parents=True)
        path.write_bytes(b'recorded render')
        digest = hashlib.sha256(b'committed source').hexdigest()
        self.entry = {'commit': 'a'*40, 'input': 'samples/vision/example/test_data/input.png', 'file': self.asset}
        self.payload = {'schema_version': 2, 'status': 'passed', 'provider': 'CPUExecutionProvider',
                        'board_execution': 'not-run', 'dataset_accuracy': 'not-run',
                        'input_sha256': digest, 'native_helpers': {'utils/py_utils/example.py': digest},
                        'graphs': {'model': {'sha256': 'b'*64}},
                        'graph_io': {'model': {'outputs': [{'name': n, 'shape': [1, 2]} for n in ('cls', 'patch')]}},
                        'forward_outputs': [{'role': 'model', 'name': n, 'shape': [1, 2],
                                             'dtype': 'float32', 'sha256': 'c'*64} for n in ('cls', 'patch')],
                        'files': {'banner.webp': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                  'size_bytes': path.stat().st_size}},
                        'visualization': {'method': 'computed feature projection'}}
        self.stack = [patch.object(catalog, 'ROOT', self.root),
                      patch.object(catalog, 'validate_evidence_path', side_effect=lambda p, *args: p),
                      patch.object(catalog.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'committed source'))]
        for context in self.stack:
            context.start()
            self.addCleanup(context.stop)

    def check(self):
        catalog._validate_inference_banner_receipt(self.payload, self.entry, 'hero')

    def test_complete_real_output_receipt_is_accepted(self):
        self.check()

    def test_missing_and_repeated_heads_are_rejected(self):
        original = copy.deepcopy(self.payload)
        for rows in ([], original['forward_outputs'][:1], original['forward_outputs']*2):
            self.payload['forward_outputs'] = rows
            with self.subTest(count=len(rows)), self.assertRaises(catalog.CatalogError):
                self.check()

    def test_changed_input_and_helper_revision_are_rejected(self):
        for key in ('input_sha256', 'native_helpers'):
            original = copy.deepcopy(self.payload)
            self.payload[key] = 'd'*64 if key == 'input_sha256' else {'utils/py_utils/example.py': 'd'*64}
            with self.subTest(key=key), self.assertRaises(catalog.CatalogError):
                self.check()
            self.payload = original

    def test_changed_output_shape_is_rejected(self):
        self.payload['forward_outputs'][0]['shape'] = [1, 99]
        with self.assertRaises(catalog.CatalogError):
            self.check()

    def test_dynamic_axes_still_require_actual_positive_integer_dimensions(self):
        self.payload['graph_io']['model']['outputs'][0]['shape'] = ['dynamic', 2]
        for dim in (0, -1, 'not-a-dimension', True):
            self.payload['forward_outputs'][0]['shape'] = [dim, 2]
            with self.subTest(dim=dim), self.assertRaises(catalog.CatalogError):
                self.check()

    def test_clip_features_cover_the_complete_prompt_batch(self):
        self.entry['kind'] = 'similarity-panel'
        self.payload['graphs'] = {role: {'sha256': 'b'*64} for role in ('image', 'text')}
        self.payload['graph_io'] = {role: {'outputs': [{'name': 'feature', 'shape': ['batch', 512]}]}
                                    for role in ('image', 'text')}
        self.payload['prompts'] = ['a dog', 'a cat']
        self.payload['forward_outputs'] = [{'role': role, 'name': 'feature', 'shape': shape,
                                            'dtype': 'float32', 'sha256': 'c'*64}
                                           for role, shape in [('image', [1, 512]), ('text', [2, 512])]]
        self.check()
        self.payload['forward_outputs'][1]['shape'] = [1, 512]
        with self.assertRaisesRegex(catalog.CatalogError, 'supplied prompt'):
            self.check()

    def test_changed_render_is_rejected(self):
        (self.root/self.asset).write_bytes(b'another render')
        with self.assertRaises(catalog.CatalogError):
            self.check()

    def test_semantic_hero_requires_the_computed_result_image(self):
        self.entry.update(kind='semantic-wipe', runtime='host-float-reference',
                          note='Host float reference, not board output.',
                          model_url='https://example.com/official.pdparams', model_file='official.pdparams',
                          framework='ONNX Runtime', documented_in='samples/vision/example/conversion/README.md')
        for key in ('model_url','model_file','framework','commit','runtime','input'):
            self.payload[key]=self.entry[key]
        source=self.asset.replace('.webp','.json')
        record=self.root/source;self.entry['predictions_source']=source
        result=self.root/self.asset.replace('.webp','-result.png')
        result.write_bytes(b'computed semantic labels')
        self.payload['files']['result.png']={'sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'size_bytes':result.stat().st_size}
        with patch.object(catalog,'_ls_tree',return_value=('100644','blob','a'*40)), \
             patch.object(catalog,'validate_commit',side_effect=lambda revision,label:revision):
            record.write_text(json.dumps(self.payload))
            catalog._validate_host_float_hero(self.entry,'hero','a'*40)
            del self.payload['files']['result.png'];record.write_text(json.dumps(self.payload))
            with self.assertRaisesRegex(catalog.CatalogError,'computed result image'):
                catalog._validate_host_float_hero(self.entry,'hero','a'*40)

    def test_different_models_cannot_reuse_the_same_input_photo(self):
        records = []
        for model in ('first', 'second'):
            source = f'model_zoo_web/release/assets/pending-heroes/{model}.json'
            (self.root/source).write_text(json.dumps({'schema_version': 2, 'input_sha256': 'a'*64}))
            records.append({'id': model, 'hero_visual': {'runtime': 'host-float-reference',
                                                        'predictions_source': source}})
        with self.assertRaisesRegex(catalog.CatalogError, 'already used by first'):
            catalog.validate_unique_hero_inputs(records)
        (self.root/records[1]['hero_visual']['predictions_source']).write_text(
            json.dumps({'schema_version': 2, 'input_sha256': 'b'*64}))
        catalog.validate_unique_hero_inputs(records)

    def test_plate_hero_cannot_replace_the_actual_readout(self):
        self.entry.update(kind='plate-readout', runtime='host-float-reference',
                          note='Host float reference, not board output.',
                          model_url='https://example.com/official.pth', model_file='official.pth',
                          framework='ONNX Runtime', documented_in='samples/vision/example/conversion/README.md')
        for key in ('model_url', 'model_file', 'framework', 'commit', 'runtime', 'input'):
            self.payload[key] = self.entry[key]
        source = self.asset.replace('.webp', '.json')
        record = self.root/source
        self.entry['predictions_source'] = source
        result = self.root/self.asset.replace('.webp', '-result.png')
        result.write_bytes(b'actual plate result')
        self.payload['files']['result.png'] = {
            'sha256': hashlib.sha256(result.read_bytes()).hexdigest(), 'size_bytes': result.stat().st_size}
        self.payload['recognition'] = {'plate': '皖831289'}
        self.payload['visualization']['rendered_text'] = '皖831289'
        with patch.object(catalog, '_ls_tree', return_value=('100644', 'blob', 'a'*40)), \
             patch.object(catalog, 'validate_commit', side_effect=lambda revision, label: revision):
            record.write_text(json.dumps(self.payload))
            catalog._validate_host_float_hero(self.entry, 'hero', 'a'*40)
            self.payload['visualization']['rendered_text'] = '皖A31289'
            record.write_text(json.dumps(self.payload))
            with self.assertRaisesRegex(catalog.CatalogError, 'actual CTC readout'):
                catalog._validate_host_float_hero(self.entry, 'hero', 'a'*40)

    def check_new_hero(self, kind):
        self.entry.update(kind=kind, runtime='host-float-reference', note='Host float reference, not board output.',
                          model_url='https://example.com/source.pt', model_file='source.pt',
                          framework='ONNX Runtime', documented_in='samples/vision/example/conversion/README.md')
        for key in ('model_url','model_file','framework','commit','runtime','input'):
            self.payload[key]=self.entry[key]
        source=self.asset.replace('.webp','.json');self.entry['predictions_source']=source
        result=self.root/self.asset.replace('.webp','-result.png');result.write_bytes(b'actual computed result')
        self.payload['files']['result.png']={'sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'size_bytes':result.stat().st_size}
        (self.root/source).write_text(json.dumps(self.payload))
        with patch.object(catalog,'_ls_tree',return_value=('100644','blob','a'*40)), \
             patch.object(catalog,'validate_commit',side_effect=lambda revision,label:revision):
            catalog._validate_host_float_hero(self.entry,'hero','a'*40)

    def test_track_labels_must_use_actual_cpu_tracker_ids(self):
        self.payload.update(tracks=[{'id':1},{'id':3}])
        self.payload['visualization']['rendered_track_ids']=[1,3]
        self.check_new_hero('tracking-overlay')
        self.payload['visualization']['rendered_track_ids']=[999]
        with self.assertRaisesRegex(catalog.CatalogError,'actual tracker'):
            self.check_new_hero('tracking-overlay')

    def test_transcript_requires_every_asr_stage_and_exact_native_audio(self):
        roles=('encoder','predictor','decoder')
        self.payload['graphs']={r:{'sha256':'b'*64}for r in roles}
        self.payload['graph_io']={r:{'outputs':[{'name':n,'shape':[1,2]}for n in (['alphas','hidden']if r=='predictor'else['value'])]}for r in roles}
        self.payload['forward_outputs']=[{'role':r,'name':v['name'],'shape':[1,2],'dtype':'float32','sha256':'c'*64}for r in roles for v in self.payload['graph_io'][r]['outputs']]
        self.payload['recognition']={'text':'actual speech'}
        self.payload['visualization']['rendered_text']='actual speech'
        self.payload['audio']={'input':'samples/speech/paraformer/test_data/audio/example.wav',
            'sha256':hashlib.sha256(b'committed source').hexdigest(),'sample_rate':16000,'sample_count':16000}
        self.check_new_hero('speech-transcript')
        self.payload['audio']['sha256']='0'*64
        with self.assertRaisesRegex(catalog.CatalogError,'audio bytes'):
            self.check_new_hero('speech-transcript')
        self.payload['audio']['sha256']=hashlib.sha256(b'committed source').hexdigest()
        self.payload['forward_outputs']=[v for v in self.payload['forward_outputs']if v['role']!='predictor']
        with self.assertRaisesRegex(catalog.CatalogError,'omit or repeat'):
            self.check_new_hero('speech-transcript')


if __name__ == '__main__':
    unittest.main()
