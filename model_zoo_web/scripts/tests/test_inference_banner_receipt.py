"""New visualizations retain their source input, every output head and render."""
import copy
import hashlib
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

    def test_changed_render_is_rejected(self):
        (self.root/self.asset).write_bytes(b'another render')
        with self.assertRaises(catalog.CatalogError):
            self.check()


if __name__ == '__main__':
    unittest.main()
