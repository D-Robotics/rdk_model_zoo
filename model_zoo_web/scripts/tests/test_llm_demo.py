"""A compiled demo must not turn missing board evidence into release metrics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'build_catalog.py'
SPEC = importlib.util.spec_from_file_location('llm_demo_catalog', SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)
RECORD_PATH = catalog.DATA_ROOT / 'llm/gemma/gemma4/text-generation.yaml'


class TextDemoTest(unittest.TestCase):
    def setUp(self):
        self.record = yaml.safe_load(RECORD_PATH.read_text())
        self.manifest = json.loads((catalog.ROOT / self.record['variants'][0]['platforms'][0]['model_manifest']).read_text())
        self.board = json.loads((catalog.ROOT / self.record['variants'][0]['platforms'][0]['board_result']).read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'model_zoo_web/data'
        for key, content in [('model_manifest', self.manifest), ('board_result', self.board)]:
            path = self.root / self.platform[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        for field, value in [('ROOT', self.root), ('DATA_ROOT', self.data)]:
            patcher = mock.patch.object(catalog, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @property
    def platform(self):
        return self.record['variants'][0]['platforms'][0]

    def validate(self):
        return catalog.validate_record(self.record, self.data / 'llm/gemma/gemma4/text-generation.yaml')

    def write_board(self):
        (self.root / self.platform['board_result']).write_text(json.dumps(self.board))

    def test_compiled_demo_accepts_null_board_measurements(self):
        result = self.validate()
        self.assertEqual(result['variants'][0]['platforms'][0]['status'], 'demo')

    def test_host_ppl_and_zero_are_not_accepted_as_untested_board_data(self):
        for value in (43.035, 0):
            with self.subTest(value=value):
                self.board['ppl'] = value
                self.write_board()
                with self.assertRaisesRegex(catalog.CatalogError, 'untested board ppl must be null'):
                    self.validate()

    def test_untested_demo_cannot_supply_a_download_or_benchmark(self):
        for field in ('artifact', 'accuracy', 'performance', 'reports'):
            with self.subTest(field=field):
                self.platform[field] = {'url': 'https://example.invalid/model.hbm'}
                with self.assertRaisesRegex(catalog.CatalogError, 'must be null'):
                    self.validate()
                self.platform[field] = None

    def test_verified_archive_package_does_not_require_board_measurements(self):
        self.manifest['availability'] = {
            'kind': 'archive', 'name': 'gemma4-s600.tar.gz',
            'package_url': catalog.ARCHIVE_BASE + '/rdk_s600/gemma/gemma4-s600.tar.gz',
            'checksums_url': catalog.ARCHIVE_BASE + '/rdk_s600/gemma/SHA256SUMS',
            'sha256': '1' * 64, 'size_bytes': 123,
            'verification': {'status': 'verified', 'method': 'anonymous-https-full-sha256',
                             'checked_at': '2026-11-04T00:00:00+00:00'},
        }
        path = self.root / self.platform['model_manifest']
        path.write_text(json.dumps(self.manifest))
        self.validate()
        self.assertIsNone(self.board['ppl'])
        for field, value, message in (
                ('sha256', '1' * 63, 'full SHA256'),
                ('package_url', 'https://example.invalid/model.tar.gz', 'Archive directory'),
                ('verification', {'status': 'pending', 'method': 'HEAD'}, 'full anonymous HTTPS')):
            with self.subTest(field=field):
                content = copy.deepcopy(self.manifest)
                content['availability'][field] = value
                path.write_text(json.dumps(content))
                with self.assertRaisesRegex(catalog.CatalogError, message):
                    self.validate()

    def test_wrong_target_or_context_is_rejected(self):
        self.platform['platform'] = 's100'
        with self.assertRaisesRegex(catalog.CatalogError, 'S600 only'):
            self.validate()
        self.platform['platform'] = 's600'
        self.record['variants'][0]['input']['context_length'] = 8192
        with self.assertRaisesRegex(catalog.CatalogError, 'compiled chunk/context'):
            self.validate()

    def test_missing_runtime_companion_or_ple_is_rejected(self):
        for role in ('runtime-vision-companion', 'ple-embedding'):
            content = copy.deepcopy(self.manifest)
            content['files'] = [asset for asset in content['files'] if asset['role'] != role]
            (self.root / self.platform['model_manifest']).write_text(json.dumps(content))
            with self.subTest(role=role), self.assertRaisesRegex(catalog.CatalogError, 'incomplete'):
                self.validate()

    def test_ordinary_llm_runtime_cannot_replace_gemma4_runner(self):
        self.manifest['runtime']['type'] = 'LLM'
        (self.root / self.platform['model_manifest']).write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(catalog.CatalogError, 'official SDK 2.0 GEMMA4'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
