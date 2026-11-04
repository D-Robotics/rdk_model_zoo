"""Official references cannot overwrite local measurements or broaden OSS access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'build_catalog.py'
SPEC = importlib.util.spec_from_file_location('llm_catalog', SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)
RECORD = catalog.DATA_ROOT / 'llm/gemma/gemma4/text-generation.yaml'


class OfficialGemmaTest(unittest.TestCase):
    def setUp(self):
        self.record = yaml.safe_load(RECORD.read_text())
        self.record.pop("sample_revision", None)  # Temporary fixture has no Git object database.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'model_zoo_web/data'
        for variant in self.record['variants']:
            for key in ('model_manifest', 'board_result', 'official_benchmark'):
                name = variant['platforms'][0][key]
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((catalog.ROOT / name).read_text())
        for field, value in [('ROOT', self.root), ('DATA_ROOT', self.data)]:
            patcher = mock.patch.object(catalog, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @property
    def platform(self):
        return self.record['variants'][0]['platforms'][0]

    def edit(self, field, update):
        path = self.root / self.platform[field]
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def validate(self):
        return catalog.validate_record(self.record, self.data / 'llm/gemma/gemma4/text-generation.yaml')

    def test_four_official_profiles_keep_local_board_results_empty(self):
        result = self.validate()
        self.assertEqual(len(result['variants']), 4)
        for variant in result['variants']:
            self.assertEqual(variant['platforms'][0]['status'], 'official')

    def test_vendor_numbers_cannot_become_local_board_measurements(self):
        self.edit('board_result', lambda x: x.update(ttft_ms=74))
        with self.assertRaisesRegex(catalog.CatalogError, 'untested board ttft_ms must be null'):
            self.validate()

    def test_official_download_cannot_redirect_to_arbitrary_oss_or_signed_urls(self):
        original = (self.root / self.platform['model_manifest']).read_text()
        for url in ('https://example.invalid/Gemma4-E2B_w4.zip',
                    'https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip?token=x'):
            (self.root / self.platform['model_manifest']).write_text(original)
            self.edit('model_manifest', lambda x: x['availability'].update(package_url=url))
            with self.subTest(url=url), self.assertRaisesRegex(catalog.CatalogError, 'canonical official S600 OSS ZIP'):
                self.validate()

    def test_partial_checks_and_unknown_hashes_do_not_publish(self):
        original = (self.root / self.platform['model_manifest']).read_text()
        for field, value in [('sha256', '1' * 63), ('verification', {'status': 'verified', 'method': 'HEAD'})]:
            (self.root / self.platform['model_manifest']).write_text(original)
            self.edit('model_manifest', lambda x: x['availability'].update({field: value}))
            with self.subTest(field=field), self.assertRaises(catalog.CatalogError):
                self.validate()

    def test_upstream_revision_cannot_be_borrowed_from_a_different_conversion(self):
        self.edit('model_manifest', lambda x: x['source'].update(revision='1' * 40))
        with self.assertRaisesRegex(catalog.CatalogError, 'unknown upstream revision'):
            self.validate()

    def test_mismatched_variant_and_missing_embeddings_are_rejected(self):
        self.edit('model_manifest', lambda x: x.update(profile='e4b-w4'))
        with self.assertRaisesRegex(catalog.CatalogError, 'artifact identity mismatch'):
            self.validate()
        self.edit('model_manifest', lambda x: x.update(profile='e2b-w4', files=[f for f in x['files'] if f['role'] != 'ple-embedding']))
        with self.assertRaisesRegex(catalog.CatalogError, 'incomplete official Gemma package'):
            self.validate()

    def test_only_sourced_single_request_benchmarks_are_accepted(self):
        self.edit('official_benchmark', lambda x: x['source'].update(runtime_concurrency=2))
        with self.assertRaisesRegex(catalog.CatalogError, 'single-request S600 profile'):
            self.validate()

    def test_official_profile_cannot_bypass_ordinary_release_requirements(self):
        self.record['domain'] = 'vision'
        with self.assertRaises(catalog.CatalogError):
            self.validate()

    def test_local_release_measurements_remain_separate(self):
        self.platform['performance'] = {'ttft_ms': 74}
        with self.assertRaisesRegex(catalog.CatalogError, 'vendor benchmarks are recorded separately'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
