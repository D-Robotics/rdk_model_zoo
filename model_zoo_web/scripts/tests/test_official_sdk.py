"""Keep vendor packages scoped, variant licenses accurate and local results empty."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'build_catalog.py'
SPEC = importlib.util.spec_from_file_location('official_sdk_catalog', SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class OfficialSDKTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'model_zoo_web/data'
        self.path = self.data / 'llm/qwen/qwen3/text-gen.yaml'
        self.record = yaml.safe_load((catalog.DATA_ROOT / 'llm/qwen/qwen3/text-gen.yaml').read_text())
        self.record.pop('sample_revision')  # This temporary fixture has no Git object database.
        self.sample = self.root / 'samples/llm/oellm'
        shutil.copytree(catalog.ROOT / 'samples/llm/oellm', self.sample, ignore=shutil.ignore_patterns('__pycache__', 'build'))
        self.cover = self.root / self.record['official_cover']['file']
        self.cover.parent.mkdir(parents=True)
        for field in ('file', 'provenance'):
            shutil.copyfile(catalog.ROOT / self.record['official_cover'][field], self.root / self.record['official_cover'][field])
        for field, value in [('ROOT', self.root), ('DATA_ROOT', self.data)]:
            patch = mock.patch.object(catalog, field, value)
            patch.start()
            self.addCleanup(patch.stop)

    def variant(self, name='qwen3-4b-w4-eagle3'):
        return next(v for v in self.record['variants'] if v['size'] == name)

    def edit(self, key, transform, profile='qwen3-4b-w4-eagle3'):
        path = self.root / self.variant(profile)['platforms'][0][key]
        value = json.loads(path.read_text())
        transform(value)
        path.write_text(json.dumps(value))

    def validate(self):
        return catalog.validate_record(self.record, self.path)

    def test_profiles_keep_complete_vendor_bindings(self):
        result = self.validate()
        self.assertEqual(len(result['variants']), 11)

    def test_local_measurements_cannot_be_populated_from_vendor_tables(self):
        self.edit('board_result', lambda x: x.update(ttft_ms=88))
        with self.assertRaisesRegex(catalog.CatalogError, 'untested board ttft_ms'):
            self.validate()

    def test_other_tasks_cannot_use_the_sdk_exception(self):
        self.edit('model_manifest', lambda x: x.update(scope='manip'))
        with self.assertRaisesRegex(catalog.CatalogError, 'identity mismatch'):
            self.validate()

    def test_missing_draft_model_is_rejected(self):
        self.edit('model_manifest', lambda x: x.update(files=[a for a in x['files'] if a['role'] != 'draft-hbm']))
        with self.assertRaisesRegex(catalog.CatalogError, 'incomplete official SDK'):
            self.validate()

    def test_changed_vendor_config_requires_a_new_source_identity(self):
        path = self.sample / 'runtime/cpp/configs/qwen3-4b-w4-eagle3.json'
        value = json.loads(path.read_text())
        value['lm_model_file'] = 'wrong.hbm'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(catalog.CatalogError, 'configuration changed'):
            self.validate()

    def test_downloads_must_use_the_canonical_unsigned_official_url(self):
        self.edit('model_manifest', lambda x: x['availability'].update(package_url=x['availability']['package_url'] + '?token=x'))
        with self.assertRaisesRegex(catalog.CatalogError, 'canonical SDK S600 OSS'):
            self.validate()

    def test_speculative_speed_range_cannot_be_replaced_with_its_maximum(self):
        self.edit('official_benchmark', lambda x: x['profiles']['qwen3-4b-w4-eagle3']['metrics'].update(decode_tokens_per_second=89.3))
        with self.assertRaisesRegex(catalog.CatalogError, 'published range'):
            self.validate()

    def test_another_series_cannot_reuse_this_banner(self):
        self.record['official_cover']['file'] = self.record['official_cover']['file'].replace('qwen3.png', 'qwen2-5.png')
        with self.assertRaisesRegex(catalog.CatalogError, 'own reviewed banner'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
