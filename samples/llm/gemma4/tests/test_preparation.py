"""Host-only artifact integrity and SDK configuration checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gemma4_prepare', ROOT / 'model/prepare.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.language = self.root / 'language'
        self.companion = self.root / 'sdk-companion'
        self.output = self.root / 'prepared'
        self.language.mkdir()
        self.companion.mkdir()
        self.manifest = copy.deepcopy(prepare.load_manifest())
        for asset in self.manifest['files']:
            content = ('host fixture: ' + asset['name']).encode()
            asset.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            directory = self.companion if asset['role'] == 'runtime-vision-companion' else self.language
            (directory / asset['name']).write_bytes(content)

    def test_preparation_binds_both_embeddings_and_sdk_vision_without_board_execution(self):
        path = prepare.prepare(self.language, self.output, self.companion, manifest=self.manifest)
        config = json.loads(path.read_text())
        self.assertEqual(config['work_dir'], str(self.output.resolve()))
        self.assertEqual(config['runtime_type'], 'GEMMA4')
        self.assertEqual(config['model_runner']['prefill'], 'gemma4_prefill')
        for field in ('lm_model_file', 'embed_weight_name', 'ple_embed_weight_name', 'vit_model_file'):
            self.assertTrue((self.output / config[field]).is_file())
        self.assertEqual(config['embed_table_scale'], 1.0)
        self.assertEqual(config['ple_embed_table_scale'], 1.0)
        self.assertFalse(config['prefix_cache']['enable'])
        self.assertEqual(json.loads((self.output / 'model-manifest.json').read_text()), self.manifest)

    def test_corrupted_source_is_rejected_without_installing_a_partial_model(self):
        path = self.language / self.manifest['files'][0]['name']
        content = path.read_bytes()
        path.write_bytes(b'X' * len(content))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            prepare.prepare(self.language, self.output, self.companion, manifest=self.manifest)
        self.assertFalse(self.output.exists())

    def test_missing_ple_or_runtime_companion_is_rejected(self):
        for role, directory in [('ple-embedding', self.language), ('runtime-vision-companion', self.companion)]:
            asset = next(asset for asset in self.manifest['files'] if asset['role'] == role)
            path = directory / asset['name']
            original = path.read_bytes()
            path.unlink()
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'Missing file'):
                prepare.prepare(self.language, self.output, self.companion, manifest=self.manifest)
            path.write_bytes(original)

    def test_existing_mismatched_file_is_not_overwritten(self):
        self.output.mkdir()
        path = self.output / self.manifest['files'][0]['name']
        path.write_bytes(b'existing user data')
        with self.assertRaises(ValueError):
            prepare.prepare(self.language, self.output, self.companion, manifest=self.manifest)
        self.assertEqual(path.read_bytes(), b'existing user data')

    def test_recorded_board_results_remain_unmeasured(self):
        board = json.loads((ROOT / 'evaluator/board_result.json').read_text())
        self.assertEqual(board['status'], 'not-run')
        for field in ('ppl', 'ttft_ms', 'decode_tokens_per_second', 'peak_memory_bytes', 'generation_checks', 'environment'):
            self.assertIsNone(board[field])


if __name__ == '__main__':
    unittest.main()
