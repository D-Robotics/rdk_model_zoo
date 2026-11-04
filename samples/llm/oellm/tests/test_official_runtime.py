"""Host checks for official profile binding; no SDK inference or downloads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

SAMPLE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('sdk_prepare', SAMPLE / 'model/prepare.py')
launcher = load('sdk_launcher', SAMPLE / 'runtime/cpp/run.py')


class OfficialRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample = self.root / 'sample'
        shutil.copytree(SAMPLE / 'model', self.sample / 'model', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(SAMPLE / 'runtime/cpp/configs', self.sample / 'runtime/cpp/configs')
        shutil.copyfile(SAMPLE / 'LICENSE', self.sample / 'LICENSE')
        for module, field, value in [(prepare, 'SAMPLE', self.sample), (prepare, 'INDEX', self.sample / 'model/artifacts.json'), (launcher, 'SAMPLE', self.sample)]:
            patch = mock.patch.object(module, field, value)
            patch.start()
            self.addCleanup(patch.stop)

    def fixture(self, profile):
        manifest = prepare.load_manifest(profile)
        directory = self.root / profile
        directory.mkdir()
        for asset in manifest['files']:
            content = ('host-only fixture: ' + asset['name']).encode()
            asset.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            path = directory / asset['name']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        path = self.sample / 'model' / prepare.load_index()['profiles'][profile]
        path.write_text(json.dumps(manifest))
        return directory, manifest

    def test_scope_contains_every_requested_profile_and_preserves_empty_local_results(self):
        profiles = prepare.load_index()['profiles']
        self.assertEqual(len(profiles), 32)
        self.assertEqual({prepare.load_manifest(p)['scope'] for p in profiles}, {'text-gen', 'vlm', 'asr'})
        for profile in profiles:
            manifest = prepare.load_manifest(profile)
            with self.subTest(profile=profile):
                self.assertIsNone(manifest['source']['revision'])
                self.assertEqual(manifest['board_tests'], 'not-run')
                self.assertTrue(all(asset['crc_verified'] for asset in manifest['files']))
                board = json.loads((SAMPLE / f'evaluator/board_results/{profile}.json').read_text())
                self.assertEqual(board['status'], 'not-run')
                self.assertTrue(all(board[k] is None for k in ('ppl', 'ttft_ms', 'decode_tokens_per_second', 'rtf', 'peak_memory_bytes', 'environment')))

    def test_every_sdk_configuration_binds_all_original_package_members(self):
        for profile in prepare.load_index()['profiles']:
            manifest = prepare.load_manifest(profile)
            config = json.loads((self.sample / manifest['runtime']['config_template']).read_text())
            names = {asset['name'] for asset in manifest['files']}
            with self.subTest(profile=profile):
                self.assertTrue(set(value for _, value in prepare.config_bindings(config)) <= names)
                self.assertEqual(config['runtime_type'], manifest['runtime']['type'])
                self.assertEqual(hashlib.sha256((self.sample / manifest['runtime']['config_template']).read_bytes()).hexdigest(), manifest['runtime']['config_source_sha256'])

    def test_eagle_and_dflash_keep_their_draft_resources(self):
        for profile in ('qwen3-4b-w4-eagle3', 'qwen3-4b-w4-dflash'):
            source, manifest = self.fixture(profile)
            output = self.root / (profile + '-prepared')
            config_path = prepare.prepare(source, output, manifest)
            config = json.loads(config_path.read_text())
            self.assertEqual(config['work_dir'], str(output.resolve()))
            for _, filename in prepare.config_bindings(config):
                self.assertTrue((output / filename).is_file())
            self.assertTrue(config['infer_acceleration'][0]['speculative']['enable'])

    def test_corruption_is_rejected_without_installing_a_partial_directory(self):
        source, manifest = self.fixture('qwen3-0-6b-w8')
        (source / manifest['files'][0]['name']).write_bytes(b'broken')
        output = self.root / 'output'
        with self.assertRaisesRegex(ValueError, 'Missing file or wrong size'):
            prepare.prepare(source, output, manifest)
        self.assertFalse(output.exists())

    def test_existing_user_data_is_not_replaced(self):
        source, manifest = self.fixture('qwen3-0-6b-w8')
        output = self.root / 'output'
        output.mkdir()
        path = output / manifest['files'][0]['name']
        path.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            prepare.prepare(source, output, manifest)
        self.assertEqual(path.read_bytes(), b'user data')

    def test_native_commands_use_the_corresponding_sdk_cpp_program(self):
        for profile in ('qwen3-0-6b-w8', 'qwen3-vl-2b-w4', 'qwen3-asr-0-6b-w8', 'whisper-small-w8'):
            source, manifest = self.fixture(profile)
            config = prepare.prepare(source, source, manifest)
            options = ['--config_path', str(config), '--dry-run']
            if manifest['scope'] == 'vlm':
                options += ['--image_path', str(self.root / 'image.jpg')]
            if manifest['scope'] == 'asr':
                options += ['--audio_path', str(self.root / 'audio.flac')]
            args = launcher.build_parser().parse_args(options)
            command, _, _ = launcher.native_command(args, Path('/official-sdk'))
            self.assertEqual(command[0], '/official-sdk/' + manifest['runtime']['entrypoint'])
            if manifest['scope'] != 'asr':
                self.assertIn('sync', command)
                self.assertIn('--no_warmup', command)
            else:
                self.assertNotIn('--mode', command)
                self.assertIn('--audio_path', command)

    def test_task_mismatches_and_modified_model_identity_are_rejected(self):
        source, manifest = self.fixture('qwen3-0-6b-w8')
        config = prepare.prepare(source, source, manifest)
        args = launcher.build_parser().parse_args(['--config_path', str(config), '--image_path', '/image.jpg'])
        with self.assertRaisesRegex(ValueError, 'vision-language'):
            launcher.native_command(args, Path('/sdk'))
        args.image_path = None
        identity = source / 'model-manifest.json'
        changed = json.loads(identity.read_text())
        changed['source']['revision'] = '0' * 40
        identity.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            launcher.native_command(args, Path('/sdk'))

    def test_unsafe_manifest_member_is_rejected(self):
        profile = 'qwen3-0-6b-w8'
        manifest = prepare.load_manifest(profile)
        manifest['files'][0]['name'] = '../outside'
        (self.sample / 'model' / prepare.load_index()['profiles'][profile]).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            prepare.load_manifest(profile)


if __name__ == '__main__':
    unittest.main()
