"""Pin the published Detect release through the active downloader, offline."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SAMPLE = Path(__file__).resolve().parents[1]
ROOT = SAMPLE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAMPLE / 'runtime/python'))

from samples._shared import assets
from yolo_assets import manifest_asset, model_url
from yolo_platform import resolve_platform
import yolo_download

# Fixed publication records from 6ee6ab7's legacy manifests, not active values.
RELEASE = json.loads((SAMPLE / 'tests/fixtures/yolo26_detect_release.json').read_text())


class Yolo26ReleaseTests(unittest.TestCase):
    def setUp(self):
        assets._models.cache_clear()

    def tearDown(self):
        assets._models.cache_clear()

    def test_release_covers_all_twenty_combinations(self):
        self.assertEqual(len(RELEASE), 20)
        self.assertEqual(
            {(row['platform'], row['size']) for row in RELEASE},
            {(platform, size) for platform in ('x5', 's100', 's100p', 's600')
             for size in 'nsmlx'})

    def test_resolver_uses_active_manifest_with_exact_url_and_sha256(self):
        for row in RELEASE:
            with self.subTest(platform=row['platform'], size=row['size']):
                profile = resolve_platform(row['platform'])
                asset = manifest_asset(profile, 'yolo26', 'detect', row['size'])
                group = 'x5' if row['platform'] == 'x5' else 's'
                self.assertEqual(asset.source_path, f'docs/release/{group}/models.yaml')
                self.assertEqual(asset.filename, row['filename'])
                self.assertEqual(asset.url, row['url'])
                self.assertEqual(asset.sha256, row['sha256'])
                self.assertEqual(model_url(profile, 'yolo26', 'detect', row['size']), row['url'])

    def test_downloader_passes_exact_release_identity_to_verified_download(self):
        for row in RELEASE:
            with self.subTest(platform=row['platform'], size=row['size']), tempfile.TemporaryDirectory() as directory:
                argv = ['yolo_download', '--platform', row['platform'], '--family', 'yolo26',
                        '--task', 'detect', '--model-size', row['size'], '--model-dir', directory]
                with patch.object(sys, 'argv', argv), patch.object(assets, 'download_asset') as download:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(yolo_download.main(), 0)
                download.assert_called_once()
                asset, destination = download.call_args.args
                self.assertEqual(asset.filename, row['filename'])
                self.assertEqual(asset.url, row['url'])
                self.assertEqual(asset.sha256, row['sha256'])
                self.assertEqual(destination.name, Path(row['filename']).name)

    def test_downloader_rejects_wrong_bytes_for_every_published_digest(self):
        for row in RELEASE:
            with self.subTest(platform=row['platform'], size=row['size']), tempfile.TemporaryDirectory() as directory:
                argv = ['yolo_download', '--platform', row['platform'], '--family', 'yolo26',
                        '--task', 'detect', '--model-size', row['size'], '--model-dir', directory]
                errors = io.StringIO()
                with patch.object(sys, 'argv', argv), patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong model')) as request:
                    with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                        self.assertEqual(yolo_download.main(), 2)
                request.assert_called_once_with(row['url'], timeout=60)
                self.assertIn('SHA-256 mismatch', errors.getvalue())
                self.assertFalse(any(path.is_file() for path in Path(directory).rglob('*')))


if __name__ == '__main__':
    unittest.main()
