import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { quickstartForUsage } from '../pending-preview.mjs';

const revision = '7a97c1780b0cfa131ff0007d662e255e5e41c12e';
const repoRoot = fileURLToPath(new URL('../../..', import.meta.url));

function blob(path) {
  return execFileSync('git', ['show', `${revision}:${path}`], { cwd: repoRoot, encoding: 'utf8' });
}

test('quick start commands are the pinned README text, including continuations', () => {
  const clip = quickstartForUsage(blob('samples/vision/clip/README.md'), [
    'python3 samples/vision/clip/model/download.py --target x5',
    'python3 samples/vision/clip/runtime/python/main.py --target x5',
  ], 'X5');
  assert.equal(clip.download, 'python3 samples/vision/clip/model/download.py --target x5');
  assert.equal(clip.run, 'python3 samples/vision/clip/runtime/python/main.py --target x5');

  const convnext = quickstartForUsage(blob('samples/vision/convnext/README.md'), [
    'bash samples/vision/convnext/model/download.sh x5',
    'python3 samples/vision/convnext/runtime/python/main.py',
    '--target x5',
    '--asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin',
    '--model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin',
    '--test-img samples/vision/convnext/test_data/cheetah.JPEG',
    '--label-file datasets/imagenet/imagenet_classes.names',
  ], 'X5');
  assert.equal(convnext.download, 'bash samples/vision/convnext/model/download.sh x5');
  assert.equal(convnext.run, [
    'python3 samples/vision/convnext/runtime/python/main.py \\',
    '  --target x5 \\',
    '  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \\',
    '  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \\',
    '  --test-img samples/vision/convnext/test_data/cheetah.JPEG \\',
    '  --label-file datasets/imagenet/imagenet_classes.names',
  ].join('\n'));
  assert.equal(blob('samples/vision/convnext/README.md').includes(convnext.run), true);
});

test('a platform page keeps that platform command and drops dry runs', () => {
  const text = [
    'python3 sample/model/download.py --target s100',
    'python3 sample/model/download.py --target x5',
    'python3 sample/runtime/python/main.py --target x5 --dry-run',
    'python3 sample/runtime/python/main.py --target x5',
  ].join('\n');
  const selected = quickstartForUsage(text, text.split('\n'), 'X5');
  assert.equal(selected.download, 'python3 sample/model/download.py --target x5');
  assert.equal(selected.run, 'python3 sample/runtime/python/main.py --target x5');

  const video = [
    "curl -fL 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \\",
    '  -o samples/vision/bytetrack/test_data/track_test.mp4',
    'python3 -m samples.vision.bytetrack.model.download --target s100 \\',
    '  --output-dir samples/vision/bytetrack/model',
  ].join('\n');
  const tracker = quickstartForUsage(video, video.split('\n').map(line => line.replace(/ \\$/, '')), 'S100');
  assert.equal(tracker.download, video.split('\n').slice(2).join('\n'));
});
