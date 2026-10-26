import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { quickstartForUsage, scopeQuickstart } from '../pending-preview.mjs';

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

test('a sized quick start stays on the released variant it names', () => {
  const quickstart = {
    download: 'bash samples/vision/convnext/model/download.sh x5',
    run: [
      'python3 samples/vision/convnext/runtime/python/main.py \\',
      '  --target x5 \\',
      '  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \\',
      '  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \\',
    ].join('\n'),
  };
  const atto = scopeQuickstart(quickstart, 'ConvNeXt Atto 224x224', true, true);
  assert.equal(atto.download, quickstart.download);
  assert.equal(atto.run, quickstart.run);
  const nano = scopeQuickstart(quickstart, 'ConvNeXt Nano 224x224', false, true);
  assert.equal(nano.download, '');
  assert.equal(nano.run, '');
  const clip = {
    download: 'python3 samples/vision/clip/model/download.py --target x5',
    run: 'python3 samples/vision/clip/runtime/python/main.py --target x5',
  };
  assert.deepEqual(scopeQuickstart(clip, 'CLIP', false, false), clip);
});

test('SigLIP quick starts keep their patch size and resolution', () => {
  const quickstart = {
    download: 'python3 samples/vision/siglip/model/download.py --target s100 --variant base-patch16-224',
    run: 'python3 samples/vision/siglip/runtime/python/main.py --target s100 --variant base-patch16-224 --submodel pooler_output',
  };
  assert.deepEqual(scopeQuickstart(quickstart, 'SigLIP base patch16 224 on RDK S100', true, true), quickstart);
  for (const label of ['SigLIP base patch16 384 on RDK S100', 'SigLIP base patch16 512 on RDK S100', 'SigLIP base patch32 224']) {
    assert.deepEqual(scopeQuickstart(quickstart, label, true, true), { download: '', run: '' });
  }
});

test('YOLOE quick starts do not cross composite model sizes', () => {
  const quickstart = {
    download: 'bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s',
    run: 'python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11s',
  };
  assert.deepEqual(scopeQuickstart(quickstart, '11s', true, true), quickstart);
  for (const label of ['11m', '11l', '26s']) {
    assert.deepEqual(scopeQuickstart(quickstart, label, true, true), { download: '', run: '' });
  }
});

test('quoted and equals-style variant arguments keep the same matching rules', () => {
  for (const argument of ['--variant=11s', '--variant "11s"', "--variant='11s'"]) {
    const quickstart = { download: '', run: `python3 sample/runtime/python/main.py ${argument}` };
    assert.equal(scopeQuickstart(quickstart, '11s', true, true).run, quickstart.run);
    assert.equal(scopeQuickstart(quickstart, '11m', true, true).run, '');
  }
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
