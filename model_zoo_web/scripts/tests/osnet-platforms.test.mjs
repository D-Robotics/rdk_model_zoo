import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../../src/detail-view.js', import.meta.url), 'utf8');
for (const platform of ['X5', 'S100', 'S100P', 'S600']) {
  test(`OSNet ${platform} quickstart uses its exact runtime and artifact`, () => {
    const context = { window: {} };
    vm.createContext(context);
    vm.runInContext(source, context);
    const runtimeVariant = platform === 'X5' ? 'spatial-rgb-late8' : `${platform.toLowerCase()}-f32-int16`;
    const extension = platform === 'X5' ? 'bin' : 'hbm';
    const model = {
      id: `osnet-${platform}`, catalogId: 'osnet/osnet/reid', family: 'osnet', modelSize: 'x1.0',
      releasePlatform: platform, releaseStatus: 'candidate', runtimeVariant,
      name: 'OSNet', task: '行人重识别', taskId: 'person-reidentification', shape: '1 × 3 × 256 × 128',
      coverImage: 'cover.png', assets: [{ filename: `${platform}.${extension}`, format: extension, url: `models/${platform}.${extension}`, sizeBytes: 1024 }],
      benchmark: { environment: { hardware: `RDK ${platform}` }, performance: [] },
    };
    const html = context.window.ModelDetail.render({ model, models: [model], data: { release: { compatibility: { hardware: `RDK ${platform}` } } } });
    assert(html.includes(`--target ${platform.toLowerCase()} --variant ${runtimeVariant}`));
    assert(html.includes(`--model-path /path/to/${platform}.${extension}`));
    assert(!html.includes('/tree/main/samples/vision/osnet'), 'Unpublished sample must use the candidate source archive');
  });
}
