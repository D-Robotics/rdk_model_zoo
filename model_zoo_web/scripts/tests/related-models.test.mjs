import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const detailSource = readFileSync(new URL('../../src/detail-view.js', import.meta.url), 'utf8');

function model({ id, family, catalogId, modelSize = 'n', platform = 'S100', name = 'YOLO Detect' }) {
  return {
    id,
    family,
    catalogId,
    modelSize,
    releasePlatform: platform,
    name,
    task: '目标检测',
    taskId: 'detect',
    shape: '1 × 3 × 640 × 640',
    coverImage: 'assets/models/detect.webp',
    coverLabel: '参考推理结果',
    source: 'https://example.com/model',
    assets: [{ format: 'hbm', filename: `${id}.hbm`, url: 'https://example.com/model.hbm', sizeBytes: 1024 }],
    benchmark: { environment: { hardware: `RDK ${platform}` }, performance: [] },
  };
}

function relatedLinks(current, models) {
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(detailSource, context);
  const data = { release: { compatibility: { hardware: 'RDK S100' } } };
  const html = context.window.ModelDetail.render({ model: current, models, data });
  const section = html.match(/<section class="mz-related">([\s\S]*?)<\/section>/)?.[1] || '';
  return [...section.matchAll(/<a href="#model\/([^\"]+)" class="mz-related-model"[\s\S]*?<h3>([^<]+)<\/h3>/g)]
    .map(([, id, name]) => ({ id, name }));
}

test('related models collapse family variants and link to a representative released entry', () => {
  const current = model({ id: 'yolo11-n-s100p', family: 'yolo11', catalogId: 'yolo11/detect', platform: 'S100P' });
  const models = [
    current,
    model({ id: 'yolo11-s-s100', family: 'yolo11', catalogId: 'yolo11/detect', modelSize: 's' }),
    model({ id: 'yolo11-n-x5', family: 'yolo11', catalogId: 'legacy-yolo11/detect', platform: 'X5' }),
    model({ id: 'yolo26-n-s100', family: 'yolo26', catalogId: 'yolo26/detect', name: 'YOLO26 Detect' }),
    model({ id: 'yolo26-n-s600', family: 'yolo26', catalogId: 'yolo26/detect', platform: 'S600', name: 'YOLO26 Detect' }),
    model({ id: 'yolo26-s-s600', family: 'yolo26', catalogId: 'yolo26/detect', modelSize: 's', platform: 'S600', name: 'YOLO26 Detect' }),
    model({ id: 'other-n-x5', family: 'other', catalogId: 'other/detect', platform: 'X5', name: 'Other Detect' }),
  ];

  assert.deepEqual(relatedLinks(current, models), [
    { id: 'yolo26-n-s600', name: 'YOLO26 Detect' },
    { id: 'other-n-x5', name: 'Other Detect' },
  ]);
});
