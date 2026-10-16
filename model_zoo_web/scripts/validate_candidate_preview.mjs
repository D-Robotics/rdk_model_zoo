import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import vm from 'node:vm';

const webRoot = resolve(import.meta.dirname, '..');
const outputRoot = resolve(webRoot, 'dist-candidates');
const dataSource = await readFile(resolve(outputRoot, 'data.js'), 'utf8');
const context = { window: {} };
vm.createContext(context);
vm.runInContext(dataSource, context);
const data = context.window.MODEL_DATA;

assert.equal(data.candidatePreview, true, 'output must be the isolated candidate preview');
assert.equal(data.catalog.status, 'candidate', 'candidate catalog must not be marked published');
assert.equal(data.release.status, 'candidate', 'candidate release must not be marked published');
assert.equal(data.models.length, 60, 'expected one preview entry per board-evaluated platform');
assert.equal(new Set(data.models.map(model => model.id)).size, 60, 'candidate preview ids must be unique');

const expectedMetrics = {
  'image-classification': 2,
  'instance-segmentation': 2,
  'pose-estimation': 1,
  'object-detection': 1,
};
const counts = new Map();
for (const model of data.models) {
  assert.equal(model.releaseStatus, 'candidate', `${model.id}: release status must remain candidate`);
  assert.deepEqual(Array.from(model.assets || []), [], `${model.id}: candidate must not expose a downloadable URL`);
  assert.match(model.coverLabel || '', /非模型推理结果/, `${model.id}: cover must be labeled as illustrative`);
  const coverPath = resolve(outputRoot, model.coverImage);
  assert((await stat(coverPath)).size > 0, `${model.id}: candidate cover is missing`);
  assert(model.reportDataUrl, `${model.id}: staged OE report data is required in preview`);
  const reportPath = resolve(outputRoot, model.reportDataUrl);
  assert((await stat(reportPath)).size > 0, `${model.id}: OE report data is missing`);
  const accuracy = model.benchmark?.accuracy || [];
  assert.equal(accuracy.length, expectedMetrics[model.taskId], `${model.id}: task metric count mismatch`);
  assert(accuracy.every(metric => metric.model_stage === 'board'), `${model.id}: only standalone board metrics may be displayed`);
  assert(!accuracy.some(metric => ['float', 'quantized'].includes(metric.model_stage)), `${model.id}: Float-vs-Runtime comparison must not be inferred`);
  assert(accuracy.every(metric => metric.source_scope?.en === 'Standalone board evaluation'), `${model.id}: board accuracy source note is missing`);
  if (model.taskId === 'object-detection' && /OBB/i.test(model.name)) {
    assert(accuracy.every(metric => metric.scope?.en === 'Local DOTA val · single scale'), `${model.id}: OBB local DOTA val scope is missing`);
  }
  counts.set(model.taskId, (counts.get(model.taskId) || 0) + 1);
}

for (const [taskId, count] of counts) assert.equal(count, 15, `${taskId}: expected 15 size/platform combinations`);
const reportsInventory = JSON.parse(await readFile(resolve(outputRoot, 'reports/inventory.json'), 'utf8'));
assert.equal(reportsInventory.reports.length, 60, 'candidate preview must include 60 OE report mappings');
console.log('candidate preview validated: 60 board-only model entries, 60 OE reports, candidate status retained');
