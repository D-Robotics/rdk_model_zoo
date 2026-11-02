import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import vm from 'node:vm';

const [rootArg, candidateArg] = process.argv.slice(2);
if (!rootArg || !candidateArg) throw new Error('Usage: node validate_osnet_candidate.mjs OUTPUT_ROOT CANDIDATE_ROOT');
const root = resolve(rootArg), candidate = resolve(candidateArg);
const context = { window: {} };
vm.createContext(context);
for (const name of ['data.js', 'model-properties.js', 'reports/reports-data.js']) {
  vm.runInContext(await readFile(resolve(root, name), 'utf8'), context);
}
const data = context.window.MODEL_DATA;
assert.equal(data.catalog.status, 'local-candidate');
const models = data.models.filter(m => m.catalogId === 'osnet/osnet/reid');
const catalog = JSON.parse(await readFile(resolve(candidate, 'catalog.json'), 'utf8'));
const platforms = catalog.models[0].variants[0].platforms;
assert.equal(models.length, platforms.length);
for (const model of models) {
const target = model.releasePlatform.toLowerCase();
const release = JSON.parse(await readFile(resolve(candidate, target, 'release.json'), 'utf8'));
const platformRecord = platforms.find(p => p.platform === target);
assert(platformRecord);
assert.deepEqual(platformRecord.performance.raw_measurements.map(r => r.log), release.performance.raw_measurements.map(r => r.log));
assert.equal(model.releaseStatus, 'candidate');
assert.equal(model.releasePlatform, release.platform.toUpperCase());
assert.equal(model.runtimeVariant, release.provenance.runtime_variant || 'spatial-rgb-late8');
assert.equal(model.taskId, 'person-reidentification');
assert.equal(model.benchmark.precisionPolicy, release.provenance.precision_policy);
assert.equal(model.source, null, 'Unpublished sample must not link to a nonexistent public path');
assert(model.sourceArchive);
assert.equal(model.benchmark.accuracy.length, 4);
for (const [field, metric] of [['rank1', 'reid-rank-1'], ['map', 'reid-map']]) {
  for (const [source, stage] of [['float_onnx', 'float'], ['runtime', 'quantized']]) {
    const row = model.benchmark.accuracy.find(r => r.metric === metric && r.model_stage === stage);
    assert.equal(row.unit, 'ratio');
    assert.equal(row.value, release.accuracy[source][field]);
  }
}
for (const threads of [1, 2]) {
  const original = release.performance.measurements.find(r => r.threads === threads);
  assert.equal(model.benchmark.performance.find(r => r.metric === 'latency' && r.concurrency === threads).value, original.average_latency_ms);
  assert.equal(model.benchmark.performance.find(r => r.metric === 'throughput' && r.concurrency === threads).value, original.aggregate_fps);
  const e2e = model.benchmark.endToEnd.find(r => r.timing.pipelineStreams === threads);
  const raw = release.performance.end_to_end.find(r => r.pipeline_streams === threads);
  assert.equal(e2e.timing.cpuWorkerThreads, threads);
  assert.equal(e2e.timing.opencvThreads, undefined);
  assert.equal(e2e.metrics.endToEnd.value, raw.metrics_ms.end_to_end.mean);
  assert.equal(e2e.throughputFps, raw.throughput_fps);
}
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(digest(await readFile(resolve(root, model.sourceArchive))), platformRecord.source_archive.sha256);
assert.equal(platformRecord.provenance.sample_source_commit, platformRecord.source_archive.commit);
for (const asset of model.assets) {
  const bytes = await readFile(resolve(root, asset.url));
  assert.equal(bytes.length, asset.sizeBytes);
  assert.equal(digest(bytes), asset.sha256);
}
assert.equal(model.assets[0].sha256, release.artifact.sha256);
const image = await readFile(resolve(root, model.coverImage));
const width = image.readUInt32BE(16), height = image.readUInt32BE(20);
assert(width >= 1280 && height >= 720, 'Cover must remain sharp at the 16:9 hero size');
assert(Math.abs(width / height - 16 / 9) < 0.005, 'Cover must use a 16:9 composition');
const properties = context.window.MODEL_PROPERTIES[model.id];
assert.equal(properties.runtimeInput.payload_layout, release.artifact.runtime_tensor.payload_layout);
assert.equal(properties.runtimeInput.metadata_layout, release.artifact.runtime_tensor.metadata_layout);
assert.equal(properties.parameterCount, release.model.parameter_count);
const report = context.window.OE_REPORTS.find(r => r.modelIds.includes(model.id));
const oe = JSON.parse(await readFile(resolve(root, report.dataUrl), 'utf8'));
assert.equal(oe.provenance.artifact_sha256, release.artifact.sha256);
assert.equal(digest(await readFile(resolve(root, report.path))), release.reports.oe_report_html_sha256);
if (target === 'x5') assert.equal(oe.quantization.nodes.length, 0, 'Unavailable original layer data must stay unavailable');
else assert(oe.quantization.nodes.length > 0, 'S compile must retain native node similarity evidence');
}
console.log('OSNet multi-platform local candidate accepted: paired full accuracy, Runtime/C++ 1/2, model/source downloads, original OE HTML, tensor contract and 16:9 cover.');
