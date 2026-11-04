import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';
import { textDemoEntry } from '../llm-demo.mjs';

const webRoot = resolve(import.meta.dirname, '../..');
const record = JSON.parse(execFileSync('python3', ['-c',
  'import yaml,json,sys;print(json.dumps(yaml.safe_load(open(sys.argv[1]))))',
  resolve(webRoot, 'data/llm/gemma/gemma4/text-generation.yaml')], { encoding: 'utf8' }));
const detailSource = await readFile(resolve(webRoot, 'src/detail-view.js'), 'utf8');

for (const variant of record.variants) test(`official ${variant.size} renders its OSS package and single-request reference`, async t => {
  const outputRoot = await mkdtemp(resolve(tmpdir(), 'gemma4-web-'));
  t.after(() => rm(outputRoot, { recursive: true, force: true }));
  const entry = await textDemoEntry({ record, variant, release: variant.platforms[0], webRoot, outputRoot,
    id: `gemma-gemma4-text-generation-${variant.size}-s600`, coverFilename: 'gemma4-text.png' });
  const { model, properties } = entry;
  assert.equal(model.releaseStatus, 'official');
  assert.equal(model.sampleRevision, record.sample_revision);
  assert.equal(model.source, `https://github.com/D-Robotics/rdk_model_zoo/tree/${record.sample_revision}/${record.sample_path}`);
  assert.equal(model.variantName, `Gemma4 ${variant.size.split('-').map(s => s.toUpperCase()).join(' ')}`);
  assert.equal(model.assets.length, 1);
  assert.equal(model.assets[0].format, 'zip');
  assert.match(model.assets[0].url, /^https:\/\/d-robotics-aitoolchain\.oss-cn-beijing\.aliyuncs\.com\/llm\/2\.0\.0\/s600_models\/Gemma4-E[24]B_w[48]\.zip$/);
  assert.equal(model.benchmark, null);
  assert.equal(model.boardMeasurements.status, 'not-run');
  for (const field of ['ppl', 'ttft_ms', 'decode_tokens_per_second', 'peak_memory_bytes']) assert.equal(model.boardMeasurements[field], null);
  assert.equal(model.officialBenchmark.source.runtime_concurrency, 1);
  assert.equal(model.officialBenchmark.source.memory_scope, 'linux-plus-ion');
  assert.equal(model.deployment.artifactManifest.source.revision, null);
  assert.match(model.quickStart[0].command, /python3 -m zipfile -e/);
  assert.match(model.quickStart[0].command, new RegExp(`--profile ${variant.size}`));
  assert.doesNotMatch(model.quickStart[0].command, /archive\.d-robotics|tar -xzf|w4w8/);
  for (const filename of [...Object.values(model.documents), ...Object.values(model.documentsEn), model.sourceArchive]) {
    assert((await stat(resolve(outputRoot, filename))).size > 0);
  }
  const sources = execFileSync('tar', ['-tzf', resolve(outputRoot, model.sourceArchive)], { encoding: 'utf8' });
  assert.doesNotMatch(sources, /\.(hbm|bin|so|pyc)(\n|$)|\/prepared\//);
  for (const locale of ['zh', 'en']) {
    const context = { window: { HubI18n: { locale }, MODEL_PROPERTIES: { [model.id]: properties } } };
    vm.runInNewContext(detailSource, context);
    const html = context.window.ModelDetail.render({ model, models: [model], data: { release: { compatibility: { hardware: 'RDK S600' } } } });
    assert.match(html, /Runtime 并发/);
    assert.match(html, /Runtime 单路/);
    assert.match(html, /id="benchmark-condition"[^>]*aria-label="Runtime 并发"(?![^>]*disabled)/);
    assert.doesNotMatch(html, /Runtime 2 路并发|未板测|SDK 2\.0|峰值内存|视觉配套文件|权重版本|data-ask-model/);
    assert.match(html, /运行内存/);
    assert.match(html, /官方性能基准/);
    assert.match(html, new RegExp(String(model.officialBenchmark.metrics.decode_tokens_per_second).replace('.', '\\.')));
    assert.match(html, /data-open-download/);
    assert(html.includes(model.source), 'Sample source links to the published native commit');
  }
});
