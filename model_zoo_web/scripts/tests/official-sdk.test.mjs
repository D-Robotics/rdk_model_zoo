import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';
import { textDemoEntry } from '../llm-demo.mjs';

const webRoot = resolve(import.meta.dirname, '../..');
const records = JSON.parse(execFileSync('python3', ['-c', `import json,pathlib,yaml,sys
out=[]
for p in pathlib.Path(sys.argv[1]).rglob('*.yaml'):
 r=yaml.safe_load(p.read_text())
 if r.get('sample_path')=='samples/llm/oellm':out.append(r)
print(json.dumps(out))`, resolve(webRoot, 'data')], { encoding: 'utf8' }));
const detail = await readFile(resolve(webRoot, 'src/detail-view.js'), 'utf8');

test('official SDK catalog covers all requested profiles and separate series banners', () => {
  assert.equal(records.length, 12);
  assert.equal(records.flatMap(r => r.variants).length, 32);
  assert.equal(new Set(records.map(r => r.official_cover.file)).size, 12);
  assert.deepEqual(new Set(records.map(r => r.task)), new Set(['text-gen', 'vlm', 'asr']));
});

for (const record of records) for (const variant of record.variants) test(`SDK ${variant.size}: own package, reference and native input command`, async t => {
  const outputRoot = await mkdtemp(resolve(tmpdir(), 'sdk-web-'));
  t.after(() => rm(outputRoot, { recursive: true, force: true }));
  const entry = await textDemoEntry({ record, variant, release: variant.platforms[0], webRoot, outputRoot,
    id: `official-${variant.size}`, coverFilename: `${record.family}.png` });
  const { model, properties } = entry;
  const manifest = model.deployment.artifactManifest;
  assert.equal(model.variantName, manifest.display_name);
  assert.equal(model.releaseStatus, 'official');
  assert.equal(model.deployment.boardStatus, 'not-run');
  assert.equal(model.benchmark, null);
  assert.equal(model.officialBenchmark.source.runtime_concurrency, 1);
  assert.equal(model.assets[0].sha256, manifest.availability.sha256);
  assert.equal(model.assets[0].url, manifest.availability.package_url);
  assert.equal(model.assets[0].format, 'zip');
  assert.equal(model.licenseName, manifest.source.license.name);
  assert.equal(model.source, `https://github.com/D-Robotics/rdk_model_zoo/tree/${record.sample_revision}/samples/llm/oellm`);
  assert(model.quickStart.every(s => s.command.split('\n').length <= 3));
  assert.match(model.quickStart[0].command, new RegExp(`--profile ${variant.size}`));
  assert.doesNotMatch(model.quickStart[0].command, /curl|sha256sum|Gemma4/);
  if (record.task === 'asr') {
    assert.match(model.quickStart[1].command, /--audio_path/);
    assert.doesNotMatch(model.quickStart[1].command, /--prompt|--max_new_tokens/);
    assert.equal(model.taskId, 'speech-recognition');
    assert.equal(properties.sampleRate, 16000);
  } else if (record.task === 'vlm') {
    assert.match(model.quickStart[1].command, /--image_path/);
    assert.equal(model.taskId, 'vision-language-model');
  } else assert.equal(model.taskId, 'text-generation');
  const context = { window: { HubI18n: { locale: 'zh' }, MODEL_PROPERTIES: { [model.id]: properties } } };
  vm.runInNewContext(detail, context);
  const html = context.window.ModelDetail.render({ model, models: [model], data: { release: { compatibility: { hardware: 'RDK S600' } } } });
  assert.match(html, /Runtime 单路/);
  assert.match(html, /id="benchmark-condition"[^>]*aria-label="Runtime 并发"(?![^>]*disabled)/);
  assert.doesNotMatch(html, /Runtime 2 路并发|未板测|SDK 2\.0|官方性能基准|转换说明|板端 PPL/);
  assert.match(html, /运行内存/);
  if (model.officialBenchmark.metrics.decode_tokens_per_second_range) {
    assert(html.includes(model.officialBenchmark.metrics.decode_tokens_per_second_range.join('–')));
    assert.equal(model.officialBenchmark.metrics.decode_tokens_per_second, null);
  }
  if (record.task === 'asr') assert.match(html, /实时率 RTF/);
  const files = execFileSync('tar', ['-tzf', resolve(outputRoot, model.sourceArchive)], { encoding: 'utf8' });
  assert.doesNotMatch(files, /\.(hbm|bin|so|pyc)(\n|$)|simple_demo\.cc|\/prepared\//);
});
