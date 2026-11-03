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
const variant = record.variants[0];
const release = variant.platforms[0];
const detailSource = await readFile(resolve(webRoot, 'src/detail-view.js'), 'utf8');

test('SDK text demo packages native sample source with unmeasured board fields', async t => {
  const outputRoot = await mkdtemp(resolve(tmpdir(), 'gemma4-web-'));
  t.after(() => rm(outputRoot, { recursive: true, force: true }));
  const entry = await textDemoEntry({ record, variant, release, webRoot, outputRoot,
    id: 'gemma-gemma4-text-generation-e2b-s600', coverFilename: 'gemma4-text.png' });
  const { model, properties } = entry;
  assert.equal(model.releaseStatus, 'demo');
  assert.equal(model.name, 'Gemma4 E2B');
  assert.equal(model.variantName, 'Gemma4 E2B W4/W8');
  assert.equal(model.source, null);
  assert.equal(model.benchmark, null);
  const published = model.deployment.artifactManifest.availability;
  assert.equal(model.assets.length, published.package_url ? 1 : 0);
  if (published.package_url) {
    assert.equal(model.assets[0].url, published.package_url);
    assert.equal(model.assets[0].sha256, published.sha256);
    assert.equal(model.assets[0].sizeBytes, published.size_bytes);
    assert.equal(model.assets[0].checksumsUrl, published.checksums_url);
    assert.match(model.quickStart[0].command, /sha256sum -c -/);
  }
  assert.equal(model.boardMeasurements.status, 'not-run');
  for (const field of ['ppl', 'ttft_ms', 'decode_tokens_per_second', 'peak_memory_bytes']) assert.equal(model.boardMeasurements[field], null);
  assert.equal(properties.contextLength, 4096);
  assert.equal(properties.prefillChunkSize, 512);
  assert.equal(model.deployment.artifactManifest.source.revision.length, 40);
  assert.equal(model.deployment.fileBytes, 7298660276);
  for (const filename of [...Object.values(model.documents), ...Object.values(model.documentsEn), model.sourceArchive]) {
    assert((await stat(resolve(outputRoot, filename))).size > 0);
  }
  const sourceFiles = execFileSync('tar', ['-tzf', resolve(outputRoot, model.sourceArchive)], { encoding: 'utf8' });
  assert.match(sourceFiles, /samples\/llm\/gemma4\/runtime\/cpp\/src\/gemma4\.cc/);
  assert.match(sourceFiles, /samples\/llm\/gemma4\/model\/artifacts\.json/);
  assert.doesNotMatch(sourceFiles, /\.(hbm|bin|so|pyc)(\n|$)|\/prepared\//);
  assert.match(model.quickStart[0].command, /\\\n  --source-dir/);
  for (const step of model.quickStart) assert.doesNotMatch(step.command, /^\s*\+\s/m);

  for (const locale of ['zh', 'en']) {
    const context = { window: { HubI18n: { locale }, MODEL_PROPERTIES: { [model.id]: properties } } };
    vm.runInNewContext(detailSource, context);
    const html = context.window.ModelDetail.render({ model, models: [model], data: {
      release: { compatibility: { hardware: 'RDK X5 / RDK S600' } },
    } });
    assert.match(html, /Runtime 并发/);
    assert.match(html, /Runtime 单路/);
    assert.match(html, /Runtime 2 路并发/);
    assert.match(html, /id="benchmark-condition"[^>]*aria-label="Runtime 并发"(?![^>]*disabled)/);
    assert.match(html, /首 token 延迟/);
    assert.match(html, /板端 PPL/);
    assert.match(html, /峰值内存/);
    assert.match(html, /--config_path/);
    assert.match(html, /示例源码/);
    if (published.package_url) assert.match(html, /data-open-download/);
    else assert.doesNotMatch(html, /data-open-download/);
    assert.doesNotMatch(html, /data-ask-model|未板测|板端测试尚未执行|SDK 2\.0|视觉配套文件|SDK 初始化依赖|权重版本|>6befbaca7398925921802abd1f277b495b78b738<|准备模型文件和官方 Runtime 后|OE 转换报告|GFLOPs|43\.035|href="null/);
    assert.match(html, new RegExp(locale === 'en' ? 'runtime-en\\.md' : 'runtime\\.md'));
  }
});
