import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, lstat } from 'node:fs/promises';
import { resolve, relative, isAbsolute } from 'node:path';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');

/** Load only source-model inference covers whose provenance matches the catalog. */
export async function loadFamilyInferenceBanners({ webRoot, catalog }) {
  const manifestPath = resolve(webRoot, 'release/assets/model-inference-banners.json');
  let bytes;
  try { bytes = await readFile(manifestPath); }
  catch (error) {
    if (error.code === 'ENOENT') return { items: new Map() };
    throw error;
  }
  const manifest = JSON.parse(bytes);
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.kind, 'verified-source-model-inference-banners');
  const assetsRoot = resolve(webRoot, 'release/assets');
  const verified = new Map();
  const file = async record => {
    assert.ok(record && typeof record.file === 'string');
    const path = resolve(assetsRoot, record.file);
    const rel = relative(assetsRoot, path);
    assert.ok(rel && !rel.startsWith('..') && !isAbsolute(rel));
    const stat = await lstat(path);
    assert.ok(stat.isFile() && !stat.isSymbolicLink());
    const data = await readFile(path);
    assert.equal(data.length, record.size_bytes, `${record.file}: size differs`);
    assert.equal(sha(data), record.sha256, `${record.file}: SHA-256 differs`);
    return { path, data };
  };
  const generation = JSON.parse((await file(manifest.source_generation)).data);
  assert.equal(generation.tool, 'built-in Codex image_gen');
  for (const [id, item] of Object.entries(manifest.items)) {
    const model = catalog.models.find(row => row.id === id);
    assert.ok(model, `${id}: cover must identify a catalog model`);
    const variant = model.variants.find(row => row.size === item.model.variant);
    assert.ok(variant, `${id}: cover must identify its real reference variant`);
    assert.ok(variant.platforms.some(row =>
      row.provenance?.source_checkpoint_sha256 === item.model.source_checkpoint_sha256
      && row.provenance?.onnx_sha256 === item.model.onnx_sha256
      && row.provenance?.repository_commit === item.model.conversion_source_commit),
    `${id}: source checkpoint / float graph must match the converted catalog`);
    const image = await file(item.banner);
    await file(item.source_image);
    await file(item.outputs);
    const prediction = JSON.parse((await file(item.predictions)).data);
    const provenance = JSON.parse((await file(item.provenance)).data);
    assert.equal(provenance.status, 'passed');
    assert.equal(provenance.board_execution, 'not-run');
    assert.equal(provenance.model.onnx_sha256, item.model.onnx_sha256);
    assert.equal(prediction.model.onnx_sha256, item.model.onnx_sha256);
    assert.equal(prediction.model.catalog_id, id);
    assert.equal(prediction.image.sha256, item.source_image.sha256);
    const generated = generation.assets.find(row => row.stem === item.stem);
    assert.equal(generated?.input_sha256, item.source_image.sha256);
    for (const [key, name] of [['banner','banner.webp'], ['source_image','source.webp'],
      ['outputs','outputs.npz'], ['predictions','predictions.json']]) {
      assert.equal(provenance.files[name].sha256, item[key].sha256);
    }
    if (item.kind === 'classification') {
      assert.equal(prediction.topk.length, 5);
      assert.deepEqual(provenance.render.rendered_predictions, prediction.topk);
      for (let i = 0; i < prediction.topk.length; i += 1) {
        const row = prediction.topk[i];
        assert.ok(Number.isInteger(row.class_id) && row.class_id >= 0 && row.class_id < 1000);
        assert.ok(Number.isFinite(row.score) && row.score >= 0 && row.score <= 1);
        if (i) assert.ok(prediction.topk[i - 1].score >= row.score);
      }
    }
    assert.equal(item.cover_label, '主机浮点推理参考');
    verified.set(id, { ...item, path: image.path });
  }
  return { items: verified, selection: {
    manifest_path: 'release/assets/model-inference-banners.json',
    manifest_sha256: sha(bytes), verified_models: [...verified.keys()],
  } };
}
