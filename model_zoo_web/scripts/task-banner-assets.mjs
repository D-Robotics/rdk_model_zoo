import assert from 'node:assert/strict';
import { readFile, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative } from 'node:path';

const tasks = ['cls', 'seg', 'pose', 'obb'];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

export async function loadInferenceTaskBanners({ webRoot, catalog }) {
  const manifestPath = resolve(webRoot, 'release/assets/yolo26-real-task-banners.json');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes);
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.kind, 'verified-model-inference-task-covers');
  assert.deepEqual(Object.keys(manifest.items).sort(), [...tasks].sort());
  const assetsRoot = resolve(webRoot, 'release/assets');
  const verified = new Map();
  const verifyFile = async record => {
    assert.ok(record && typeof record.file === 'string');
    const path = resolve(assetsRoot, record.file);
    const pathRelative = relative(assetsRoot, path);
    assert.ok(pathRelative && !pathRelative.startsWith('..') && !pathRelative.startsWith('/'));
    const details = await lstat(path);
    assert.ok(details.isFile() && !details.isSymbolicLink());
    const bytes = await readFile(path);
    assert.ok(bytes.length > 0);
    assert.equal(bytes.length, record.size_bytes, `${record.file}: size mismatch`);
    assert.equal(digest(bytes), record.sha256, `${record.file}: SHA-256 mismatch`);
    return { path, bytes };
  };
  const sourceGeneration = JSON.parse((await verifyFile(manifest.source_generation)).bytes);
  assert.equal(sourceGeneration.tool, 'built-in Codex image_gen');
  for (const task of tasks) {
    const item = manifest.items[task];
    const id = `ultralytics_yolo/yolo26/${task}`;
    const model = catalog.models.find(record => record.id === id);
    assert.ok(model, `${id}: banner must refer to a catalog task`);
    assert.equal(item.model.family, 'yolo26');
    assert.equal(item.model.task, task);
    assert.equal(item.model.size, 'x');
    const checkpointHashes = new Set((model.variants.find(variant => variant.size === 'x')?.platforms || [])
      .map(release => release.provenance?.source_checkpoint_sha256).filter(Boolean));
    assert.equal(checkpointHashes.size, 1, `${id}: expected one source checkpoint hash`);
    assert.ok(checkpointHashes.has(item.model.sha256), `${id}: banner checkpoint must match catalog`);
    const image = await verifyFile(item);
    await verifyFile(item.source_image);
    await verifyFile(item.raw_predictions);
    const predictions = await verifyFile(item.predictions);
    const provenance = await verifyFile(item.provenance);
    const inference = JSON.parse(provenance.bytes);
    const generatedSource = sourceGeneration.assets.find(asset => asset.task === task);
    assert.equal(generatedSource?.sha256, item.source_image.sha256);
    assert.equal(inference.model.family, item.model.family);
    assert.equal(inference.model.task, task);
    assert.equal(inference.model.size, item.model.size);
    assert.equal(inference.model.sha256, item.model.sha256);
    assert.equal(inference.image.sha256, item.source_image.sha256);
    assert.equal(inference.output.annotated_sha256, item.sha256);
    assert.equal(inference.output.predictions_sha256, item.predictions.sha256);
    assert.equal(inference.output.tensors_sha256, item.raw_predictions.sha256);
    const result = JSON.parse(predictions.bytes);
    assert.equal(result.model.sha256, item.model.sha256);
    assert.equal(result.model.task, task);
    assert.equal(result.model.family, item.model.family);
    assert.equal(result.model.size, item.model.size);
    assert.equal(result.image.sha256, item.source_image.sha256);
    assert.equal(inference.runtime.framework, 'Ultralytics');
    assert.equal(inference.runtime.device, 'cpu');
    assert.ok(inference.render.rendered_predictions.length > 0);
    for (const rendered of inference.render.rendered_predictions) {
      const prediction = task === 'cls'
        ? result.topk.find(row => row.class_id === rendered.class_id)
        : result.detections[rendered.prediction_index];
      assert.ok(prediction);
      assert.equal(rendered.class_id, prediction.class_id);
      assert.equal(rendered.confidence, prediction.confidence);
    }
    assert.match(item.cover_label, /实际推理结果/);
    verified.set(id, { ...item, path: image.path });
  }
  return {
    items: verified,
    selection: {
      manifest_path: 'release/assets/yolo26-real-task-banners.json',
      manifest_sha256: digest(manifestBytes),
      verified_tasks: tasks,
    },
  };
}
