import assert from 'node:assert/strict';
import { cp, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, relative } from 'node:path';
import { runInNewContext } from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadVerifiedPreviewDownloads, previewDownloadAssets } from './preview-downloads.mjs';
import { loadVerifiedRebuildSelection } from './yolo26-repaired-selection.mjs';
import { stampAssetVersions } from './asset-versions.mjs';

const args = process.argv.slice(2);
let rebuildSelectionDir = null;
for (let index = 0; index < args.length; index += 1) {
  if (args[index] === '--rebuild-selection-dir' && args[index + 1]) {
    if (rebuildSelectionDir) throw new Error('--rebuild-selection-dir may be supplied once');
    rebuildSelectionDir = resolve(args[++index]);
    continue;
  }
  throw new Error(`Unknown or incomplete option: ${args[index]}`);
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const activeRoot = resolve(root, 'dist');
const outputRoot = resolve(root, '.dist-candidates-passed-next');
const activeCatalogPath = resolve(root, 'build/catalog.json');
const activeInputsPath = resolve(root, 'release/inputs.json');
const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex');
const toPlainJson = value => JSON.parse(JSON.stringify(value));
const loadWindowData = async path => {
  const bytes = await readFile(path);
  const context = { window: {} };
  runInNewContext(bytes.toString('utf8'), context);
  return { value: context.window, bytes };
};

const activeCatalog = JSON.parse(await readFile(activeCatalogPath, 'utf8'));
const activeInputs = JSON.parse(await readFile(activeInputsPath, 'utf8'));
assert.equal(activeInputs.schema_version, 1);
assert.equal(Object.keys(activeInputs.models || {}).length, 2);
assert.equal(Object.keys(activeInputs.releases || {}).length, 40);
const detectCatalog = activeCatalog.models || [];
assert.equal(detectCatalog.length, 2);
assert.ok(detectCatalog.every(record => record.task === 'detect' && ['yolo11', 'yolo26'].includes(record.family)));
assert.deepEqual(new Set(detectCatalog.map(record => record.family)), new Set(['yolo11', 'yolo26']));
const activeReleaseIds = new Set(detectCatalog.flatMap(record => (record.variants || []).flatMap(variant =>
  (variant.platforms || []).filter(release => release.status === 'released')
    .map(release => `${record.id}/${variant.size}/${release.platform}`))));
assert.equal(activeReleaseIds.size, 40);
assert.ok([...activeReleaseIds].every(id => Object.hasOwn(activeInputs.releases, id)));

const [{ value: activeWindow, bytes: activeDataBytes }, { value: previewWindow },
  { value: activePropsWindow, bytes: activePropsBytes }, { value: previewPropsWindow },
  { value: activeReportsWindow, bytes: activeReportsBytes }, { value: previewReportsWindow }] = await Promise.all([
  loadWindowData(resolve(activeRoot, 'data.js')),
  loadWindowData(resolve(outputRoot, 'data.js')),
  loadWindowData(resolve(activeRoot, 'model-properties.js')),
  loadWindowData(resolve(outputRoot, 'model-properties.js')),
  loadWindowData(resolve(activeRoot, 'reports/reports-data.js')),
  loadWindowData(resolve(outputRoot, 'reports/reports-data.js')),
]);
const activeData = activeWindow.MODEL_DATA;
const previewData = previewWindow.MODEL_DATA;
const activeModels = activeData.models || [];
const candidateModels = previewData.models || [];
assert.equal(activeModels.length, 40);
assert.ok(activeModels.every(model => model.releaseStatus === 'released' && model.taskId === 'object-detection'));
assert.ok(activeModels.every(model => ['yolo11', 'yolo26'].includes(model.family)));
const selectionPath = resolve(outputRoot, 'technical-selection.json');
const selection = JSON.parse(await readFile(selectionPath, 'utf8'));
assert.ok(candidateModels.length > 0);
assert.equal(candidateModels.length, selection.selected.length);
assert.ok(candidateModels.every(model => model.releaseStatus === 'candidate'));
const rebuildSelection = rebuildSelectionDir
  ? await loadVerifiedRebuildSelection({
    directory: rebuildSelectionDir,
    snapshotId: selection.snapshot_id,
    auditSha: selection.source_hashes?.authoritative_audit_sha256,
  })
  : null;
const rebuildModels = rebuildSelection?.models || [];
const rebuildProperties = rebuildSelection?.properties || {};
const rebuildReports = rebuildSelection?.reports || [];
const candidateCatalogPath = resolve(root, 'build', `candidate-${selection.snapshot_id}`, 'catalog.json');
const candidateCatalog = JSON.parse(await readFile(candidateCatalogPath, 'utf8'));
const selectedReleaseIds = new Set(selection.selected.map(row => row.model_id));
const verifiedDownloads = await loadVerifiedPreviewDownloads({
  webRoot: root,
  snapshotId: selection.snapshot_id,
  auditSha: selection.source_hashes?.authoritative_audit_sha256,
  catalog: candidateCatalog,
  selectedReleaseIds,
});
if (verifiedDownloads) {
  assert.deepEqual(selection.preview_downloads, verifiedDownloads.selection,
    'selection must bind the exact verified preview download manifest set');
  assert.equal(selection.release_gates?.model_download_assets, 'verified-preview-downloads');
} else {
  assert.equal(Object.hasOwn(selection, 'preview_downloads'), false,
    'selection cannot claim verified downloads without a manifest');
  assert.ok(candidateModels.every(model => Array.isArray(model.assets) && model.assets.length === 0),
    'candidate assets require the verified preview download manifest');
}
for (const model of candidateModels) {
  const releaseId = `${model.catalogId}/${model.modelSize}/${model.releasePlatform.toLowerCase()}`;
  const expectedAssets = previewDownloadAssets(verifiedDownloads, releaseId, model.releasePlatform);
  assert.deepEqual(toPlainJson(model.assets), expectedAssets,
    `candidate assets must match the verified overlay exactly: ${releaseId}`);
}
const candidateDownloadCount = candidateModels.reduce((sum, model) => sum + model.assets.length, 0);
const verifiedDownloadCount = verifiedDownloads?.selection?.verified_models || 0;
assert.equal(candidateDownloadCount, verifiedDownloadCount,
  'only selected models bound by verified preview manifests may have download assets');
const modelIds = new Set([...activeModels, ...candidateModels].map(model => model.id));
assert.equal(modelIds.size, activeModels.length + candidateModels.length, 'published Detect and candidate IDs must not collide');
for (const model of activeModels) {
  const id = `${model.catalogId}/${model.modelSize}/${model.releasePlatform.toLowerCase()}`;
  assert.ok(activeReleaseIds.has(id), `active model is not in the Detect catalog: ${id}`);
  assert.ok(Object.hasOwn(activeInputs.releases, id), `active model has no release inputs: ${id}`);
  assert.ok(model.assets.length, `published Detect model has no download asset: ${model.id}`);
}

const activeProps = activePropsWindow.MODEL_PROPERTIES || {};
const candidateProps = previewPropsWindow.MODEL_PROPERTIES || {};
const activeReports = activeReportsWindow.OE_REPORTS || [];
const candidateReports = previewReportsWindow.OE_REPORTS || [];
assert.equal(Object.keys(activeProps).length, 40);
assert.equal(activeReports.length, 40);
assert.equal(candidateReports.length, candidateModels.length);
for (const report of activeReports) {
  assert.ok(modelIds.has(report.modelIds?.[0]), `active report references unknown model ${report.id}`);
  if (report.dataUrl) assert.ok(await stat(resolve(activeRoot, report.dataUrl)).then(value => value.isFile()));
}
for (const report of candidateReports) {
  if (report.dataUrl) assert.ok(await stat(resolve(outputRoot, report.dataUrl)).then(value => value.isFile()));
}

async function listFiles(directory, prefix = '') {
  const paths = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const child = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) paths.push(...await listFiles(resolve(directory, item.name), child));
    else if (item.isFile()) paths.push(child);
  }
  return paths;
}

async function copyTreeWithoutCollisions(source, destination) {
  const files = await listFiles(source);
  for (const file of files) {
    const target = resolve(destination, file);
    assert.equal(await stat(target).then(() => true).catch(() => false), false,
      `refusing to overwrite candidate preview file ${relative(outputRoot, target)}`);
  }
  for (const file of files) {
    const from = resolve(source, file);
    const to = resolve(destination, file);
    await mkdir(dirname(to), { recursive: true });
    await cp(from, to);
  }
}

async function copyTreeIfIdentical(source, destination) {
  const files = await listFiles(source);
  for (const file of files) {
    const target = resolve(destination, file);
    const existing = await readFile(target).catch(error => {
      if (error.code === 'ENOENT') return null;
      throw error;
    });
    if (existing) {
      const incoming = await readFile(resolve(source, file));
      assert.equal(hashBytes(existing), hashBytes(incoming),
        `refusing to replace different preview file ${relative(outputRoot, target)}`);
    }
  }
  for (const file of files) {
    const from = resolve(source, file);
    const to = resolve(destination, file);
    const exists = await stat(to).then(() => true).catch(() => false);
    if (exists) continue;
    await mkdir(dirname(to), { recursive: true });
    await cp(from, to);
  }
}

for (const relativePath of ['assets/models', 'reports/data', 'reports/models']) {
  const source = resolve(activeRoot, relativePath);
  if (await stat(source).then(value => value.isDirectory()).catch(() => false)) {
    await copyTreeWithoutCollisions(source, resolve(outputRoot, relativePath));
  }
}

if (rebuildSelection) {
  for (const relativePath of ['assets/models', 'reports/data', 'reports/models']) {
    const source = resolve(rebuildSelection.directory, relativePath);
    if (await stat(source).then(value => value.isDirectory()).catch(() => false)) {
      await copyTreeIfIdentical(source, resolve(outputRoot, relativePath));
    }
  }
}

const combinedReports = [...activeReports, ...candidateReports, ...rebuildReports];
const combinedProperties = { ...activeProps, ...candidateProps, ...rebuildProperties };
const combinedModels = [...activeModels, ...candidateModels, ...rebuildModels];
assert.equal(Object.keys(combinedProperties).length, combinedModels.length);
assert.equal(new Set(combinedModels.map(model => model.id)).size, combinedModels.length,
  'published Detect, candidate, and rebuilt model IDs must be unique');
const platforms = [...new Set(combinedModels.flatMap(model => model.platforms || []))];
const combinedData = {
  ...previewData,
  catalog: {
    status: 'mixed-preview',
    summary: {
      sample_count: new Set(combinedModels.map(model => model.sample)).size,
      asset_count: combinedModels.reduce((sum, model) => sum + (model.assets?.length || 0), 0),
      downloadable_asset_count: combinedModels.reduce((sum, model) => sum + (model.assets?.length || 0), 0),
      benchmark_count: combinedModels.filter(model => model.benchmark).length,
    },
  },
  release: {
    ...previewData.release,
    status: 'mixed-preview',
    compatibility: { hardware: platforms.map(platform => `RDK ${platform}`).join(' / ') },
  },
  candidatePreview: true,
  technicalPassedPreview: false,
  models: combinedModels,
};

if (rebuildSelection) {
  selection.rebuild_selection = {
    schema_version: 1,
    kind: 'yolo26-repaired-release-selection',
    status: 'ready-and-hash-verified',
    manifest_path: rebuildSelection.manifestPath,
    manifest_sha256: rebuildSelection.manifestSha256,
    base_snapshot_id: rebuildSelection.manifest.base_snapshot_id,
    base_authoritative_audit_sha256: rebuildSelection.manifest.base_authoritative_audit_sha256,
    rebuild_input_inventory: rebuildSelection.manifest.rebuild_input_inventory,
    source_evidence_index_sha256: rebuildSelection.manifest.source_evidence_index_sha256,
    releases: rebuildSelection.records.map(record => ({
      release_id: record.releaseId,
      model_id: record.model.id,
      build_id: record.record.build_id,
      artifact_sha256: record.model.assets[0].sha256,
      row_sha256: hashBytes(Buffer.from(JSON.stringify(record.model))),
      evidence_index: Object.fromEntries(Object.entries(record.evidence)
        .map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
      source_evidence_index: record.sourceEvidence,
      package_index: Object.fromEntries(Object.entries(record.files)
        .map(([role, ref]) => [role, { path: ref.path, sha256: hashBytes(ref.bytes) }])),
      upload_index: Object.fromEntries(Object.entries(record.uploads)
        .map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
      metric_bindings: record.record.metric_bindings,
    })),
  };
}

selection.preview_composition = {
  kind: rebuildSelection
    ? 'published-detect-plus-technical-candidates-plus-repaired-releases'
    : 'published-detect-plus-technical-candidates',
  published_detect: {
    count: activeModels.length,
    model_ids: activeModels.map(model => model.id),
    catalog_sha256: hashBytes(await readFile(activeCatalogPath)),
    inputs_sha256: hashBytes(await readFile(activeInputsPath)),
    generated_data_sha256: hashBytes(activeDataBytes),
    model_properties_sha256: hashBytes(activePropsBytes),
    reports_registry_sha256: hashBytes(activeReportsBytes),
  },
  technical_candidates: candidateModels.length,
  repaired_releases: rebuildModels.length,
  verified_candidate_downloads: candidateDownloadCount,
  combined_count: combinedData.models.length,
  candidate_release_gates_unchanged: true,
};
await writeFile(selectionPath, `${JSON.stringify(selection, null, 2)}\n`, 'utf8');
await writeFile(resolve(outputRoot, 'data.js'), `window.MODEL_DATA = ${JSON.stringify(combinedData, null, 2)};\n`, 'utf8');
await writeFile(resolve(outputRoot, 'model-properties.js'),
  `window.MODEL_PROPERTIES = Object.freeze(${JSON.stringify(combinedProperties, null, 2)});\n`, 'utf8');
await writeFile(resolve(outputRoot, 'reports/reports-data.js'),
  `window.OE_REPORTS = ${JSON.stringify(combinedReports, null, 2)};\n`, 'utf8');
await writeFile(resolve(outputRoot, 'reports/inventory.json'),
  `${JSON.stringify({ reports: combinedReports.map(entry => ({ id: entry.id, kind: entry.kind })) }, null, 2)}\n`, 'utf8');
await stampAssetVersions(outputRoot);

console.log(`Merged ${activeModels.length} published Detect entries, ${candidateModels.length} technical candidates, and ${rebuildModels.length} rebuilt releases into the isolated preview build.`);
