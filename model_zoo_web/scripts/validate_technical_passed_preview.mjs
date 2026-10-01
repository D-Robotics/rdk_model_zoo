import assert from 'node:assert/strict';
import { readFile, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { runInNewContext } from 'node:vm';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { selectTechnicalPassedEntries } from './technical-passed-selection.mjs';
import { comparableAccuracyRecords } from './accuracy-metrics.mjs';
import { loadVerifiedPreviewDownloads, previewDownloadAssets } from './preview-downloads.mjs';
import { loadInferenceTaskBanners } from './task-banner-assets.mjs';
import { loadVerifiedRebuildSelection } from './yolo26-repaired-selection.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const useStaging = process.argv.includes('--staging');
const output = resolve(root, useStaging ? '.dist-candidates-passed-next' : 'dist-candidates-passed');
const selection = JSON.parse(await readFile(resolve(output, 'technical-selection.json'), 'utf8'));
const snapshot = resolve(root, 'candidate-staging', selection.snapshot_id);
const evidencePath = resolve(snapshot, 'evidence-index.json');
const refreshPath = resolve(snapshot, 'refresh-audit.json');
const evidenceBytes = await readFile(evidencePath);
const evidence = JSON.parse(evidenceBytes.toString('utf8'));
const refresh = JSON.parse(await readFile(refreshPath, 'utf8'));
const auditPath = resolve(evidence.authoritative_audit.path);
const reviewPath = resolve(evidence.review_matrix.path);
const auditBytes = await readFile(auditPath);
const reviewBytes = await readFile(reviewPath);
const audit = JSON.parse(auditBytes.toString('utf8'));
const review = JSON.parse(reviewBytes.toString('utf8'));
const digest = async path => createHash('sha256').update(await readFile(path)).digest('hex');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const toPlainJson = value => JSON.parse(JSON.stringify(value));
const expectedSelection = selectTechnicalPassedEntries({
  audit, review, auditPath, auditSha: sha256(auditBytes),
});
const context = { window: {} };
const loadScript = async path => runInNewContext(await readFile(path, 'utf8'), context);
await loadScript(resolve(output, 'data.js'));
await loadScript(resolve(output, 'model-properties.js'));
await loadScript(resolve(output, 'reports/reports-data.js'));
const data = context.window.MODEL_DATA;
const properties = context.window.MODEL_PROPERTIES;
const reports = Array.from(context.window.OE_REPORTS);
const inventory = JSON.parse(await readFile(resolve(output, 'reports/inventory.json'), 'utf8'));
const models = data.models;
const released = models.filter(model => model.releaseStatus === 'released');
const releasedDetect = released.filter(model => model.catalogId.endsWith('/detect'));
const rebuilt = released.filter(model => ['pose', 'obb'].includes(model.catalogId.split('/').at(-1)));
const candidate = models.filter(model => model.releaseStatus === 'candidate');
const candidateCatalogPath = resolve(root, 'build', `candidate-${selection.snapshot_id}`, 'catalog.json');
const candidateCatalog = JSON.parse(await readFile(candidateCatalogPath, 'utf8'));
let rebuildSelection = null;
if (selection.rebuild_selection) {
  assert.equal(selection.rebuild_selection.kind, 'yolo26-repaired-release-selection');
  assert.equal(selection.rebuild_selection.status, 'ready-and-hash-verified');
  assert.equal(selection.rebuild_selection.base_snapshot_id, selection.snapshot_id);
  assert.equal(selection.rebuild_selection.base_authoritative_audit_sha256,
    selection.source_hashes.authoritative_audit_sha256);
  assert.equal(sha256(await readFile(selection.rebuild_selection.manifest_path)),
    selection.rebuild_selection.manifest_sha256);
  rebuildSelection = await loadVerifiedRebuildSelection({
    directory: resolve(selection.rebuild_selection.manifest_path, '..'),
    snapshotId: selection.snapshot_id,
    auditSha: selection.source_hashes.authoritative_audit_sha256,
  });
  assert.deepEqual(selection.rebuild_selection.rebuild_input_inventory,
    rebuildSelection.manifest.rebuild_input_inventory,
    'mixed preview must preserve the authorized four-build source inventory');
  assert.equal(selection.rebuild_selection.source_evidence_index_sha256,
    rebuildSelection.manifest.source_evidence_index_sha256,
    'mixed preview must preserve the full source evidence-index hash');
  const indexedRebuildRows = rebuildSelection.records.map(row => ({
    release_id: row.releaseId,
    model_id: row.model.id,
    build_id: row.record.build_id,
    artifact_sha256: row.model.assets[0].sha256,
    row_sha256: sha256(Buffer.from(JSON.stringify(row.model))),
    evidence_index: Object.fromEntries(Object.entries(row.evidence)
      .map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
    source_evidence_index: row.sourceEvidence,
    package_index: Object.fromEntries(Object.entries(row.files)
      .map(([role, ref]) => [role, { path: ref.path, sha256: sha256(ref.bytes) }])),
    upload_index: Object.fromEntries(Object.entries(row.uploads)
      .map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
    metric_bindings: row.record.metric_bindings,
  })).sort((left, right) => left.release_id.localeCompare(right.release_id));
  assert.deepEqual([...selection.rebuild_selection.releases]
    .sort((left, right) => left.release_id.localeCompare(right.release_id)), indexedRebuildRows,
  'rebuild selection must preserve each source, evidence, package, upload, and metric pointer');
  for (const rebuiltRecord of rebuildSelection.records) {
    const outputModel = models.find(model => model.id === rebuiltRecord.model.id);
    assert.deepEqual(toPlainJson(outputModel), toPlainJson(rebuiltRecord.model),
      `${rebuiltRecord.releaseId} rendered row differs from its hashed rebuild selection`);
  }
}
const presentationOverrides = JSON.parse(await readFile(resolve(root, 'release/presentation-overrides.json'), 'utf8'));
const inferenceTaskBanners = await loadInferenceTaskBanners({ webRoot: root, catalog: candidateCatalog });
if (inferenceTaskBanners) {
  assert.deepEqual(selection.inference_task_banners, inferenceTaskBanners.selection);
} else {
  assert.equal(Object.hasOwn(selection, 'inference_task_banners'), false);
}
const selectedReleaseIds = new Set(selection.selected.map(row => row.model_id));

assert.equal(selection.kind, 'technical-passed-candidate-preview-selection');
assert.equal(selection.snapshot_status, 'candidate-pending');
assert.equal(selection.snapshot_id, evidence.snapshot_id);
assert.equal(selection.frozen_snapshot_unchanged, true);
assert.deepEqual(selection.selected, expectedSelection.selected);
assert.deepEqual(selection.excluded, expectedSelection.excluded);
assert.equal(selection.counts.selected, expectedSelection.selected.length);
assert.equal(selection.counts.excluded, expectedSelection.excluded.length);
assert.deepEqual(selection.counts.by_task, Object.fromEntries(['cls', 'seg', 'pose', 'obb'].map(task =>
  [task, expectedSelection.selected.filter(row => row.task === task).length])));
assert.deepEqual(selection.counts.by_platform, Object.fromEntries(['s600', 's100p', 's100'].map(platform =>
  [platform, expectedSelection.selected.filter(row => row.platform === platform).length])));
assert.equal(selection.selected.length, 56);
assert.equal(selection.excluded.length, 4);
assert.deepEqual(selection.counts.by_task, { cls: 15, seg: 15, pose: 14, obb: 12 });
assert.equal(selection.source_hashes.evidence_index_sha256, sha256(evidenceBytes));
assert.equal(selection.source_hashes.authoritative_audit_sha256, sha256(auditBytes));
assert.equal(selection.source_hashes.review_matrix_sha256, sha256(reviewBytes));
assert.equal(selection.snapshot_sha256, sha256(evidenceBytes));
assert.equal(selection.audit_sha256, sha256(auditBytes));
assert.equal(selection.review_sha256, sha256(reviewBytes));
assert.equal(selection.source_files.authoritative_audit, auditPath);
assert.equal(selection.source_files.review_matrix, reviewPath);
assert.equal(selection.source_files.candidate_snapshot, selection.snapshot_id);
assert.equal(evidence.authoritative_audit.sha256, sha256(auditBytes));
assert.equal(evidence.review_matrix.sha256, sha256(reviewBytes));
assert.equal(refresh.authoritative_audit.sha256, sha256(auditBytes));
assert.equal(refresh.review_matrix.sha256, sha256(reviewBytes));

const verifiedDownloads = await loadVerifiedPreviewDownloads({
  webRoot: root,
  snapshotId: selection.snapshot_id,
  auditSha: sha256(auditBytes),
  catalog: candidateCatalog,
  selectedReleaseIds,
});
if (verifiedDownloads) {
  assert.deepEqual(selection.preview_downloads, verifiedDownloads.selection,
    'selection must bind each verified preview download manifest');
  assert.equal(selection.release_gates?.model_download_assets, 'verified-preview-downloads');
} else {
  assert.equal(Object.hasOwn(selection, 'preview_downloads'), false,
    'selection cannot claim verified downloads without a manifest');
  assert.equal(selection.release_gates?.model_download_assets, 'withheld');
}
assert.equal(selection.release_gates?.candidate_status, 'pending');
assert.equal(selection.release_gates?.formal_release_approval, 'pending');
assert.equal(selection.release_gates?.human_accuracy_review, 'pending');
assert.equal(selection.release_gates?.validation_json, 'pending');

assert.equal(data.candidatePreview, true);
assert.equal(data.technicalPassedPreview, false);
assert.equal(Object.hasOwn(data, 'technicalPassedPreviewNotice'), false);
assert.equal(data.catalog.status, 'mixed-preview');
assert.equal(data.release.status, 'mixed-preview');
assert.equal(rebuilt.length, rebuildSelection?.records.length || 0);
assert.equal(models.length, 40 + candidate.length + rebuilt.length);
assert.equal(releasedDetect.length, 40);
assert.equal(candidate.length, selection.selected.length);
assert.equal(new Set(models.map(model => model.id)).size, models.length);
const candidateDownloadCount = verifiedDownloads?.selection?.verified_models || 0;
const expectedDownloadCount = 40 + candidateDownloadCount + rebuilt.length;
assert.equal(candidate.reduce((sum, model) => sum + model.assets.length, 0), candidateDownloadCount);
assert.equal(models.reduce((sum, model) => sum + model.assets.length, 0), expectedDownloadCount);
assert.equal(data.catalog.summary.asset_count, expectedDownloadCount);
assert.equal(data.catalog.summary.downloadable_asset_count, expectedDownloadCount);
assert.ok(releasedDetect.every(model => model.taskId === 'object-detection'));
assert.deepEqual(Object.fromEntries(['yolo11', 'yolo26'].map(family =>
  [family, releasedDetect.filter(model => model.family === family).length])), { yolo11: 20, yolo26: 20 });
assert.ok(candidate.every(model => ['S600', 'S100P', 'S100'].includes(model.releasePlatform)));
const candidateCatalogReleases = new Map((candidateCatalog.models || []).flatMap(record =>
  (record.variants || []).flatMap(variant => (variant.platforms || []).map(release => [
    `${record.id}/${variant.size}/${release.platform.toLowerCase()}`,
    { record, variant, release },
  ]))));
const excludedReleaseIds = new Set(selection.excluded.map(row => row.model_id));
for (const model of candidate) {
  const task = model.catalogId.split('/').at(-1);
  const releaseId = `${model.catalogId}/${model.modelSize}/${model.releasePlatform.toLowerCase()}`;
  assert.equal(selectedReleaseIds.has(releaseId), true,
    `${releaseId} must be part of the selected technical rows`);
  assert.equal(excludedReleaseIds.has(releaseId), false,
    `${releaseId} must not be one of the four excluded anomaly rows`);
  const expectedAssets = previewDownloadAssets(verifiedDownloads, releaseId, model.releasePlatform);
  assert.deepEqual(toPlainJson(model.assets), expectedAssets,
    `candidate assets must match the verified download overlay: ${releaseId}`);
  const catalogRow = candidateCatalogReleases.get(releaseId);
  assert.ok(catalogRow, `candidate model is missing from the frozen catalog: ${releaseId}`);
  assert.equal(catalogRow.release.status, 'candidate');
  assert.equal(model.description, presentationOverrides.models[model.catalogId].zh);
  assert.equal(model.descriptionEn, presentationOverrides.models[model.catalogId].en);
  const inferenceCover = inferenceTaskBanners?.items.get(model.catalogId);
  if (inferenceCover) {
    assert.equal(model.coverLabel, inferenceCover.cover_label);
    assert.equal(await digest(resolve(output, model.coverImage)), inferenceCover.sha256);
  } else {
    assert.match(model.coverLabel, /示意图/);
    assert.match(model.coverLabel, /非模型推理结果/);
  }
  assert.equal(model.assets.length, expectedAssets.length,
    `${releaseId} needs one asset when covered by a manifest, otherwise none`);

  const samplePath = catalogRow.record.sample_path;
  const sourceCommit = model.sourceRepositoryCommit;
  assert.equal(sourceCommit, 'eed26ce610d7fba03a68d1c0ee6e62603cd9b85d',
    `${releaseId} must use the pinned public sample revision`);
  assert.equal(model.source,
    `https://github.com/D-Robotics/rdk_model_zoo/tree/${sourceCommit}/${samplePath}`,
    `${releaseId} primary source must link to its pinned RDK sample`);
  assert.equal(model.source.includes('huggingface.co'), false,
    `${releaseId} primary source cannot be the upstream checkpoint`);
  assert.equal(model.conversionRepositoryCommit || null,
    catalogRow.release.provenance?.workbench_repository_commit || null,
    `${releaseId} conversion source provenance must match catalog`);
  if (['cls', 'seg'].includes(task)) {
    assert.equal(model.upstreamWeightUrl, catalogRow.release.provenance?.source_weight_url,
      `${releaseId} must retain its exact upstream checkpoint URL separately`);
    assert.match(model.upstreamWeightUrl || '', /^https:\/\/huggingface\.co\//,
      `${releaseId} upstream checkpoint URL is missing`);
  } else {
    assert.equal(Object.hasOwn(model, 'upstreamWeightUrl'), false,
      `${releaseId} Pose/OBB checkpoint URL must remain hidden from detail links`);
    assert.deepEqual(toPlainJson(model.checkpointProvenance), {
      sourceWeightUrl: catalogRow.release.provenance?.source_weight_url || null,
      sourceCheckpointSha256: catalogRow.release.provenance?.source_checkpoint_sha256 || null,
    }, `${releaseId} must retain internal checkpoint provenance`);
  }

  const expectedAccuracy = comparableAccuracyRecords(task,
    catalogRow.release.accuracy?.float_onnx,
    catalogRow.release.accuracy?.runtime,
    {
      dataset: catalogRow.release.accuracy?.dataset,
      evaluation_scope: catalogRow.release.accuracy?.evaluation_scope,
    },
    catalogRow.release.accuracy?.comparison,
  );
  assert.deepEqual(toPlainJson(model.benchmark?.accuracy), expectedAccuracy,
    `${releaseId} must show exact catalog float/runtime values and comparison metadata`);
  for (const metric of model.benchmark.accuracy) {
    assert.equal(metric.dataset, catalogRow.release.accuracy.dataset,
      `${releaseId} accuracy row must retain its dataset`);
    assert.deepEqual(toPlainJson(metric.comparison_scope),
      catalogRow.release.accuracy.comparison?.comparison_scope,
      `${releaseId} accuracy row must retain its exact comparison scope`);
    if (catalogRow.release.accuracy.evaluation_scope) {
      assert.equal(metric.evaluation_scope, catalogRow.release.accuracy.evaluation_scope,
        `${releaseId} accuracy row must retain its evaluation scope`);
    }
  }
}
assert.equal(Object.keys(properties).length, models.length);
assert.equal(reports.length, models.length);
assert.equal(inventory.reports.length, models.length);
assert.equal(new Set(reports.map(report => report.id)).size, models.length);
assert.equal(new Set(inventory.reports.map(report => report.id)).size, models.length);

for (const row of selection.selected) {
  assert.ok(row.required_technical_checks.every(name => row.technical_checks_passed.includes(name),
    `${row.audit_name} is missing a required passing check`));
  assert.ok(row.pending_review_gates.length > 0, `${row.audit_name} must retain pending release gates`);
  assert.equal(row.validation_json_pending, true, `${row.audit_name} must retain pending validation.json state`);
  const model = candidate.find(item => item.catalogId === `ultralytics_yolo/yolo26/${row.task}`
    && item.modelSize === row.size && item.releasePlatform.toLowerCase() === row.platform);
  assert.ok(model, `missing candidate model for ${row.audit_name}`);
  assert.ok(model.reportDataUrl, `missing conversion report for ${row.audit_name}`);
  assert.ok(await stat(resolve(output, model.reportDataUrl)).then(details => details.isFile()));
}
for (const model of released) {
  assert.ok(model.assets.length, `published Detect download asset missing: ${model.id}`);
  assert.ok(model.assets.every(asset => asset.url && asset.sha256 && asset.sizeBytes > 0));
  if (model.reportDataUrl) assert.ok(await stat(resolve(output, model.reportDataUrl)).then(details => details.isFile()));
}
for (const model of rebuilt) {
  assert.equal(model.assets.length, 1, `${model.id} must expose one verified rebuilt HBM`);
  assert.ok(model.reportDataUrl, `${model.id} must link to its structured OE report`);
  assert.ok(await stat(resolve(output, model.reportDataUrl)).then(details => details.isFile()));
  assert.equal(model.description, presentationOverrides.models[model.catalogId].zh);
  assert.equal(model.descriptionEn, presentationOverrides.models[model.catalogId].en);
  const inferenceCover = inferenceTaskBanners.items.get(model.catalogId);
  assert.ok(inferenceCover, `${model.catalogId} must retain its verified task inference banner`);
  assert.equal(model.coverLabel, inferenceCover.cover_label);
  assert.equal(await digest(resolve(output, model.coverImage)), inferenceCover.sha256);
}

const composition = selection.preview_composition;
assert.equal(composition.kind, rebuildSelection
  ? 'published-detect-plus-technical-candidates-plus-repaired-releases'
  : 'published-detect-plus-technical-candidates');
assert.equal(composition.published_detect.count, 40);
assert.equal(composition.technical_candidates, candidate.length);
assert.equal(composition.repaired_releases || 0, rebuilt.length);
assert.equal(composition.combined_count, models.length);
assert.equal(composition.candidate_release_gates_unchanged, true);
assert.equal(composition.published_detect.catalog_sha256, await digest(resolve(root, 'build/catalog.json')));
assert.equal(composition.published_detect.inputs_sha256, await digest(resolve(root, 'release/inputs.json')));
assert.equal(composition.published_detect.generated_data_sha256, await digest(resolve(root, 'dist/data.js')));
assert.equal(composition.published_detect.model_properties_sha256, await digest(resolve(root, 'dist/model-properties.js')));
assert.equal(composition.published_detect.reports_registry_sha256, await digest(resolve(root, 'dist/reports/reports-data.js')));
assert.equal((await readdir(resolve(output, 'reports/data'))).length, models.length);

for (const file of ['src/app.js', 'src/detail-view.js']) {
  const source = await readFile(resolve(root, file), 'utf8');
  assert.doesNotMatch(source, /技术验证预览|Technical validation preview/);
}
assert.doesNotMatch(await readFile(resolve(output, 'data.js'), 'utf8'), /技术验证预览|Technical validation preview/);
console.log(`Mixed preview validated: 40 published YOLO11/YOLO26 Detect + ${candidate.length} technical candidates + ${rebuilt.length} finalized rebuild rows; ${models.length} reports, ${expectedDownloadCount} downloads, candidate gates pending, no technical banner.`);
