import { cp, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { basename, dirname, extname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  accuracyMetricLabels,
  accuracyRecords,
  boardAccuracyRecords,
  comparableAccuracyRecords,
} from './accuracy-metrics.mjs';
import { selectTechnicalPassedEntries } from './technical-passed-selection.mjs';
import { loadVerifiedPreviewDownloads, previewDownloadAssets } from './preview-downloads.mjs';
import { loadInferenceTaskBanners } from './task-banner-assets.mjs';

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptRoot, '..');
const args = process.argv.slice(2);
const candidatePreview = args.includes('--candidates');
const technicalPassedPreview = args.includes('--technical-passed');
if (technicalPassedPreview && !candidatePreview) {
  throw new Error('--technical-passed requires --candidates');
}
let requestedCandidateDir = null;
let requestedSnapshotDir = null;
let requestedAuditPath = null;
let requestedReviewPath = null;
let expectedSelectedCount = null;
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === '--candidates' || args[i] === '--technical-passed') continue;
  if (['--candidate-dir', '--snapshot-dir', '--audit-path', '--review-path', '--expected-selected-count'].includes(args[i])
      && args[i + 1]) {
    const option = args[i];
    const value = args[i + 1];
    if (option === '--candidate-dir') requestedCandidateDir = value;
    if (option === '--snapshot-dir') requestedSnapshotDir = value;
    if (option === '--audit-path') requestedAuditPath = value;
    if (option === '--review-path') requestedReviewPath = value;
    if (option === '--expected-selected-count') {
      expectedSelectedCount = Number(value);
      if (!Number.isInteger(expectedSelectedCount) || expectedSelectedCount < 1) {
        throw new Error('--expected-selected-count must be a positive integer');
      }
    }
    i += 1;
    continue;
  }
  throw new Error(`Unknown preview option: ${args[i]}`);
}
if (technicalPassedPreview && (!requestedSnapshotDir || !requestedAuditPath || !requestedReviewPath)) {
  throw new Error('--technical-passed requires explicit --snapshot-dir, --audit-path, and --review-path');
}
const candidateStagingRoot = resolve(webRoot, 'candidate-staging');
const candidateDirs = [];
for (const entry of candidatePreview ? await readdir(candidateStagingRoot, { withFileTypes: true }) : []) {
  if (!entry.isDirectory() || !/^yolo26-task-b8-audit-\d{8}T\d{6}Z(?:-r\d+)?$/.test(entry.name)) continue;
  try {
    const refreshAudit = JSON.parse(await readFile(resolve(candidateStagingRoot, entry.name, 'refresh-audit.json'), 'utf8'));
    if (refreshAudit.snapshot_id === entry.name && refreshAudit.snapshot_status === 'candidate-pending') {
      candidateDirs.push(entry.name);
    }
  } catch {
    // Ignore incomplete, invalid, or superseded historical snapshots.
  }
}
candidateDirs.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
let candidateDir;
let candidateRoot;
if (requestedSnapshotDir) {
  candidateRoot = isAbsolute(requestedSnapshotDir)
    ? resolve(requestedSnapshotDir)
    : resolve(candidateStagingRoot, requestedSnapshotDir);
  const snapshotRelative = relative(candidateStagingRoot, candidateRoot);
  if (!snapshotRelative || snapshotRelative.startsWith('..') || isAbsolute(snapshotRelative)) {
    throw new Error('--snapshot-dir must point to a child of candidate-staging');
  }
  candidateDir = basename(candidateRoot);
  if (requestedCandidateDir && requestedCandidateDir !== candidateDir) {
    throw new Error('--candidate-dir and --snapshot-dir identify different snapshots');
  }
} else {
  candidateDir = requestedCandidateDir || candidateDirs.at(-1) || 'yolo26-task-b8';
  if (candidateDir.includes('/') || candidateDir.includes('\\') || candidateDir === '.' || candidateDir === '..') {
    throw new Error('Candidate directory must be one candidate-staging child name');
  }
  candidateRoot = resolve(candidateStagingRoot, candidateDir);
}
if (candidatePreview) {
  const buildArgs = [resolve(scriptRoot, 'build_candidate_catalog.py'), '--snapshot-dir', candidateRoot];
  if (requestedAuditPath) buildArgs.push('--audit-path', resolve(requestedAuditPath));
  execFileSync('python3', buildArgs, { stdio: 'inherit' });
}
const outputRoot = technicalPassedPreview
  ? resolve(webRoot, '.dist-candidates-passed-next')
  : candidatePreview ? resolve(webRoot, 'dist-candidates') : resolve(webRoot, 'dist');
if (candidatePreview) process.env.MODEL_ZOO_OUTPUT_ROOT = outputRoot;
else delete process.env.MODEL_ZOO_OUTPUT_ROOT;
const catalogPath = candidatePreview
  ? resolve(webRoot, 'build', `candidate-${candidateDir}`, 'catalog.json')
  : process.env.MODEL_ZOO_CATALOG || resolve(webRoot, 'build', 'catalog.json');
const inputsPath = candidatePreview
  ? resolve(candidateRoot, 'release', 'inputs.json')
  : process.env.MODEL_ZOO_INPUTS || resolve(webRoot, 'release', 'inputs.json');
const presentationOverrides = JSON.parse(await readFile(
  resolve(webRoot, 'release', 'presentation-overrides.json'), 'utf8'));
const presentationOverrideIds = [
  'ultralytics_yolo/yolo26/cls',
  'ultralytics_yolo/yolo26/seg',
  'ultralytics_yolo/yolo26/pose',
  'ultralytics_yolo/yolo26/obb',
];
if (presentationOverrides.schema_version !== 1
    || !presentationOverrides.models || Array.isArray(presentationOverrides.models)
    || JSON.stringify(Object.keys(presentationOverrides.models).sort())
      !== JSON.stringify([...presentationOverrideIds].sort())) {
  throw new Error('release/presentation-overrides.json must define exactly the four YOLO26 task descriptions');
}
for (const id of presentationOverrideIds) {
  const description = presentationOverrides.models[id];
  if (!description || typeof description.zh !== 'string' || !description.zh.trim()
      || typeof description.en !== 'string' || !description.en.trim()) {
    throw new Error(`release/presentation-overrides.json has an invalid zh/en description for ${id}`);
  }
}
const repositoryUrl = process.env.MODEL_ZOO_REPOSITORY_URL || 'https://github.com/D-Robotics/rdk_model_zoo';
const candidateSampleReferenceCommit = 'eed26ce610d7fba03a68d1c0ee6e62603cd9b85d';
const officialRepositoryUrl = 'https://github.com/D-Robotics/rdk_model_zoo';

const fullCatalog = JSON.parse(await readFile(resolve(catalogPath), 'utf8'));
const expectedCatalogSource = candidatePreview
  ? `model_zoo_web/candidate-staging/${candidateDir}`
  : 'model_zoo_web/data';
if (fullCatalog.source !== expectedCatalogSource) {
  throw new Error(`Unexpected catalog source: ${fullCatalog.source || 'missing'}`);
}
if (candidatePreview ? fullCatalog.status !== 'candidate' : fullCatalog.status === 'candidate') {
  throw new Error(`Catalog status does not match ${candidatePreview ? 'candidate' : 'release'} preview mode`);
}

const sha256Bytes = bytes => createHash('sha256').update(bytes).digest('hex');
const expectedReleaseStatus = candidatePreview ? 'candidate' : 'released';
const candidateReleaseIds = new Set((fullCatalog.models || []).flatMap(record =>
  (record.variants || []).flatMap(variant => (variant.platforms || [])
    .filter(release => release.status === expectedReleaseStatus)
    .map(release => `${record.id}/${variant.size}/${release.platform}`))));
let catalog = fullCatalog;
let technicalSelection = null;
let selectedReleaseIds = null;
if (technicalPassedPreview) {
  const refreshAudit = JSON.parse(await readFile(resolve(candidateRoot, 'refresh-audit.json'), 'utf8'));
  const evidenceIndexBytes = await readFile(resolve(candidateRoot, 'evidence-index.json'));
  const evidenceIndex = JSON.parse(evidenceIndexBytes.toString('utf8'));
  const auditPath = resolve(requestedAuditPath);
  if (resolve(refreshAudit.authoritative_audit?.path || '') !== auditPath) {
    throw new Error('Explicit audit path does not match the immutable snapshot audit reference');
  }
  if (!auditPath || !isAbsolute(auditPath)) throw new Error('Snapshot does not bind an absolute authoritative audit path');
  const auditBytes = await readFile(auditPath);
  const auditSha = sha256Bytes(auditBytes);
  const evidenceSha = sha256Bytes(evidenceIndexBytes);
  if (refreshAudit.snapshot_id !== candidateDir || refreshAudit.snapshot_status !== 'candidate-pending'
      || evidenceIndex.snapshot_id !== candidateDir || evidenceIndex.snapshot_status !== 'candidate-pending') {
    throw new Error('Candidate snapshot identity/status is not candidate-pending and consistent');
  }
  if (refreshAudit.authoritative_audit.sha256 !== auditSha
      || evidenceIndex.authoritative_audit?.sha256 !== auditSha) {
    throw new Error('Authoritative audit SHA does not match both candidate snapshot manifests');
  }
  const reviewPath = resolve(requestedReviewPath);
  const reviewBytes = await readFile(reviewPath);
  const reviewSha = sha256Bytes(reviewBytes);
  const audit = JSON.parse(auditBytes.toString('utf8'));
  const review = JSON.parse(reviewBytes.toString('utf8'));
  const result = selectTechnicalPassedEntries({ audit, review, auditPath, auditSha });
  if (expectedSelectedCount !== null && result.selected.length !== expectedSelectedCount) {
    throw new Error(`Expected ${expectedSelectedCount} selected rows, found ${result.selected.length}`);
  }
  if (!result.selected.length) throw new Error('No technically passed rows remain after review exclusions');
  const auditIds = new Set(audit.entries.map(row => `ultralytics_yolo/yolo26/${row.task}/${row.size}/${row.platform}`));
  const indexedIds = new Set((evidenceIndex.records || []).map(record => record.id));
  if (auditIds.size !== indexedIds.size || [...auditIds].some(id => !indexedIds.has(id))) {
    throw new Error('Frozen snapshot evidence IDs do not exactly match authoritative audit rows');
  }
  if (review.snapshot_verification?.source_row_count !== undefined
      && review.snapshot_verification.source_row_count !== audit.entries.length) {
    throw new Error('Review matrix source row count does not match the audit');
  }
  for (const row of result.selected) {
    if (!indexedIds.has(row.model_id)) throw new Error(`Selected audit row is absent from evidence index: ${row.model_id}`);
  }
  selectedReleaseIds = new Set(result.selected.map(row => row.model_id));
  const availableSelected = new Set();
  const selectedModels = (fullCatalog.models || []).flatMap(record => {
    const variants = (record.variants || []).map(variant => {
      const platforms = (variant.platforms || []).filter(release => {
        const id = `${record.id}/${variant.size}/${release.platform}`;
        if (selectedReleaseIds.has(id) && release.status === 'candidate') availableSelected.add(id);
        return selectedReleaseIds.has(id) && release.status === 'candidate';
      });
      return { ...variant, platforms };
    }).filter(variant => variant.platforms.length);
    return variants.length ? [{ ...record, variants }] : [];
  });
  if (availableSelected.size !== result.selected.length) {
    throw new Error('Selected technical rows do not all bind to candidate catalog releases');
  }
  catalog = { ...fullCatalog, models: selectedModels };
  const byTask = Object.fromEntries(['cls', 'seg', 'pose', 'obb']
    .map(task => [task, result.selected.filter(row => row.task === task).length]));
  technicalSelection = {
    schema_version: 1,
    kind: 'technical-passed-candidate-preview-selection',
    snapshot_id: candidateDir,
    snapshot_status: 'candidate-pending',
    frozen_snapshot_unchanged: true,
    source_hashes: {
      authoritative_audit_sha256: auditSha,
      review_matrix_sha256: reviewSha,
      evidence_index_sha256: evidenceSha,
    },
    snapshot_sha256: evidenceSha,
    audit_sha256: auditSha,
    review_sha256: reviewSha,
    source_files: {
      authoritative_audit: auditPath,
      review_matrix: reviewPath,
      candidate_snapshot: candidateDir,
    },
    rule: 'Include any platform row only when all required nonhuman technical audit checks are present and passed and the review matrix reports no missing technical evidence or anomaly.',
    selected: result.selected,
    excluded: result.excluded,
    counts: {
      selected: result.selected.length,
      excluded: result.excluded.length,
      by_task: byTask,
      by_platform: Object.fromEntries(['s600', 's100p', 's100']
        .map(platform => [platform, result.selected.filter(row => row.platform.toLowerCase() === platform).length])),
    },
    release_gates: {
      candidate_status: 'pending',
      formal_release_approval: 'pending',
      human_accuracy_review: 'pending',
      validation_json: 'pending',
      model_download_assets: 'withheld',
    },
  };
}

const previewDownloads = technicalPassedPreview
  ? await loadVerifiedPreviewDownloads({ webRoot, snapshotId: candidateDir,
    auditSha: technicalSelection.audit_sha256, catalog })
  : null;
const releasedTaskBanners = !candidatePreview && presentationOverrideIds.every(id =>
  fullCatalog.models.some(model => model.id === id));
const inferenceTaskBanners = technicalPassedPreview || releasedTaskBanners
  ? await loadInferenceTaskBanners({ webRoot, catalog: fullCatalog }) : null;
if (inferenceTaskBanners && technicalSelection) {
  technicalSelection.inference_task_banners = inferenceTaskBanners.selection;
}
if (previewDownloads) {
  technicalSelection.preview_downloads = previewDownloads.selection;
  technicalSelection.release_gates.model_download_assets = 'verified-preview-downloads';
}

const resolvedInputsPath = resolve(inputsPath);
const inputRoot = dirname(resolvedInputsPath);
const inputs = JSON.parse(await readFile(resolvedInputsPath, 'utf8'));
if (inputs.schema_version !== 1) {
  throw new Error(`Unsupported MODEL_ZOO_INPUTS schema_version: ${inputs.schema_version || 'missing'}`);
}
if (!inputs.models || typeof inputs.models !== 'object' || Array.isArray(inputs.models)) {
  throw new Error('MODEL_ZOO_INPUTS.models must be an object keyed by catalog model id');
}
if (!inputs.releases || typeof inputs.releases !== 'object' || Array.isArray(inputs.releases)) {
  throw new Error('MODEL_ZOO_INPUTS.releases must be an object keyed by release id');
}

// Defer generate.mjs until provenance and selection preflight has passed. It
// recreates only the selected output directory on import.
await import('./generate.mjs');

const resolveInputFile = async (value, label) => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty path`);
  const path = isAbsolute(value) ? value : resolve(inputRoot, value);
  const details = await stat(path).catch(() => null);
  if (!details?.isFile()) throw new Error(`${label} does not exist or is not a file: ${path}`);
  return path;
};
const oeDataCache = new Map();
const loadOeData = async (value, label) => {
  const path = await resolveInputFile(value, label);
  if (oeDataCache.has(path)) return oeDataCache.get(path);
  const oeData = JSON.parse(await readFile(path, 'utf8'));
  if (oeData.schema_version !== 1) {
    throw new Error(`${label} has unsupported schema_version: ${oeData.schema_version}`);
  }
  if (!oeData.provenance?.artifact_sha256) {
    throw new Error(`${label} is missing provenance.artifact_sha256`);
  }
  oeDataCache.set(path, oeData);
  return oeData;
};
const usedModelInputs = new Set();
const usedReleaseInputs = new Set();

const taskMetadata = {
  detect: { id: 'object-detection', label: '目标检测', description: 'COCO 目标检测模型' },
  classify: { id: 'image-classification', label: '图像分类', description: '图像分类模型' },
  cls: { id: 'image-classification', label: '图像分类', description: '图像分类模型' },
  segment: { id: 'instance-segmentation', label: '实例分割', description: '实例分割模型' },
  seg: { id: 'instance-segmentation', label: '实例分割', description: '实例分割模型' },
  pose: { id: 'pose-estimation', label: '姿态估计', description: '人体姿态估计模型' },
  obb: { id: 'object-detection', label: '旋转框检测', description: '旋转框目标检测模型' },
};

const safeId = (...parts) => parts.join('-').toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
const displayNames = (family, size, task) => {
  const familyName = family.replace(/^yolo/i, 'YOLO');
  // Acronym tasks keep their capitals: "YOLO26 OBB", not "YOLO26 Obb".
  const taskName = { detect: 'Detect', obb: 'OBB' }[task] || task[0].toUpperCase() + task.slice(1);
  return {
    model: `${familyName} ${taskName}`,
    variant: `${familyName}${size} ${taskName}`,
  };
};
const models = [];
const properties = {};
const reportEntries = [];
const assetsRoot = resolve(outputRoot, 'assets', 'models');
const reportsRoot = resolve(outputRoot, 'reports', 'models');
await mkdir(assetsRoot, { recursive: true });
await mkdir(reportsRoot, { recursive: true });

for (const record of catalog.models || []) {
  const modelInput = inputs.models[record.id];
  if (!modelInput || typeof modelInput !== 'object' || Array.isArray(modelInput)) {
    throw new Error(`MODEL_ZOO_INPUTS.models is missing ${record.id}`);
  }
  usedModelInputs.add(record.id);
  const inferenceCover = inferenceTaskBanners?.items.get(record.id);
  const coverPath = inferenceCover?.path
    || await resolveInputFile(modelInput.cover, `MODEL_ZOO_INPUTS.models.${record.id}.cover`);
  const coverExtension = extname(coverPath).toLowerCase();
  if (!['.jpg', '.jpeg', '.png', '.webp', '.svg'].includes(coverExtension)) {
    throw new Error(`Unsupported cover image extension for ${record.id}: ${coverExtension || 'missing'}`);
  }
  const normalizedCoverExtension = coverExtension === '.jpeg' ? '.jpg' : coverExtension;
  const coverFilename = `${safeId(record.source, record.family, record.task)}${normalizedCoverExtension}`;
  await cp(coverPath, resolve(assetsRoot, coverFilename));

  const task = taskMetadata[record.task] || {
    id: record.task,
    label: record.task,
    description: `${record.family} ${record.task} model`,
  };
  for (const variant of record.variants || []) {
    for (const release of variant.platforms || []) {
      if (candidatePreview ? release.status !== 'candidate' : release.status !== 'released') continue;
      const platform = String(release.platform).toUpperCase();
      const id = safeId(record.source, record.family, record.task, variant.size, release.platform);
      const releaseId = `${record.id}/${variant.size}/${release.platform}`;
      const names = displayNames(record.family, variant.size, record.task);

      const artifact = release.artifact;
      const clsSegCandidate = candidatePreview && ['cls', 'seg'].includes(record.task);
      const poseObbTechnicalCandidate = technicalPassedPreview && ['pose', 'obb'].includes(record.task);
      const sampleLinkedCandidate = clsSegCandidate || poseObbTechnicalCandidate;
      const conversionRepositoryCommit = release.provenance?.workbench_repository_commit || null;
      const sourceRepositoryCommit = sampleLinkedCandidate ? candidateSampleReferenceCommit : null;
      const releasedDevelopTask = !candidatePreview && record.family === 'yolo26'
        && ['cls', 'seg', 'pose', 'obb'].includes(record.task);
      const sourceUrl = candidatePreview
        ? sampleLinkedCandidate
          ? `${officialRepositoryUrl}/tree/${sourceRepositoryCommit}/${record.sample_path}`
          : release.provenance?.source_weight_url || null
        : `${repositoryUrl}/tree/${releasedDevelopTask ? 'develop' : release.provenance.repository_commit}/${record.sample_path}`;
      const floatAccuracy = release.accuracy?.float_onnx;
      const runtimeAccuracy = release.accuracy?.runtime;
      const performance = release.performance || {};
      const measurements = performance.measurements || [];
      const endToEndSource = performance.end_to_end;
      const endToEndRecords = Array.isArray(endToEndSource)
        ? endToEndSource
        : endToEndSource ? [endToEndSource] : [];
      const endToEndMetric = (endToEnd, name) => {
        const summary = endToEnd?.metrics_ms?.[name];
        if (!summary || !Number.isFinite(Number(summary.mean))) return null;
        return {
          value: Number(summary.mean),
          unit: 'ms',
          p50: Number(summary.p50),
          p95: Number(summary.p95),
          min: Number(summary.min),
          max: Number(summary.max),
        };
      };
      const reportSourceUrl = release.reports?.oe_conversion_url;
      const releaseInput = inputs.releases[releaseId];
      let oeData;
      if (releaseInput !== undefined) {
        if (!releaseInput || typeof releaseInput !== 'object' || Array.isArray(releaseInput)) {
          throw new Error(`MODEL_ZOO_INPUTS.releases.${releaseId} must be an object`);
        }
        usedReleaseInputs.add(releaseId);
        oeData = await loadOeData(
          releaseInput.oe_data,
          `MODEL_ZOO_INPUTS.releases.${releaseId}.oe_data`,
        );
        if (oeData.provenance.artifact_sha256 !== artifact.sha256) {
          throw new Error(
            `${releaseId}: OE data artifact sha256 does not match the catalog artifact`,
          );
        }
      }
      let reportUrl;
      let reportDataUrl;
      if (oeData) {
        // Structured path: ship the small extracted JSON. Full HTML payloads
        // stay out of dist and are only retained as a report fallback source.
        const dataRoot = resolve(outputRoot, 'reports', 'data');
        await mkdir(dataRoot, { recursive: true });
        const dataFilename = `${id}-oe-data.json`;
        const dataBytes = JSON.stringify(oeData, null, 2);
        await writeFile(resolve(dataRoot, dataFilename), dataBytes, 'utf8');
        reportDataUrl = `reports/data/${dataFilename}`;
      } else if (reportSourceUrl) {
        const response = await fetch(reportSourceUrl);
        if (!response.ok) throw new Error(`Unable to fetch OE report: ${response.status} ${reportSourceUrl}`);
        const reportBytes = Buffer.from(await response.arrayBuffer());
        if (!/<(?:!doctype\s+html|html)[\s>]/i.test(reportBytes.subarray(0, 512).toString('utf8'))) {
          throw new Error(`OE report is not an HTML document: ${reportSourceUrl}`);
        }
        const reportFilename = `${id}-oe-report.html`;
        await writeFile(resolve(reportsRoot, reportFilename), reportBytes);
        reportUrl = `reports/models/${reportFilename}`;
      }
      if (reportDataUrl || reportUrl) {
        reportEntries.push({
          id,
          // The library lists one report per platform release; carry the
          // platform in the name so sibling entries stay distinguishable.
          name: `${names.variant} (${platform})`,
          hardware: platform,
          march: artifact.march,
          kind: 'conversion',
          bytes: reportDataUrl
            ? Buffer.byteLength(JSON.stringify(oeData), 'utf8')
            : 0,
          path: reportDataUrl ? reportSourceUrl || reportDataUrl : reportUrl,
          dataUrl: reportDataUrl,
          modelIds: [id],
          sources: [oeData?.generated_from_run].filter(Boolean),
        });
      }
      const accuracyMetadata = {
        dataset: release.accuracy?.dataset,
        evaluation_scope: release.accuracy?.evaluation_scope,
      };
      const comparison = release.accuracy?.comparison;
      const comparisonEvidenceValid = comparison?.status === 'valid-evidence'
        && comparison?.direct_metric_comparison === 'valid'
        && comparison?.comparison_scope?.comparable === true;
      const stagedAccuracyRecords = candidatePreview
        ? technicalPassedPreview
          && ['cls', 'seg', 'pose', 'obb'].includes(record.task)
          && comparisonEvidenceValid
          ? comparableAccuracyRecords(
            record.task,
            floatAccuracy,
            runtimeAccuracy,
            accuracyMetadata,
            comparison,
          )
          : boardAccuracyRecords(record.task, release.accuracy?.board_runtime, accuracyMetadata)
        : accuracyRecords(
          record.task,
          floatAccuracy,
          runtimeAccuracy,
          accuracyMetadata,
        );

      models.push({
        id,
        catalogId: record.id,
        name: names.model,
        variantName: names.variant,
        family: record.family,
        modelSize: variant.size,
        releaseStatus: release.status,
        releasePlatform: platform,
        task: task.label,
        taskId: task.id,
        tasks: [task.id],
        description: presentationOverrides.models[record.id]?.zh
          || record.description?.zh || task.description,
        descriptionEn: presentationOverrides.models[record.id]?.en
          || record.description?.en || `${record.family} ${record.task} model.`,
        platforms: [platform],
        coverImage: `assets/models/${coverFilename}`,
        coverLabel: inferenceCover?.cover_label || modelInput.cover_label || '模型参考推理结果',
        sample: `${record.source}/${record.family}/${record.task}/${variant.size}`,
        source: sourceUrl,
        ...(clsSegCandidate ? {
          upstreamWeightUrl: release.provenance?.source_weight_url || null,
          sourceRepositoryCommit,
          ...(conversionRepositoryCommit ? { conversionRepositoryCommit } : {}),
        } : {}),
        ...(poseObbTechnicalCandidate ? {
          sourceRepositoryCommit,
          ...(conversionRepositoryCommit ? { conversionRepositoryCommit } : {}),
          checkpointProvenance: {
            sourceWeightUrl: release.provenance?.source_weight_url || null,
            sourceCheckpointSha256: release.provenance?.source_checkpoint_sha256 || null,
          },
        } : {}),
        licenseName: record.license?.name,
        licenseUrl: record.license?.url,
        shape: (variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width]).join(' × '),
        reportUrl,
        reportDataUrl,
        reportSourceUrl,
        assets: candidatePreview ? previewDownloadAssets(previewDownloads, releaseId, platform) : [
          {
            role: `${platform} 部署模型`,
            format: artifact.format,
            filename: basename(new URL(artifact.url).pathname),
            url: artifact.url,
            sha256: artifact.sha256,
            sizeBytes: Number(artifact.size_bytes),
          },
        ],
        benchmark: {
          precision: release.provenance?.precision_policy || 'int8',
          ...(release.provenance?.precision_policy
            ? { precisionPolicy: release.provenance.precision_policy } : {}),
          environment: { hardware: `RDK ${platform}` },
          timing: {
            scope: performance.timing_scope,
            tool: performance.tool,
            implementation: performance.implementation,
            threadSemantics: performance.thread_semantics,
            stages: performance.stages,
            runsPerCondition: Number(performance.runs_per_condition),
            framesPerRun: Number(performance.frames_per_run),
            warmupFramesPerCondition: Number(performance.warmup_frames_per_condition),
          },
          endToEnd: endToEndRecords.map(endToEnd => ({
            timing: {
              scope: endToEnd.timing_scope,
              tool: endToEnd.tool,
              implementation: endToEnd.implementation,
              pipelineStreams: Number(endToEnd.pipeline_streams),
              runtimeSubmissionThreads: Number(endToEnd.runtime_submission_threads),
              cpuThreadPolicy: endToEnd.cpu_thread_policy,
              onlineCpuThreads: Number(endToEnd.online_cpu_threads),
              opencvThreads: Number(endToEnd.opencv_threads),
              cpuGovernor: endToEnd.cpu_governor,
              cpuFrequencyMhz: Number(endToEnd.cpu_frequency_mhz),
              bpuFrequencyMhz: Number(endToEnd.bpu_frequency_mhz),
              warmupFramesPerRound: Number(endToEnd.warmup_frames_per_round),
              rounds: Number(endToEnd.rounds),
              framesPerRound: Number(endToEnd.frames_per_round),
              timedFrames: Number(endToEnd.timed_frames),
              aggregateWallMs: Number(endToEnd.aggregate_wall_ms),
            },
            metrics: {
              preprocess: endToEndMetric(endToEnd, 'preprocess'),
              runtime: endToEndMetric(endToEnd, 'runtime'),
              postprocess: endToEndMetric(endToEnd, 'postprocess'),
              endToEnd: endToEndMetric(endToEnd, 'end_to_end'),
            },
            throughputFps: Number(endToEnd.throughput_fps),
          })),
          performance: measurements.flatMap(measurement => [
            {
              metric: 'latency',
              value: Number(measurement.average_latency_ms),
              unit: 'ms',
              concurrency: Number(measurement.threads),
              observedMin: Number(measurement.observed_min_latency_ms),
              observedMax: Number(measurement.observed_max_latency_ms),
            },
            {
              metric: 'throughput',
              value: Number(measurement.aggregate_fps),
              unit: 'fps',
              concurrency: Number(measurement.threads),
            },
          ]),
          accuracy: stagedAccuracyRecords,
        },
      });
      properties[id] = {
        parameterCount: Number(variant.model?.parameter_count),
        gflops: Number(variant.model?.gflops),
        inputShape: variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width],
        sourceInput: variant.input.source,
        runtimeInput: {
          format: artifact.runtime_input,
          resolution: [variant.input.height, variant.input.width],
        },
      };
    }
  }
}

if (!models.length) throw new Error('The samples-only catalog contains no released models');
const knownInputModelIds = new Set((fullCatalog.models || []).map(record => record.id));
for (const modelId of Object.keys(inputs.models)) {
  if (!knownInputModelIds.has(modelId)) throw new Error(`MODEL_ZOO_INPUTS references unknown model ${modelId}`);
  if (!technicalPassedPreview && !usedModelInputs.has(modelId)) {
    throw new Error(`MODEL_ZOO_INPUTS references unknown model ${modelId}`);
  }
}
for (const releaseId of Object.keys(inputs.releases)) {
  if (!candidateReleaseIds.has(releaseId)) throw new Error(`MODEL_ZOO_INPUTS references unknown release ${releaseId}`);
  if (!technicalPassedPreview && !usedReleaseInputs.has(releaseId)) {
    throw new Error(`MODEL_ZOO_INPUTS references unknown release ${releaseId}`);
  }
}
if (technicalPassedPreview) {
  for (const releaseId of selectedReleaseIds) {
    if (!usedReleaseInputs.has(releaseId)) throw new Error(`MODEL_ZOO_INPUTS is missing selected release ${releaseId}`);
  }
  if (usedReleaseInputs.size !== selectedReleaseIds.size) {
    throw new Error('Preview loaded a release outside the technical-passed selection');
  }
}

const platformNames = [...new Set(models.flatMap(model => model.platforms))];
const data = {
  schemaVersion: 1,
  catalog: {
    status: candidatePreview ? 'candidate' : 'published',
    summary: {
      sample_count: new Set(models.map(model => model.sample)).size,
      asset_count: models.reduce((count, model) => count + model.assets.length, 0),
      downloadable_asset_count: models.reduce((count, model) => count + model.assets.length, 0),
      benchmark_count: models.filter(model => model.benchmark).length,
    },
  },
  repository: { url: repositoryUrl },
  accuracyMetricLabels: accuracyMetricLabels(),
  release: {
    status: candidatePreview ? 'candidate' : 'published',
    compatibility: { hardware: platformNames.map(name => `RDK ${name}`).join(' / ') },
  },
  candidatePreview,
  technicalPassedPreview: false,
  models,
};

if (technicalSelection) {
  await writeFile(resolve(outputRoot, 'technical-selection.json'), `${JSON.stringify(technicalSelection, null, 2)}\n`, 'utf8');
}

await writeFile(resolve(outputRoot, 'data.js'), `window.MODEL_DATA = ${JSON.stringify(data, null, 2)};\n`, 'utf8');
await writeFile(
  resolve(outputRoot, 'model-properties.js'),
  `window.MODEL_PROPERTIES = Object.freeze(${JSON.stringify(properties, null, 2)});\n`,
  'utf8',
);
await writeFile(
  resolve(outputRoot, 'reports', 'reports-data.js'),
  `window.OE_REPORTS = ${JSON.stringify(reportEntries, null, 2)};\n`,
  'utf8',
);
await writeFile(
  resolve(outputRoot, 'reports', 'inventory.json'),
  `${JSON.stringify({ reports: reportEntries.map(entry => ({ id: entry.id, kind: entry.kind })) }, null, 2)}\n`,
  'utf8',
);

console.log(`Built local Model Zoo ${candidatePreview ? 'candidate preview' : 'preview'} with ${models.length} ${candidatePreview ? 'candidate' : 'released'} model entries.`);
