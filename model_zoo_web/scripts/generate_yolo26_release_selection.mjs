import assert from 'node:assert/strict';
import { copyFile, lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { basename, dirname, extname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { accuracyMetricLabels, comparableAccuracyRecords } from './accuracy-metrics.mjs';
import { loadInferenceTaskBanners } from './task-banner-assets.mjs';
import {
  expectedRebuildReleaseIds,
  loadVerifiedRebuildSelection,
  validateObbPrecisionEvidence,
  validateSingleRebuildRecord,
} from './yolo26-repaired-selection.mjs';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultWorkbench = '/home/zhengyi/work/rdk_model_zoo_workbench';
const defaultPreview = resolve(webRoot, 'dist-candidates-passed');
const defaultCatalog = snapshotId => resolve(webRoot, 'build', `candidate-${snapshotId}`, 'catalog.json');
const OSS_BASE = 'https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com';
const HASH_RE = /^[a-f0-9]{64}$/;
const RELEASE_TARGETS = [
  { name: 'pose-s-s100p', releaseId: 'ultralytics_yolo/yolo26/pose/s/s100p' },
  { name: 'obb-x-s600', releaseId: 'ultralytics_yolo/yolo26/obb/x/s600' },
  { name: 'obb-x-s100p', releaseId: 'ultralytics_yolo/yolo26/obb/x/s100p' },
  { name: 'obb-x-s100', releaseId: 'ultralytics_yolo/yolo26/obb/x/s100' },
];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const plain = value => JSON.parse(JSON.stringify(value));

function parseArgs(argv) {
  const result = { workbenchRoot: defaultWorkbench, previewDir: defaultPreview };
  const supplied = new Set();
  for (let i = 0; i < argv.length; i += 1) {
    const option = argv[i];
    if (option === '--help' || option === '-h') {
      result.help = true;
      continue;
    }
    if (option === '--validate-pose-only') {
      if (result.validatePoseOnly) throw new Error(`${option} may be supplied once`);
      result.validatePoseOnly = true;
      continue;
    }
    if (option === '--validate-obb-manifest-only') {
      if (result.validateObbManifestOnly) throw new Error(`${option} may be supplied once`);
      result.validateObbManifestOnly = true;
      continue;
    }
    if (!['--output-dir', '--workbench-root', '--preview-dir', '--catalog', '--obb-runs-index', '--obb-runs-manifest', '--targets'].includes(option)
        || !argv[i + 1] || argv[i + 1].startsWith('--')) {
      throw new Error(`Unknown or incomplete option: ${option}`);
    }
    const key = ({
      '--output-dir': 'outputDir', '--workbench-root': 'workbenchRoot',
      '--preview-dir': 'previewDir', '--catalog': 'catalogPath',
      '--obb-runs-index': 'obbRunsManifestPath', '--obb-runs-manifest': 'obbRunsManifestPath',
      '--targets': 'targetNames',
    })[option];
    if (supplied.has(key)) throw new Error(`${option} may be supplied once`);
    supplied.add(key);
    result[key] = argv[++i];
  }
  if (!result.help && !result.validatePoseOnly && !result.validateObbManifestOnly && !result.outputDir) {
    throw new Error('--output-dir is required');
  }
  if ((result.validatePoseOnly || result.validateObbManifestOnly) && result.outputDir) {
    throw new Error('read-only validation options cannot be combined with --output-dir');
  }
  if (result.validatePoseOnly && result.validateObbManifestOnly) {
    throw new Error('choose one read-only validation mode');
  }
  const targetNames = result.targetNames
    ? result.targetNames.split(',').map(item => item.trim()).filter(Boolean)
    : RELEASE_TARGETS.map(item => item.name);
  if (!targetNames.length || new Set(targetNames).size !== targetNames.length) {
    throw new Error('--targets must contain one or more unique target names');
  }
  const unknownTargets = targetNames.filter(name => !RELEASE_TARGETS.some(target => target.name === name));
  if (unknownTargets.length) throw new Error(`Unsupported --targets value(s): ${unknownTargets.join(', ')}`);
  result.targets = RELEASE_TARGETS.filter(target => targetNames.includes(target.name));
  if (result.validatePoseOnly && !result.targetNames) {
    result.targets = RELEASE_TARGETS.filter(target => target.name === 'pose-s-s100p');
  } else if (result.validatePoseOnly && result.targets.map(target => target.name).join(',') !== 'pose-s-s100p') {
    throw new Error('--validate-pose-only requires --targets pose-s-s100p or no --targets option');
  }
  if (result.validateObbManifestOnly && !result.targets.some(target => target.releaseId.includes('/obb/'))) {
    throw new Error('--validate-obb-manifest-only requires at least one OBB target');
  }
  for (const key of ['outputDir', 'workbenchRoot', 'previewDir', 'catalogPath', 'obbRunsManifestPath']) {
    if (result[key] && !isAbsolute(result[key])) throw new Error(`${key} must be an absolute path`);
  }
  return result;
}

function usage() {
  return [
    'Generate a hash-verified YOLO26 Pose/OBB release selection for one or more authorized targets.',
    '',
    'Usage:',
    '  node scripts/generate_yolo26_release_selection.mjs --output-dir /absolute/new/path',
    '  node scripts/generate_yolo26_release_selection.mjs --targets pose-s-s100p --output-dir /absolute/new/path',
    '  node scripts/generate_yolo26_release_selection.mjs --validate-pose-only',
    '  node scripts/generate_yolo26_release_selection.mjs --validate-obb-manifest-only',
    '',
    'Options:',
    `  --workbench-root ${defaultWorkbench}`,
    `  --preview-dir ${defaultPreview}`,
    '  --catalog /absolute/path/to/frozen-candidate-catalog.json',
    '  --obb-runs-manifest /absolute/path/to/final-obb-rebuild-runs.json (default: integer_runs.json)',
    `  --targets ${RELEASE_TARGETS.map(target => target.name).join(',')} (default: all four targets; comma-separated subset allowed)`,
    '  --validate-pose-only  Validate the finalized Pose-s/S100P rebuild in /tmp only; does not read OBB inventory.',
    '  --validate-obb-manifest-only  Verify the three per-board build/campaign/receipt bindings without creating a preview.',
    '',
    'The command refuses to overwrite an existing output directory and requires',
    'all four finalized bundles plus public-read OSS receipts for HBM, both OE',
    'reports, release.json, and SHA256SUMS.',
  ].join('\n');
}

function fail(condition, message) {
  assert.ok(condition, `YOLO26 rebuild selection: ${message}`);
}

async function regularFile(path, label) {
  fail(typeof path === 'string' && isAbsolute(path), `${label} path must be absolute`);
  const details = await lstat(path);
  fail(details.isFile() && !details.isSymbolicLink(), `${label} must be a regular non-symlink file: ${path}`);
  return readFile(path);
}

async function verifiedFile(path, expectedSha, label) {
  fail(HASH_RE.test(expectedSha || ''), `${label} SHA-256 is missing or malformed`);
  const bytes = await regularFile(path, label);
  fail(sha256(bytes) === expectedSha, `${label} SHA-256 mismatch`);
  return { path, sha256: expectedSha, bytes };
}

async function verifiedJson(path, expectedSha, label) {
  const file = await verifiedFile(path, expectedSha, label);
  let json;
  try {
    json = JSON.parse(file.bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`YOLO26 rebuild selection: ${label} is not valid JSON: ${error.message}`);
  }
  fail(json && typeof json === 'object' && !Array.isArray(json), `${label} must contain a JSON object`);
  return { ...file, json };
}

async function readLocalJson(path, label) {
  const bytes = await regularFile(path, label);
  let json;
  try {
    json = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`YOLO26 rebuild selection: ${label} is not valid JSON: ${error.message}`);
  }
  return { path, sha256: sha256(bytes), bytes, json };
}

function jsonPointer(value, pointer, label) {
  fail(typeof pointer === 'string' && (pointer === '' || pointer.startsWith('/')),
    `${label} must be an RFC 6901 JSON pointer`);
  if (!pointer) return value;
  return pointer.slice(1).split('/').map(token => token.replace(/~1/g, '/').replace(/~0/g, '~'))
    .reduce((current, key) => current?.[key], value);
}

function safeId(...parts) {
  return parts.join('-').toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
}

function taskUi(task) {
  return {
    pose: { id: 'pose-estimation', label: '姿态估计' },
    obb: { id: 'object-detection', label: '旋转框检测' },
  }[task];
}

function verifyFrozenSnapshot(selection, snapshotId, auditSha, reviewSha, evidenceSha) {
  fail(selection.kind === 'technical-passed-candidate-preview-selection', 'preview selection kind is unexpected');
  fail(selection.snapshot_id === snapshotId && selection.snapshot_status === 'candidate-pending',
    'preview selection snapshot identity/status is unexpected');
  fail(selection.source_hashes?.authoritative_audit_sha256 === auditSha,
    'authoritative audit SHA does not match the current preview selection');
  fail(selection.source_hashes?.review_matrix_sha256 === reviewSha,
    'review matrix SHA does not match the current preview selection');
  fail(selection.source_hashes?.evidence_index_sha256 === evidenceSha,
    'candidate evidence-index SHA does not match the current preview selection');
  fail(selection.selected?.length === 56 && selection.excluded?.length === 4,
    'expected the frozen 56-selected / 4-excluded review split');
  const expected = expectedRebuildReleaseIds();
  const observed = selection.excluded.map(row => row.model_id).sort();
  fail(JSON.stringify(observed) === JSON.stringify(expected),
    'the four excluded rows differ from the Pose-s S100P and OBB-x S600/P/100 repair set');
}

async function verifySnapshotInputs(selection, webRoot) {
  const snapshotId = selection.snapshot_id;
  const snapshotRoot = resolve(webRoot, 'candidate-staging', snapshotId);
  const evidencePath = resolve(snapshotRoot, 'evidence-index.json');
  const refreshPath = resolve(snapshotRoot, 'refresh-audit.json');
  const evidenceFile = await readLocalJson(evidencePath, 'candidate evidence index');
  const refreshFile = await readLocalJson(refreshPath, 'candidate refresh audit');
  const auditPath = resolve(selection.source_files?.authoritative_audit || '');
  const reviewPath = resolve(selection.source_files?.review_matrix || '');
  fail(isAbsolute(selection.source_files?.authoritative_audit || '')
    && isAbsolute(selection.source_files?.review_matrix || ''), 'frozen audit/review paths must be absolute');
  const auditFile = await readLocalJson(auditPath, 'authoritative audit');
  const reviewFile = await readLocalJson(reviewPath, 'review matrix');
  verifyFrozenSnapshot(selection, snapshotId, auditFile.sha256, reviewFile.sha256, evidenceFile.sha256);
  fail(evidenceFile.json.snapshot_id === snapshotId && refreshFile.json.snapshot_id === snapshotId,
    'candidate snapshot sidecars identify another snapshot');
  fail(evidenceFile.json.authoritative_audit?.sha256 === auditFile.sha256
    && refreshFile.json.authoritative_audit?.sha256 === auditFile.sha256,
  'candidate snapshot sidecars do not bind the authoritative audit');
  fail(evidenceFile.json.review_matrix?.sha256 === reviewFile.sha256
    && refreshFile.json.review_matrix?.sha256 === reviewFile.sha256,
  'candidate snapshot sidecars do not bind the review matrix');
  fail(selection.snapshot_sha256 === evidenceFile.sha256 && selection.audit_sha256 === auditFile.sha256
    && selection.review_sha256 === reviewFile.sha256,
  'preview selection sidecar hashes are inconsistent');
  return { evidenceFile, refreshFile, auditFile, reviewFile };
}

async function loadExpectedBuilds(workbenchRoot, obbRunsManifestPath, targets) {
  const expected = new Map();
  const inventory = {};
  const poseTarget = targets.find(target => target.releaseId.includes('/pose/'));
  if (poseTarget) {
    const poseReceiptPath = resolve(workbenchRoot,
      'runs/conversion/20260930-220216_ultralytics_yolo_yolo26-pose-s-s100p-fixed-max-asym-b8/output/build_receipt.json');
    const poseReceipt = await readLocalJson(poseReceiptPath, 'Pose-s/S100P fixed-max conversion receipt');
    fail(poseReceipt.json.status === 'host-compiled' && poseReceipt.json.model?.family === 'yolo26'
      && poseReceipt.json.model?.task === 'pose' && poseReceipt.json.model?.size === 's'
      && poseReceipt.json.target?.platform === 's100p', 'Pose conversion receipt is not the authorized rebuild');
    expected.set(poseTarget.releaseId, {
      buildId: poseReceipt.json.build_id,
      conversionReceipt: poseReceipt,
    });
    inventory.pose_conversion_receipt = { path: poseReceipt.path, sha256: poseReceipt.sha256 };
  }

  const obbTargets = targets.filter(target => target.releaseId.includes('/obb/'));
  if (obbTargets.length) {
    const obbRunsPath = resolve(obbRunsManifestPath || resolve(workbenchRoot,
      'outputs/analysis/obb_x_full_rebuild_20261001/integer_runs.json'));
    const obbRunsFile = await readLocalJson(obbRunsPath, 'OBB full-rebuild runs manifest');
    const obbRuns = obbRunsFile.json;
    fail(Array.isArray(obbRuns.runs) && obbRuns.runs.length >= obbTargets.length,
      'OBB runs manifest must contain every requested target');
    const rowsByPlatform = new Map();
    for (const row of obbRuns.runs) {
      const platform = String(row.platform || '').toLowerCase();
      fail(['s600', 's100p', 's100'].includes(platform) && !rowsByPlatform.has(platform),
        `OBB runs manifest has a duplicate/unsupported platform: ${platform}`);
      rowsByPlatform.set(platform, row);
    }
    const selectedPlatforms = obbTargets.map(target => target.releaseId.split('/').at(-1));
    fail(selectedPlatforms.every(platform => rowsByPlatform.has(platform)),
      'OBB runs manifest is missing a requested S-series target');
    if (obbTargets.length === 3) {
      fail(obbRuns.runs.length === 3
        && JSON.stringify([...rowsByPlatform.keys()].sort()) === JSON.stringify(['s100', 's100p', 's600']),
      'the default OBB selection requires exactly one run row for each S-series target');
    }
    const obbManifestRef = { path: obbRunsFile.path, sha256: obbRunsFile.sha256 };
    inventory.obb_runs_manifest = obbManifestRef;
    inventory.obb_conversion_receipts = {};
    for (const target of obbTargets) {
      const platform = target.releaseId.split('/').at(-1);
      const row = rowsByPlatform.get(platform);
      const buildId = row.build_id || obbRuns.build_id;
      const campaignSha256 = row.campaign_sha256 || obbRuns.campaign_sha256;
      fail(typeof buildId === 'string' && buildId.length > 0,
        `OBB ${platform} manifest row has no target-specific build_id`);
      fail(HASH_RE.test(campaignSha256 || ''),
        `OBB ${platform} manifest row has no valid campaign_sha256`);
      fail(isAbsolute(row.conversion || ''), `OBB ${platform} conversion path must be absolute`);
      const conversionDir = resolve(row.conversion);
      const rel = relative(resolve(workbenchRoot, 'runs/conversion'), conversionDir);
      fail(rel && !rel.startsWith('..') && !isAbsolute(rel),
        `OBB ${platform} conversion path is outside the Workbench conversion run root`);
      const campaignManifest = await readLocalJson(resolve(conversionDir, 'input/campaign.json'),
        `OBB-x/${platform} campaign manifest`);
      fail(campaignManifest.sha256 === campaignSha256,
        `OBB-x/${platform} campaign manifest SHA differs from its runs-manifest row`);
      const receiptPath = resolve(conversionDir, 'output/build_receipt.json');
      const receipt = await readLocalJson(receiptPath, `OBB-x/${platform} conversion receipt`);
      fail(receipt.json.status === 'host-compiled' && receipt.json.build_id === buildId
        && receipt.json.model?.family === 'yolo26' && receipt.json.model?.task === 'obb'
        && receipt.json.model?.size === 'x' && receipt.json.target?.platform === platform,
      `OBB-x/${platform} conversion receipt does not match its manifest row`);
      let sourceManifest;
      if (row.source_manifest) {
        fail(typeof row.source_manifest.path === 'string' && isAbsolute(row.source_manifest.path)
          && HASH_RE.test(row.source_manifest.sha256 || ''),
        `OBB ${platform} source_manifest must include an absolute path and SHA-256`);
        const sourceManifestFile = await readLocalJson(row.source_manifest.path, `OBB ${platform} source manifest`);
        fail(sourceManifestFile.sha256 === row.source_manifest.sha256,
          `OBB ${platform} source manifest SHA-256 mismatch`);
        const sourceRows = Array.isArray(sourceManifestFile.json.runs)
          ? sourceManifestFile.json.runs.filter(item => String(item.platform || '').toLowerCase() === platform)
          : [sourceManifestFile.json];
        fail(sourceRows.length === 1, `OBB ${platform} source manifest must identify exactly one target row`);
        const sourceRow = sourceRows[0];
        const sourceBuildId = sourceRow.build_id || sourceManifestFile.json.build_id;
        const sourceCampaignSha = sourceRow.campaign_sha256 || sourceManifestFile.json.campaign_sha256;
        const sourceConversionPath = sourceRow.conversion || sourceManifestFile.json.conversion;
        fail(sourceBuildId === buildId && sourceCampaignSha === campaignSha256
          && sourceConversionPath === conversionDir,
        `OBB ${platform} source manifest does not identify the selected build/campaign/conversion`);
        sourceManifest = { path: sourceManifestFile.path, sha256: sourceManifestFile.sha256 };
      }
      const releaseId = target.releaseId;
      const inputReceipt = {
        path: receipt.path,
        sha256: receipt.sha256,
        build_id: buildId,
        campaign_sha256: campaignSha256,
        campaign_manifest: { path: campaignManifest.path, sha256: campaignManifest.sha256 },
        runs_manifest: obbManifestRef,
        ...(sourceManifest ? { source_manifest: sourceManifest } : {}),
      };
      expected.set(releaseId, {
        buildId,
        campaignSha256,
        campaignManifest,
        conversionReceipt: receipt,
        obbRunsManifest: obbManifestRef,
        ...(sourceManifest ? { sourceManifest } : {}),
      });
      inventory.obb_conversion_receipts[releaseId] = inputReceipt;
    }
  }
  return {
    expected,
    inventory,
  };
}

function catalogReleaseMap(catalog) {
  const map = new Map();
  for (const record of catalog.models || []) {
    for (const variant of record.variants || []) {
      for (const release of variant.platforms || []) {
        const id = `${record.id}/${variant.size}/${release.platform.toLowerCase()}`;
        fail(!map.has(id), `duplicate candidate catalog release ${id}`);
        map.set(id, { record, variant, release });
      }
    }
  }
  return map;
}

function compareNumber(left, right, label, tolerance = 1e-12) {
  fail(typeof left === 'number' && Number.isFinite(left) && typeof right === 'number'
    && Number.isFinite(right) && Math.abs(left - right) <= tolerance, `${label} does not match cited evidence`);
}

function metricInfo(task) {
  return task === 'pose'
    ? { raw: 'keypoints/AP', field: 'keypoints_ap', display: 'keypoints-all-map-50-95' }
    : { raw: 'mAP50', field: 'map_50', display: 'obb-map-50' };
}

function validateMetricSources(task, candidateRelease, manifest, comparisonRef) {
  const info = metricInfo(task);
  const comparison = comparisonRef.json;
  fail(['valid-evidence', 'comparison-computed'].includes(comparison.status)
    && comparison.direct_metric_comparison === 'valid',
    `${task} comparison evidence is not valid`);
  fail(comparison.comparison_scope?.comparable === true
    && comparison.comparison_scope.kind === 'end-to-end-metric-comparison'
    && comparison.comparison_scope.not_a_pure_quantization_loss_estimate === true,
  `${task} comparison is not an end-to-end comparable result`);
  const metric = comparison.metrics?.[info.raw];
  fail(metric && ['float', 'board', 'board_minus_float', 'retention_ratio']
    .every(key => Object.hasOwn(metric, key)), `${task} comparison receipt is missing ${info.raw}`);
  for (const key of ['float', 'board']) {
    fail(typeof metric[key] === 'number' && Number.isFinite(metric[key]) && metric[key] >= 0 && metric[key] <= 1,
      `${task} comparison ${info.raw}.${key} is not a metric ratio`);
  }
  compareNumber(metric.board_minus_float, metric.board - metric.float, `${task} end-to-end delta`);
  compareNumber(metric.retention_ratio, metric.float > 0 ? metric.board / metric.float : null,
    `${task} retention ratio`);
  compareNumber(manifest.accuracy?.comparison?.[info.raw]?.float, metric.float, `${task} release float metric`);
  compareNumber(manifest.accuracy?.comparison?.[info.raw]?.board, metric.board, `${task} release board metric`);
  compareNumber(manifest.accuracy?.comparison?.[info.raw]?.board_minus_float,
    metric.board_minus_float, `${task} release metric delta`);
  compareNumber(manifest.accuracy?.board?.[info.raw], metric.board, `${task} standalone board metric`);
  const taskAccuracy = candidateRelease.accuracy || {};
  fail(typeof taskAccuracy.dataset === 'string' && taskAccuracy.dataset.trim(), `${task} catalog dataset label is missing`);
  const finalizedDataset = manifest.accuracy?.dataset;
  fail(typeof finalizedDataset === 'string'
    && (task === 'pose'
      ? /coco/i.test(finalizedDataset) && /keypoint/i.test(finalizedDataset)
      : /dota/i.test(finalizedDataset) && /val/i.test(finalizedDataset) && /single.scale/i.test(finalizedDataset)),
  `${task} finalized dataset does not match its reviewed COCO-keypoints/DOTA-val scope`);
  fail(comparison.dataset?.expected_images === taskAccuracy.images,
    `${task} comparison evidence image count differs from the reviewed catalog dataset`);
  fail(manifest.accuracy?.expected_images === taskAccuracy.images,
    `${task} release manifest image count differs from the reviewed catalog dataset`);
  if (task === 'obb') {
    fail(taskAccuracy.dataset === 'DOTA val'
      && taskAccuracy.evaluation_scope === 'local_dota_val_single_scale',
    'OBB Web dataset/evaluation scope must remain the catalog’s local DOTA val single-scale scope');
  }
  const floatPointer = `/metrics/${info.raw.replace(/~/g, '~0').replace(/\//g, '~1')}/float`;
  const runtimePointer = `/metrics/${info.raw.replace(/~/g, '~0').replace(/\//g, '~1')}/board`;
  const floatValue = jsonPointer(comparison, floatPointer, `${task} float metric pointer`);
  const runtimeValue = jsonPointer(comparison, runtimePointer, `${task} board metric pointer`);
  compareNumber(floatValue, metric.float, `${task} float metric pointer`);
  compareNumber(runtimeValue, metric.board, `${task} board metric pointer`);
  const normalizedComparison = {
    status: 'valid-evidence',
    direct_metric_comparison: comparison.direct_metric_comparison,
    comparison_scope: comparison.comparison_scope,
    metrics: { [info.field]: metric },
  };
  const records = comparableAccuracyRecords(
    task,
    { [info.field]: floatValue },
    { [info.field]: runtimeValue },
    {
      dataset: taskAccuracy.dataset,
      ...(taskAccuracy.evaluation_scope ? { evaluation_scope: taskAccuracy.evaluation_scope } : {}),
    },
    normalizedComparison,
  );
  return {
    records,
    binding: {
      [info.field]: {
        float: { source: 'comparison_receipt', pointer: floatPointer },
        runtime: { source: 'comparison_receipt', pointer: runtimePointer },
      },
    },
    metric: plain(metric),
  };
}

function normalizePrecision(conversion) {
  const value = conversion?.precision_policy || conversion?.quantization;
  fail(value !== undefined && value !== null, 'release conversion precision/quantization metadata is missing');
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function validatePerformanceSources(manifest, evidence, artifactSha) {
  const native = evidence.native_performance.json;
  const binding = evidence.native_perf_binding.json;
  fail(native.tool && native.implementation && native.timing_scope,
    'native performance receipt is missing its timing identity');
  fail((binding.model?.sha256 || binding.model_sha256 || binding.artifact?.sha256) === artifactSha,
    'native performance receipt is not bound to this HBM');
  const measurements = native.measurements || [];
  fail(measurements.length === 2 && measurements.map(row => Number(row.threads)).sort().join(',') === '1,2',
    'native performance must include single and dual concurrency');
  const manifestPerf = manifest.runtime_performance;
  fail(manifestPerf && manifestPerf.tool === native.tool
    && manifestPerf.timing_scope === native.timing_scope
    && manifestPerf.measurements?.length === measurements.length,
  'release performance summary is not bound to the native performance receipt');
  for (const measurement of measurements) {
    const summary = manifestPerf.measurements.find(row => Number(row.threads) === Number(measurement.threads));
    fail(summary, `release performance summary is missing concurrency ${measurement.threads}`);
    for (const field of ['average_latency_ms', 'aggregate_fps']) {
      compareNumber(summary[field], measurement[field], `release native performance ${field}`);
    }
  }
  const runtime = measurements.flatMap(item => [
    {
      metric: 'latency', value: Number(item.average_latency_ms), unit: 'ms',
      concurrency: Number(item.threads), observedMin: Number(item.observed_min_latency_ms),
      observedMax: Number(item.observed_max_latency_ms),
    },
    { metric: 'throughput', value: Number(item.aggregate_fps), unit: 'fps', concurrency: Number(item.threads) },
  ]);
  for (const entry of runtime) fail(Number.isFinite(entry.value), 'native performance contains invalid numbers');
  const e2eSources = [evidence.cpp_e2e_single.json, evidence.cpp_e2e_dual.json];
  const streams = manifest.cpp_end_to_end?.streams || {};
  for (const source of e2eSources) {
    const count = String(source.pipeline_streams);
    const releaseStream = streams[count];
    fail(releaseStream && Number(source.pipeline_streams) === Number(source.runtime_submission_threads),
      `release C++ E2E summary is missing a matching ${count}-stream receipt`);
    for (const stage of ['preprocess', 'runtime', 'postprocess', 'end_to_end']) {
      for (const statistic of ['mean', 'p50', 'p95', 'min', 'max']) {
        compareNumber(releaseStream.metrics_ms?.[stage]?.[statistic], source.metrics_ms?.[stage]?.[statistic],
          `release C++ E2E ${count}-stream ${stage}.${statistic}`);
      }
    }
    compareNumber(releaseStream.throughput_fps, source.throughput_fps,
      `release C++ E2E ${count}-stream throughput`);
  }
  return {
    timing: {
      scope: native.timing_scope,
      tool: native.tool,
      implementation: native.implementation,
      threadSemantics: native.thread_semantics,
      stages: native.stages,
      runsPerCondition: Number(native.runs_per_condition),
      framesPerRun: Number(native.frames_per_run),
      warmupFramesPerCondition: Number(native.warmup_frames_per_condition),
    },
    performance: runtime,
    endToEnd: e2eSources.map(source => ({
      timing: {
        scope: source.timing_scope,
        tool: source.tool,
        implementation: source.implementation,
        pipelineStreams: Number(source.pipeline_streams),
        runtimeSubmissionThreads: Number(source.runtime_submission_threads),
        cpuThreadPolicy: source.cpu_thread_policy,
        onlineCpuThreads: Number(source.online_cpu_threads),
        opencvThreads: Number(source.opencv_threads),
        cpuGovernor: source.cpu_governor,
        cpuFrequencyMhz: Number(source.cpu_frequency_mhz),
        bpuFrequencyMhz: Number(source.bpu_frequency_mhz),
        warmupFramesPerRound: Number(source.warmup_frames_per_round),
        rounds: Number(source.rounds),
        framesPerRound: Number(source.frames_per_stream_per_round),
        timedFrames: Number(source.timed_frames),
        aggregateWallMs: Number(source.aggregate_wall_ms),
      },
      metrics: Object.fromEntries(['preprocess', 'runtime', 'postprocess', 'end_to_end'].map(stage => [
        stage === 'end_to_end' ? 'endToEnd' : stage,
        Object.fromEntries(['mean', 'p50', 'p95', 'min', 'max'].map(stat => [
          stat === 'mean' ? 'value' : stat, Number(source.metrics_ms?.[stage]?.[stat]),
        ]).filter(([, value]) => Number.isFinite(value))),
      ])),
      throughputFps: Number(source.throughput_fps),
    })),
  };
}

function resolveBundleFiles(workbenchRoot, id, buildId, finalization) {
  const [catalog, family, task, size, platform] = id.split('/');
  fail(catalog === 'ultralytics_yolo' && family === 'yolo26', `${id} is outside the YOLO26 release tree`);
  const expectedDirectory = resolve(workbenchRoot, 'releases/finalized/models/ultralytics_yolo/yolo26',
    task, size, platform, 'rebuilds', buildId);
  const actualDirectory = resolve(finalization.finalized_bundle?.path || '');
  fail(actualDirectory === expectedDirectory, `${id} finalization receipt points to an unexpected bundle directory`);
  const manifestRef = finalization.finalized_bundle?.files || {};
  const names = Object.keys(manifestRef);
  const artifactName = names.find(name => name !== 'oe_report.html' && name !== 'oe_report_data.json'
    && name !== 'release.json' && name !== 'SHA256SUMS');
  fail(artifactName && artifactName.endsWith('.hbm'), `${id} finalization receipt has no HBM entry`);
  return {
    directory: actualDirectory,
    artifact_file: { path: join(actualDirectory, artifactName), sha256: manifestRef[artifactName]?.sha256 },
    oe_report_html: { path: join(actualDirectory, 'oe_report.html'), sha256: manifestRef['oe_report.html']?.sha256 },
    oe_report_data: { path: join(actualDirectory, 'oe_report_data.json'), sha256: manifestRef['oe_report_data.json']?.sha256 },
    release_manifest: { path: join(actualDirectory, 'release.json'), sha256: manifestRef['release.json']?.sha256 },
    checksums: { path: join(actualDirectory, 'SHA256SUMS'), sha256: manifestRef.SHA256SUMS?.sha256 },
  };
}

async function resolveUploadReceipt(workbenchRoot, finalization, role, key, fileRef, label) {
  const uploadRoot = resolve(workbenchRoot, 'outputs/receipts/uploads');
  let ref;
  if (['artifact', 'oe_report_html', 'oe_report_data'].includes(role)) {
    ref = finalization.verified_upload_receipts?.[role];
  } else {
    const expectedName = role === 'release_manifest' ? 'release.json' : 'SHA256SUMS';
    const pending = (finalization.pending_metadata_objects || []).find(item => item.name === expectedName);
    fail(pending?.key === key && pending?.acl === 'public-read', `${label} metadata key is not pending public-read`);
    ref = { path: join(uploadRoot, `${sha256(Buffer.from(key, 'utf8'))}.json`) };
  }
  fail(ref && isAbsolute(ref.path), `${label} upload receipt path is missing`);
  const receiptFile = await readLocalJson(ref.path, `${label} upload receipt`);
  if (ref.sha256) fail(receiptFile.sha256 === ref.sha256, `${label} receipt hash differs from finalization receipt`);
  const receipt = receiptFile.json;
  fail(receipt.status === 'verified' && receipt.acl === 'public-read'
    && receipt.key === key && receipt.sha256 === fileRef.sha256
    && receipt.size_bytes === fileRef.bytes.length && receipt.url === `${OSS_BASE}/${key}`,
  `${label} receipt does not verify the exact public object`);
  return { path: ref.path, sha256: receiptFile.sha256, json: receipt };
}

async function loadReleaseRecord({ id, catalogEntry, expectedBuild, workbenchRoot, outputDir, banners }) {
  const parts = id.split('/');
  const size = parts.at(-2);
  const platform = parts.at(-1);
  const catalogId = parts.slice(0, -2).join('/');
  const { family, task } = catalogEntry.record;
  const buildId = expectedBuild.buildId;
  const finalizationPath = resolve(workbenchRoot, 'outputs/receipts/finalizations',
    `${task}-${size}-${platform}-${buildId}.json`);
  const finalizationFile = await readLocalJson(finalizationPath, `${id} finalization receipt`);
  const finalization = finalizationFile.json;
  fail(finalization.kind === 'yolo26-task-release-finalization'
    && finalization.model?.task === task && finalization.model?.size === size
    && finalization.target?.platform === platform,
  `${id} finalization receipt identity mismatch`);
  fail(finalization.evidence_sources && typeof finalization.evidence_sources === 'object'
    && !Array.isArray(finalization.evidence_sources), `${id} finalization has no source evidence index`);
  const bundleFiles = resolveBundleFiles(workbenchRoot, id, buildId, finalization);
  const files = {};
  for (const [role, ref] of Object.entries(bundleFiles)) {
    if (role === 'directory') continue;
    files[role] = await verifiedFile(ref.path, ref.sha256, `${id}.${role}`);
    const finalizedMeta = finalization.finalized_bundle.files[basename(ref.path)];
    fail(finalizedMeta && finalizedMeta.sha256 === files[role].sha256
      && finalizedMeta.size_bytes === files[role].bytes.length,
    `${id} finalization bundle size/hash index differs for ${role}`);
  }
  const manifest = JSON.parse(files.release_manifest.bytes.toString('utf8'));
  fail(manifest.status === 'released' && manifest.publication_status === 'published'
    && manifest.publication?.status === 'published', `${id} release manifest is not finalized/published`);
  fail(manifest.model?.source === catalogEntry.record.source && manifest.model?.family === family
    && manifest.model?.task === task && manifest.model?.size === size
    && manifest.target?.platform === platform,
  `${id} release manifest identity mismatch`);
  const conversionRef = finalization.evidence_sources.conversion_receipt;
  fail(conversionRef && HASH_RE.test(conversionRef.sha256 || ''), `${id} has no conversion receipt reference`);
  fail(resolve(conversionRef.path) === expectedBuild.conversionReceipt.path
    && conversionRef.sha256 === expectedBuild.conversionReceipt.sha256,
  `${id} finalization points to another conversion build`);
  const buildEvidence = await verifiedJson(conversionRef.path, conversionRef.sha256, `${id} conversion receipt`);
  fail(manifest.provenance?.build_id === buildId && buildEvidence.json.build_id === buildId,
    `${id} final manifest does not identify the inventory build`);
  const expectedBase = `models/ultralytics_yolo/yolo26/${task}/${size}/${platform}`;
  const expectedPrefix = `${expectedBase}/rebuilds/${buildId}`;
  fail(manifest.publication?.object_prefix === expectedPrefix,
    `${id} manifest is not published under its build-versioned prefix`);
  fail(manifest.artifact?.name === basename(files.artifact_file.path), `${id} HBM filename differs from release manifest`);
  fail(manifest.provenance?.conversion_receipt_sha256 === finalization.evidence_sources.conversion_receipt?.sha256,
    `${id} manifest does not bind the finalization conversion receipt`);
  fail(manifest.provenance?.board_receipt_sha256 === finalization.evidence_sources.board_receipt?.sha256,
    `${id} manifest does not bind the finalization board receipt`);
  fail(manifest.provenance?.comparison_receipt_sha256 === finalization.evidence_sources.comparison_receipt?.sha256,
    `${id} manifest does not bind the finalization comparison receipt`);
  fail(manifest.artifact?.sha256 === files.artifact_file.sha256
    && Number(manifest.artifact?.size_bytes) === files.artifact_file.bytes.length,
  `${id} HBM bytes differ from the release manifest`);
  fail(finalization.finalized_bundle?.release_manifest_sha256 === files.release_manifest.sha256
    && finalization.finalized_bundle?.sha256sums_sha256 === files.checksums.sha256,
  `${id} finalization receipt does not bind release.json and SHA256SUMS`);

  const sums = files.checksums.bytes.toString('utf8').trim().split(/\r?\n/).map(line => {
    const match = /^([a-f0-9]{64})  ([A-Za-z0-9_.-]+)$/.exec(line);
    fail(match, `${id} has malformed SHA256SUMS`);
    return [match[2], match[1]];
  });
  const sumMap = new Map(sums);
  fail(sumMap.size === sums.length && sumMap.size === 4, `${id} SHA256SUMS must cover exactly four bundle files`);
  for (const role of ['artifact_file', 'oe_report_html', 'oe_report_data', 'release_manifest']) {
    fail(sumMap.get(basename(files[role].path)) === files[role].sha256,
      `${id} SHA256SUMS does not bind ${role}`);
  }

  const publishedObjects = manifest.publication?.uploaded_objects || {};
  for (const [role, filename] of [
    ['artifact', basename(files.artifact_file.path)],
    ['oe_report_html', 'oe_report.html'],
    ['oe_report_data', 'oe_report_data.json'],
  ]) {
    const object = publishedObjects[role];
    fail(object?.key === `${expectedPrefix}/${filename}` && object.url === `${OSS_BASE}/${object.key}`,
      `${id} published ${role} object does not use the canonical versioned key`);
  }
  const metadataObject = name => (manifest.publication?.metadata_objects || []).find(item => item.name === name);
  const releaseObject = metadataObject('release.json');
  const checksumsObject = metadataObject('SHA256SUMS');
  fail(releaseObject?.key === `${expectedPrefix}/release.json` && releaseObject.acl === 'public-read'
    && checksumsObject?.key === `${expectedPrefix}/SHA256SUMS` && checksumsObject.acl === 'public-read',
  `${id} release metadata object keys are not canonical versioned keys`);
  const urls = {
    artifact_file: publishedObjects.artifact.url,
    oe_report_html: publishedObjects.oe_report_html.url,
    oe_report_data: publishedObjects.oe_report_data.url,
    release_manifest: `${OSS_BASE}/${releaseObject.key}`,
    checksums: `${OSS_BASE}/${checksumsObject.key}`,
  };
  fail(manifest.artifact?.url === urls.artifact_file
    && manifest.artifact?.release_manifest_url === urls.release_manifest
    && manifest.artifact?.checksums_url === urls.checksums
    && manifest.reports?.oe_report_html_url === urls.oe_report_html
    && manifest.reports?.oe_report_data_url === urls.oe_report_data,
  `${id} manifest URLs differ from the versioned publication objects`);
  const uploadKeys = {
    artifact: publishedObjects.artifact.key,
    oe_report_html: publishedObjects.oe_report_html.key,
    oe_report_data: publishedObjects.oe_report_data.key,
    release_manifest: releaseObject.key,
    checksums: checksumsObject.key,
  };
  const uploads = {};
  for (const [role, key] of Object.entries(uploadKeys)) {
    const fileRole = role === 'artifact' ? 'artifact_file' : role;
    uploads[role] = await resolveUploadReceipt(
      workbenchRoot, finalization, role, key, files[fileRole], `${id}.${role}`,
    );
  }

  const sourceEvidenceIndex = {};
  const sourceEvidence = {};
  for (const [role, ref] of Object.entries(finalization.evidence_sources)) {
    fail(role && ref && isAbsolute(ref.path) && HASH_RE.test(ref.sha256 || ''),
      `${id} source evidence ref ${role} is malformed`);
    const bytes = await verifiedFile(ref.path, ref.sha256, `${id}.source.${role}`);
    sourceEvidenceIndex[role] = { path: bytes.path, sha256: bytes.sha256 };
    if (ref.path.endsWith('.json')) {
      let json;
      try { json = JSON.parse(bytes.bytes.toString('utf8')); } catch (error) {
        throw new Error(`YOLO26 rebuild selection: ${id} source ${role} is invalid JSON: ${error.message}`);
      }
      sourceEvidence[role] = { ...bytes, json };
    } else {
      sourceEvidence[role] = bytes;
    }
  }
  const requiredEvidence = {
    build_receipt: 'conversion_receipt',
    board_receipt: 'board_receipt',
    float_reference: 'float_result_reference',
    comparison_receipt: 'comparison_receipt',
    native_performance: 'runtime_performance',
    native_perf_binding: 'runtime_performance_binding',
    cpp_e2e_single: 'cpp_e2e_single',
    cpp_e2e_dual: 'cpp_e2e_dual',
    cpp_e2e_provenance: 'cpp_e2e_provenance',
  };
  for (const role of Object.values(requiredEvidence)) {
    fail(sourceEvidence[role]?.json, `${id} source evidence is missing ${role}`);
  }
  const evidence = Object.fromEntries(Object.entries(requiredEvidence).map(([target, source]) => [
    target, { path: sourceEvidenceIndex[source].path, sha256: sourceEvidenceIndex[source].sha256 },
  ]));
  evidence.finalization_receipt = { path: finalizationPath, sha256: finalizationFile.sha256 };
  const build = sourceEvidence.conversion_receipt.json;
  const board = sourceEvidence.board_receipt.json;
  const comparison = sourceEvidence.comparison_receipt;
  const nativePerf = sourceEvidence.runtime_performance.json;
  const performanceBinding = sourceEvidence.runtime_performance_binding.json;
  const cppSingle = sourceEvidence.cpp_e2e_single.json;
  const cppDual = sourceEvidence.cpp_e2e_dual.json;
  const cppProvenance = sourceEvidence.cpp_e2e_provenance.json;
  const artifactSha = files.artifact_file.sha256;
  fail(buildId === build.build_id, `${id} build ID differs between release manifest and conversion receipt`);
  fail(build.status === 'host-compiled' && build.build_id === buildId
    && build.model?.family === family && build.model?.task === task && build.model?.size === size
    && build.target?.platform === platform,
  `${id} conversion receipt identity/status mismatch`);
  fail(build.artifacts?.model?.sha256 === artifactSha
    && Number(build.artifacts?.model?.size_bytes) === files.artifact_file.bytes.length,
  `${id} conversion receipt does not bind this HBM`);
  fail(board.model?.family === family && board.model?.task === task && board.model?.size === size
    && board.model?.platform === platform && board.model?.sha256 === artifactSha
    && board.model?.board_sha256_verified === true,
  `${id} board receipt does not bind this HBM`);
  fail(board.conversion?.receipt_sha256 === evidence.build_receipt.sha256,
    `${id} board receipt points to another conversion receipt`);
  fail(comparison.json.campaign?.build_id === buildId
    && comparison.json.metrics && comparison.json.direct_metric_comparison === 'valid',
  `${id} comparison receipt does not bind this release build`);
  const floatRef = sourceEvidence.float_result_reference.json;
  fail(floatRef.kind === 's-family-float-result-reference'
    && floatRef.rebuild_campaign?.build_id === buildId
    && floatRef.lineage?.campaigns_are_distinct === true,
  `${id} float reference does not establish distinct rebuild lineage`);
  if (task === 'obb') {
    const rebuildSelection = sourceEvidence.rebuild_selection.json;
    const selectionCampaignSha = rebuildSelection.lineage?.rebuild_campaign_sha256
      || rebuildSelection.rebuild?.campaign?.sha256 || rebuildSelection.campaign?.sha256;
    fail(HASH_RE.test(expectedBuild.campaignSha256 || '')
      && selectionCampaignSha === expectedBuild.campaignSha256,
    `${id} rebuild selection campaign SHA differs from the selected OBB manifest row`);
    const selectedConversion = rebuildSelection.rebuild?.conversion_receipt
      || rebuildSelection.conversion_receipt;
    fail(selectedConversion?.path === expectedBuild.conversionReceipt.path
      && selectedConversion?.sha256 === expectedBuild.conversionReceipt.sha256,
    `${id} rebuild selection does not bind the exact OBB conversion receipt`);
    const selectedCampaign = rebuildSelection.rebuild?.campaign || rebuildSelection.campaign;
    fail(selectedCampaign?.path === expectedBuild.campaignManifest.path
      && selectedCampaign?.sha256 === expectedBuild.campaignSha256,
    `${id} rebuild selection does not bind the exact OBB campaign manifest`);
    if (manifest.conversion?.precision_policy === 'fp16') {
      const graph = build.ptq_graph_adaptation;
      fail(graph && graph.kind && graph.source_ptq && graph.adapted_ptq && graph.adaptor && graph.proof,
        `${id} FP16 Swish build receipt has no completed PTQ graph-adaptation evidence`);
      const graphRoles = {
        ptq_source_original: graph.source_ptq,
        ptq_source_adapted: graph.adapted_ptq,
        ptq_graph_adaptation_proof: graph.proof,
        ptq_graph_adaptation_code: graph.adaptor,
      };
      for (const [role, expectedRef] of Object.entries(graphRoles)) {
        const finalizedRef = finalization.evidence_sources[role];
        const verifiedRef = sourceEvidenceIndex[role];
        fail(expectedRef && isAbsolute(expectedRef.path || '') && HASH_RE.test(expectedRef.sha256 || '')
          && finalizedRef?.path === expectedRef.path && finalizedRef?.sha256 === expectedRef.sha256
          && verifiedRef?.path === expectedRef.path && verifiedRef?.sha256 === expectedRef.sha256,
        `${id} finalization evidence ${role} does not exactly bind the FP16 build receipt path/SHA`);
      }
    } else {
      fail(manifest.conversion?.precision_policy === 'int8-int16' && !build.ptq_graph_adaptation,
        `${id} native OBB integer build must not claim a PTQ graph adaptation`);
      const ptqModel = build.provenance?.ptq_model;
      const finalizedRef = finalization.evidence_sources.ptq_model;
      const indexedRef = sourceEvidenceIndex.ptq_model;
      fail(ptqModel && isAbsolute(ptqModel.path || '') && HASH_RE.test(ptqModel.sha256 || '')
        && Number.isInteger(ptqModel.size_bytes) && ptqModel.size_bytes > 0
        && finalizedRef?.path === ptqModel.path && finalizedRef?.sha256 === ptqModel.sha256
        && indexedRef?.path === ptqModel.path && indexedRef?.sha256 === ptqModel.sha256,
      `${id} native integer PTQ model provenance does not match the finalization source index`);
    }
    await validateObbPrecisionEvidence({
      releaseId: id,
      manifestConversion: manifest.conversion,
      build,
      buildReceiptRef: expectedBuild.conversionReceipt,
      campaignRef: expectedBuild.campaignManifest,
      finalizationEvidence: finalization.evidence_sources,
      sourceEvidenceIndex,
    });
  }
  const boundPerformanceModel = performanceBinding.model?.sha256
    || performanceBinding.model_sha256 || performanceBinding.artifact?.sha256;
  fail(boundPerformanceModel === artifactSha, `${id} performance input is not HBM-bound`);
  const boundCppModel = cppProvenance.model?.sha256 || cppProvenance.model_sha256
    || cppProvenance.rebuild_collection?.model_sha256;
  fail(boundCppModel === artifactSha,
    `${id} C++ E2E provenance does not bind this HBM`);
  fail(cppProvenance.execution?.status === 'completed'
    && cppProvenance.model?.runtime_path && basename(cppProvenance.model.runtime_path) === basename(files.artifact_file.path),
  `${id} C++ E2E provenance does not identify a completed run on this HBM`);
  for (const source of [cppSingle, cppDual]) {
    fail(typeof source.model === 'string' && source.model === cppProvenance.model.runtime_path,
      `${id} C++ E2E stream model path differs from its HBM-bound provenance`);
  }
  const conversion = manifest.conversion;
  fail(conversion && HASH_RE.test(conversion.config_sha256 || ''),
    `${id} compiler configuration SHA is missing from release manifest`);
  fail(build.conversion?.quantization === conversion.quantization,
    `${id} release quantization does not match the compiler build receipt`);
  fail(typeof conversion.precision_policy === 'string' || conversion.precision_policy === null,
    `${id} precision_policy must be an explicit string or null`);
  const precision = normalizePrecision(conversion);
  const candidateRelease = catalogEntry.release;
  const accuracy = validateMetricSources(task, candidateRelease, manifest, comparison);
  const benchmarkPerf = validatePerformanceSources(manifest, {
    native_performance: sourceEvidence.runtime_performance,
    native_perf_binding: sourceEvidence.runtime_performance_binding,
    cpp_e2e_single: sourceEvidence.cpp_e2e_single,
    cpp_e2e_dual: sourceEvidence.cpp_e2e_dual,
    cpp_e2e_provenance: sourceEvidence.cpp_e2e_provenance,
  }, artifactSha);

  const oeData = JSON.parse(files.oe_report_data.bytes.toString('utf8'));
  fail(oeData.schema_version === 1 && oeData.provenance?.artifact_sha256 === artifactSha,
    `${id} structured OE report does not bind this HBM`);
  const sourceDataName = `${safeId('ultralytics_yolo', 'yolo26', task, size, platform)}-oe-data.json`;
  const reportDataUrl = `reports/data/${sourceDataName}`;
  const reportDataPath = resolve(outputDir, reportDataUrl);
  await mkdir(dirname(reportDataPath), { recursive: true });
  await writeFile(reportDataPath, files.oe_report_data.bytes);

  const banner = banners.items.get(`ultralytics_yolo/yolo26/${task}`);
  fail(banner, `${task} real-inference banner is missing`);
  const coverExtension = extname(banner.file).toLowerCase();
  const coverFilename = `${safeId('ultralytics_yolo', 'yolo26', task)}${coverExtension}`;
  const coverRelative = `assets/models/${coverFilename}`;
  const coverPath = resolve(outputDir, coverRelative);
  await mkdir(dirname(coverPath), { recursive: true });
  await copyFile(banner.path, coverPath);

  const variant = catalogEntry.variant;
  const record = catalogEntry.record;
  const uiId = safeId('ultralytics_yolo', 'yolo26', task, size, platform);
  const sourceCommit = 'eed26ce610d7fba03a68d1c0ee6e62603cd9b85d';
  const names = { model: `YOLO26 ${task === 'pose' ? 'Pose' : 'OBB'}`, variant: `YOLO26${size} ${task === 'pose' ? 'Pose' : 'OBB'}` };
  const model = {
    id: uiId,
    catalogId,
    name: names.model,
    variantName: names.variant,
    family,
    modelSize: size,
    releaseStatus: 'released',
    releasePlatform: platform.toUpperCase(),
    task: taskUi(task).label,
    taskId: taskUi(task).id,
    tasks: [taskUi(task).id],
    description: JSON.parse(await readFile(resolve(webRoot, 'release/presentation-overrides.json'), 'utf8')).models[catalogId].zh,
    descriptionEn: JSON.parse(await readFile(resolve(webRoot, 'release/presentation-overrides.json'), 'utf8')).models[catalogId].en,
    platforms: [platform.toUpperCase()],
    coverImage: coverRelative,
    coverLabel: banner.cover_label,
    sample: `${record.source}/${record.family}/${record.task}/${size}`,
    source: `https://github.com/D-Robotics/rdk_model_zoo/tree/${sourceCommit}/${record.sample_path}`,
    sourceRepositoryCommit: sourceCommit,
    ...(manifest.provenance?.conversion_source_commit
      ? { conversionRepositoryCommit: manifest.provenance.conversion_source_commit } : {}),
    checkpointProvenance: {
      sourceWeightUrl: catalogEntry.release.provenance?.source_weight_url || null,
      sourceCheckpointSha256: catalogEntry.release.provenance?.source_checkpoint_sha256 || null,
    },
    licenseName: record.license?.name,
    licenseUrl: record.license?.url,
    shape: (variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width]).join(' × '),
    reportDataUrl,
    reportSourceUrl: uploads.oe_report_html.json.url,
    assets: [{
      role: `${platform.toUpperCase()} 部署模型`,
      format: 'hbm',
      filename: basename(files.artifact_file.path),
      url: uploads.artifact.json.url,
      sha256: artifactSha,
      sizeBytes: files.artifact_file.bytes.length,
    }],
    benchmark: {
      precision,
      precisionPolicy: conversion.precision_policy,
      quantization: conversion.quantization,
      environment: { hardware: `RDK ${platform.toUpperCase()}` },
      ...benchmarkPerf,
      accuracy: accuracy.records,
    },
  };
  const properties = {
    parameterCount: Number(variant.model?.parameter_count),
    gflops: Number(variant.model?.gflops),
    inputShape: variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width],
    sourceInput: variant.input.source,
    runtimeInput: { format: 'nv12', resolution: [variant.input.height, variant.input.width] },
  };
  const report = {
    id: uiId,
    name: `${names.variant} (${platform.toUpperCase()})`,
    hardware: platform.toUpperCase(),
    march: manifest.target?.march,
    kind: 'conversion',
    bytes: files.oe_report_data.bytes.length,
    path: model.reportSourceUrl,
    dataUrl: reportDataUrl,
    modelIds: [uiId],
    sources: [build.run_id, build.build_id].filter(Boolean),
  };
  const row = {
    release_id: id,
    build_id: buildId,
    ...(expectedBuild.campaignSha256 ? {
      campaign_sha256: expectedBuild.campaignSha256,
      campaign_manifest: { path: expectedBuild.campaignManifest.path, sha256: expectedBuild.campaignManifest.sha256 },
      runs_manifest: expectedBuild.obbRunsManifest,
      ...(expectedBuild.sourceManifest ? { source_manifest: expectedBuild.sourceManifest } : {}),
    } : {}),
    directory: outputDir,
    files: Object.fromEntries(Object.entries(files).map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
    evidence,
    source_evidence_index: sourceEvidenceIndex,
    upload_receipts: Object.fromEntries(Object.entries(uploads).map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }])),
    metric_bindings: accuracy.binding,
  };
  return { id, model, properties, report, row, sourceEvidenceIndex, manifest, buildId };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const workbenchRoot = resolve(args.workbenchRoot);
  const previewDir = resolve(args.previewDir);
  const selection = JSON.parse(await readFile(resolve(previewDir, 'technical-selection.json'), 'utf8'));
  const snapshot = await verifySnapshotInputs(selection, webRoot);
  const catalogPath = resolve(args.catalogPath || defaultCatalog(selection.snapshot_id));
  const catalog = JSON.parse(await readFile(catalogPath, 'utf8'));
  fail(catalog.status === 'candidate' && catalog.snapshot_id === selection.snapshot_id
    && catalog.source === `model_zoo_web/candidate-staging/${selection.snapshot_id}`,
  'candidate catalog does not belong to the frozen review snapshot');
  const releases = catalogReleaseMap(catalog);
  const banners = await loadInferenceTaskBanners({ webRoot, catalog });
  if (args.validateObbManifestOnly) {
    const expectedBuilds = await loadExpectedBuilds(workbenchRoot, args.obbRunsManifestPath, args.targets);
    fail(expectedBuilds.expected.size === args.targets.length,
      'validated target inventory is missing one or more selected builds');
    process.stdout.write(`Verified OBB runs manifest ${expectedBuilds.inventory.obb_runs_manifest.path}\n`);
    for (const [id, row] of expectedBuilds.expected) {
      if (!id.includes('/obb/')) continue;
      process.stdout.write(`${id}: build=${row.buildId} campaign_sha256=${row.campaignSha256} build_receipt_sha256=${row.conversionReceipt.sha256}\n`);
    }
    return;
  }
  if (args.validatePoseOnly) {
    const id = 'ultralytics_yolo/yolo26/pose/s/s100p';
    const catalogEntry = releases.get(id);
    fail(catalogEntry?.release?.status === 'candidate', `${id} is missing from the frozen candidate catalog`);
    const receiptPath = resolve(workbenchRoot,
      'runs/conversion/20260930-220216_ultralytics_yolo_yolo26-pose-s-s100p-fixed-max-asym-b8/output/build_receipt.json');
    const conversionReceipt = await readLocalJson(receiptPath, 'Pose-s/S100P fixed-max conversion receipt');
    const build = conversionReceipt.json;
    fail(build.status === 'host-compiled' && build.model?.family === 'yolo26'
      && build.model?.task === 'pose' && build.model?.size === 's' && build.target?.platform === 's100p',
    'Pose conversion receipt is not the authorized rebuild');
    const tempDir = await mkdtemp('/tmp/yolo26-pose-s100p-verify-');
    try {
      const loaded = await loadReleaseRecord({ id, catalogEntry,
        expectedBuild: { buildId: build.build_id, conversionReceipt },
        workbenchRoot, outputDir: tempDir, banners });
      await validateSingleRebuildRecord({ record: loaded.row, model: loaded.model, directory: tempDir });
      process.stdout.write(`Validated finalized Pose-s/S100P rebuild ${loaded.buildId}; ${Object.keys(loaded.row.source_evidence_index).length} source-evidence roles and ${Object.keys(loaded.row.upload_receipts).length} public OSS receipts verified.\n`);
      process.stdout.write(`HBM SHA-256: ${loaded.model.assets[0].sha256}\n`);
      process.stdout.write(`Download: ${loaded.model.assets[0].url}\n`);
      process.stdout.write(`Frozen preview remains unchanged: ${resolve(previewDir)} (${selection.selected.length} technical rows).\n`);
      return;
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  }
  const outputDir = resolve(args.outputDir);
  const existing = await lstat(outputDir).then(() => true).catch(error => {
    if (error.code === 'ENOENT') return false;
    throw error;
  });
  fail(!existing, `refusing to overwrite existing output directory ${outputDir}`);
  const expectedBuilds = await loadExpectedBuilds(workbenchRoot, args.obbRunsManifestPath, args.targets);
  fail(expectedBuilds.expected.size === args.targets.length,
    'authorized rebuild inventory is missing one or more selected builds');
  const tempParent = dirname(outputDir);
  await mkdir(tempParent, { recursive: true });
  const tempDir = resolve(tempParent, `.${basename(outputDir)}.tmp-${process.pid}-${Date.now()}`);
  await mkdir(tempDir, { recursive: false });
  try {
    const records = [];
    const selectedReleaseIds = args.targets.map(target => target.releaseId);
    for (const id of selectedReleaseIds) {
      const catalogEntry = releases.get(id);
      fail(catalogEntry?.release?.status === 'candidate', `${id} is missing from the frozen candidate catalog`);
      const expectedBuild = expectedBuilds.expected.get(id);
      fail(expectedBuild, `${id} is missing from the authorized rebuild inventory`);
      records.push(await loadReleaseRecord({ id, catalogEntry, expectedBuild,
        workbenchRoot, outputDir: tempDir, banners }));
    }
    const properties = Object.fromEntries(records.map(record => [record.model.id, record.properties]));
    const reports = records.map(record => record.report);
    const models = records.map(record => record.model);
    const platforms = [...new Set(models.map(model => model.releasePlatform))];
    const data = {
      schemaVersion: 1,
      catalog: {
        status: 'published',
        summary: {
          sample_count: new Set(models.map(model => model.sample)).size,
          asset_count: models.reduce((sum, model) => sum + model.assets.length, 0),
          downloadable_asset_count: models.reduce((sum, model) => sum + model.assets.length, 0),
          benchmark_count: models.filter(model => model.benchmark).length,
        },
      },
      repository: { url: 'https://github.com/D-Robotics/rdk_model_zoo' },
      accuracyMetricLabels: accuracyMetricLabels(),
      release: { status: 'published', compatibility: { hardware: platforms.map(item => `RDK ${item}`).join(' / ') } },
      candidatePreview: false,
      technicalPassedPreview: false,
      models,
    };
    const sourceIndex = {
      schema_version: 1,
      kind: 'yolo26-rebuild-source-evidence-index',
      records: records.map(record => ({ release_id: record.id, build_id: record.buildId,
        evidence_sources: record.sourceEvidenceIndex })),
    };
    const sourceIndexBytes = Buffer.from(`${JSON.stringify(sourceIndex, null, 2)}\n`, 'utf8');
    const manifest = {
      schema_version: 1,
      kind: 'yolo26-rebuild-selection',
      status: 'ready-for-merge',
      generated_at: new Date().toISOString(),
      base_snapshot_id: selection.snapshot_id,
      base_authoritative_audit_sha256: selection.source_hashes.authoritative_audit_sha256,
      base_review_matrix_sha256: snapshot.reviewFile.sha256,
      base_evidence_index_sha256: snapshot.evidenceFile.sha256,
      source_evidence_index_sha256: sha256(sourceIndexBytes),
      rebuild_input_inventory: expectedBuilds.inventory,
      excluded_release_ids: selectedReleaseIds,
      records: records.map(({ row }) => row),
    };
    await writeFile(resolve(tempDir, 'data.js'), `window.MODEL_DATA = ${JSON.stringify(data, null, 2)};\n`, 'utf8');
    await writeFile(resolve(tempDir, 'model-properties.js'),
      `window.MODEL_PROPERTIES = Object.freeze(${JSON.stringify(properties, null, 2)});\n`, 'utf8');
    await mkdir(resolve(tempDir, 'reports'), { recursive: true });
    await writeFile(resolve(tempDir, 'reports/reports-data.js'),
      `window.OE_REPORTS = ${JSON.stringify(reports, null, 2)};\n`, 'utf8');
    await writeFile(resolve(tempDir, 'reports/inventory.json'),
      `${JSON.stringify({ reports: reports.map(report => ({ id: report.id, kind: report.kind })) }, null, 2)}\n`, 'utf8');
    await writeFile(resolve(tempDir, 'rebuild-selection.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
    await writeFile(resolve(tempDir, 'source-evidence-index.json'), sourceIndexBytes);
    const checked = await loadVerifiedRebuildSelection({
      directory: tempDir,
      snapshotId: selection.snapshot_id,
      auditSha: selection.source_hashes.authoritative_audit_sha256,
      expectedReleaseIds: selectedReleaseIds,
    });
    fail(checked.records.length === selectedReleaseIds.length,
      'generated selection failed its own selected-target validation');
    const stillAbsent = await lstat(outputDir).then(() => false).catch(error => {
      if (error.code === 'ENOENT') return true;
      throw error;
    });
    fail(stillAbsent, `output directory appeared while generation was running: ${outputDir}`);
    await rename(tempDir, outputDir);
    process.stdout.write(`Generated and validated ${selectedReleaseIds.length} rebuilt YOLO26 release(s) in ${outputDir}\n`);
    process.stdout.write(`Snapshot: ${selection.snapshot_id}; authoritative audit SHA-256: ${selection.source_hashes.authoritative_audit_sha256}\n`);
  } catch (error) {
    await rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

await main();
