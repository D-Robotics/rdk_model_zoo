import assert from 'node:assert/strict';
import { lstat, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path';
import { runInNewContext } from 'node:vm';

const OSS_BASE = 'https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com';
const EXPECTED_RELEASES = [
  'ultralytics_yolo/yolo26/pose/s/s100p',
  'ultralytics_yolo/yolo26/obb/x/s600',
  'ultralytics_yolo/yolo26/obb/x/s100p',
  'ultralytics_yolo/yolo26/obb/x/s100',
].sort();
const REQUIRED_FILES = [
  'artifact_file', 'release_manifest', 'checksums', 'oe_report_html', 'oe_report_data',
];
const REQUIRED_EVIDENCE = [
  'build_receipt', 'board_receipt', 'float_reference', 'comparison_receipt',
  'native_performance', 'native_perf_binding', 'cpp_e2e_single', 'cpp_e2e_dual',
  'cpp_e2e_provenance', 'finalization_receipt',
];
const REQUIRED_UPLOADS = [
  'artifact', 'oe_report_html', 'oe_report_data', 'release_manifest', 'checksums',
];
const HASH_RE = /^[a-f0-9]{64}$/;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = (condition, message) => assert.ok(condition, `Rebuild selection: ${message}`);
const plain = value => JSON.parse(JSON.stringify(value));

function releaseId(model) {
  return `${model.catalogId}/${model.modelSize}/${model.releasePlatform.toLowerCase()}`;
}

function expectedUiId(task, size, platform) {
  const safe = value => value.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
  return safe(`ultralytics-yolo-yolo26-${task}-${size}-${platform}`);
}

async function readRegularFile(path, label) {
  fail(typeof path === 'string' && isAbsolute(path), `${label} path must be absolute`);
  const info = await lstat(path);
  fail(info.isFile() && !info.isSymbolicLink(), `${label} must be a regular non-symlink file: ${path}`);
  return readFile(path);
}

async function verifyRef(ref, label) {
  fail(ref && typeof ref === 'object' && !Array.isArray(ref), `${label} must be an object`);
  fail(HASH_RE.test(ref.sha256 || ''), `${label}.sha256 must be lowercase SHA-256`);
  const bytes = await readRegularFile(ref.path, label);
  fail(digest(bytes) === ref.sha256, `${label} file SHA-256 mismatch`);
  return { path: ref.path, sha256: ref.sha256, bytes };
}

async function readJsonRef(ref, label) {
  const verified = await verifyRef(ref, label);
  let json;
  try {
    json = JSON.parse(verified.bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Rebuild selection: ${label} is not valid JSON: ${error.message}`);
  }
  fail(json && typeof json === 'object' && !Array.isArray(json), `${label} must contain a JSON object`);
  return { ...verified, json };
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/**
 * Check OBB precision metadata against the exact finalized compiler campaign,
 * build receipt, emitted config, and (for mixed precision) static PTQ proof.
 */
export async function validateObbPrecisionEvidence({
  releaseId: id,
  manifestConversion,
  build,
  buildReceiptRef,
  campaignRef,
  finalizationEvidence,
  sourceEvidenceIndex,
}) {
  const policy = manifestConversion?.precision_policy;
  fail(['fp16', 'int8-int16'].includes(policy),
    `${id} OBB precision_policy must be fp16 or int8-int16`);
  fail(manifestConversion.quantization === policy && build.conversion?.quantization === policy,
    `${id} release/build quantization must match the declared OBB precision policy`);
  fail(typeof manifestConversion.config_sha256 === 'string' && HASH_RE.test(manifestConversion.config_sha256),
    `${id} OBB compiler config SHA is missing`);
  fail(isAbsolute(buildReceiptRef?.path || '') && HASH_RE.test(buildReceiptRef?.sha256 || ''),
    `${id} OBB conversion receipt ref is malformed`);

  const campaignFile = await readJsonRef(campaignRef, `${id}.precision.campaign`);
  const campaign = campaignFile.json;
  fail(campaign.task === 'obb' && campaign.size === 'x'
    && campaign.precision_policy === policy
    && campaign.experiment?.receipt_quantization === policy
    && campaign.source_sha256 === build.conversion?.model_contract?.checkpoint_sha256,
  `${id} OBB campaign does not bind the build model and declared precision policy`);
  const campaignQuantConfig = campaign.experiment?.quant_config;
  const buildQuantConfig = build.conversion?.quant_config;
  fail(campaignQuantConfig && buildQuantConfig
    && stableJson(campaignQuantConfig) === stableJson(buildQuantConfig),
  `${id} OBB campaign quant_config differs from the conversion build receipt`);

  const configArtifact = build.artifacts?.config;
  fail(configArtifact?.name && basename(configArtifact.name) === configArtifact.name
    && HASH_RE.test(configArtifact.sha256 || '')
    && configArtifact.sha256 === manifestConversion.config_sha256,
  `${id} OBB compiler config artifact SHA differs from the release manifest`);
  const configPath = resolve(dirname(buildReceiptRef.path), configArtifact.name);
  const configBytes = await readRegularFile(configPath, `${id}.precision.compiler-config`);
  fail(digest(configBytes) === configArtifact.sha256
    && (!Number.isInteger(configArtifact.size_bytes) || configArtifact.size_bytes === configBytes.length),
    `${id} OBB compiler config bytes do not match the build receipt`);

  const quantConfig = campaignQuantConfig;
  fail(quantConfig.model_config && typeof quantConfig.model_config.all_node_type === 'string'
    && quantConfig.node_config && typeof quantConfig.node_config === 'object',
  `${id} OBB campaign quant_config is incomplete`);
  fail(stableJson(campaign.experiment?.fixed_activation)
    === stableJson(quantConfig.model_config.activation),
  `${id} OBB fixed activation policy differs from its compiler quant_config`);
  if (policy === 'fp16') {
    fail(quantConfig.model_config.all_node_type === 'float16'
      && Object.keys(quantConfig.node_config).length === 0,
    `${id} OBB fp16 policy does not match its compiler quant_config`);
    return;
  }
  fail(quantConfig.model_config.all_node_type === 'int16'
    && stableJson(quantConfig.model_config.activation) === stableJson({
      calibration_type: 'max', max_percentile: 1.0, per_channel: false, asymmetric: true,
    })
    && Object.keys(quantConfig.node_config).length === 181
    && Object.values(quantConfig.node_config).every(node => stableJson(node) === stableJson({
      input0: 'int16', input1: 'int16',
    }))
    && stableJson(quantConfig.op_config || {}) === stableJson({}),
  `${id} OBB int8-int16 policy does not match the approved max-1.0 input0/input1 INT16 recipe`);

  const mixedRef = build.artifacts?.mixed_precision_verification;
  const finalizationRef = finalizationEvidence?.mixed_precision_verification;
  const indexedRef = sourceEvidenceIndex?.mixed_precision_verification;
  fail(mixedRef?.name && basename(mixedRef.name) === mixedRef.name
    && HASH_RE.test(mixedRef.sha256 || '')
    && Number.isInteger(mixedRef.size_bytes) && mixedRef.size_bytes > 0,
  `${id} OBB int8-int16 build receipt lacks its mixed_precision_verification artifact ref`);
  const expectedPath = resolve(dirname(buildReceiptRef.path), mixedRef.name);
  fail(finalizationRef?.path === expectedPath && finalizationRef?.sha256 === mixedRef.sha256
    && indexedRef?.path === expectedPath && indexedRef?.sha256 === mixedRef.sha256,
  `${id} OBB mixed-precision proof is not exactly bound by build and finalization evidence`);
  const inspectionFile = await readJsonRef(indexedRef, `${id}.precision.mixed-proof`);
  fail(inspectionFile.bytes.length === mixedRef.size_bytes,
    `${id} OBB mixed-precision proof size differs from the build receipt`);
  const inspection = inspectionFile.json;
  const actual = inspection.actual_ptq;
  const ptqModel = build.provenance?.ptq_model;
  const finalizedPtqModel = finalizationEvidence?.ptq_model;
  const indexedPtqModel = sourceEvidenceIndex?.ptq_model;
  fail(!build.ptq_graph_adaptation && ptqModel && isAbsolute(ptqModel.path || '')
    && HASH_RE.test(ptqModel.sha256 || '') && Number.isInteger(ptqModel.size_bytes)
    && ptqModel.size_bytes > 0
    && finalizedPtqModel?.path === ptqModel.path && finalizedPtqModel?.sha256 === ptqModel.sha256
    && indexedPtqModel?.path === ptqModel.path && indexedPtqModel?.sha256 === ptqModel.sha256,
  `${id} native OBB int8-int16 build lacks a matching native PTQ provenance ref`);
  const compilerConfig = inspection.compiler_config;
  fail(inspection.schema_version === 1 && compilerConfig?.quant_config
    && compilerConfig.node_config && stableJson(compilerConfig.quant_config) === stableJson(campaignQuantConfig)
    && stableJson(compilerConfig.node_config) === stableJson(campaignQuantConfig.node_config),
  `${id} OBB mixed-precision proof does not cite the exact campaign compiler config`);
  fail(isAbsolute(compilerConfig.quant_config_path || '')
    && HASH_RE.test(compilerConfig.quant_config_sha256 || ''),
  `${id} OBB mixed-precision proof has no hashed quant_config input`);
  const runRoot = resolve(dirname(buildReceiptRef.path), '..');
  const quantConfigRelative = relative(resolve(runRoot, 'input'), compilerConfig.quant_config_path);
  fail(quantConfigRelative && !quantConfigRelative.startsWith('..') && !isAbsolute(quantConfigRelative),
  `${id} OBB mixed-precision quant_config input is outside the selected conversion run`);
  const quantConfigBytes = await readRegularFile(compilerConfig.quant_config_path,
    `${id}.precision.quant-config-input`);
  fail(digest(quantConfigBytes) === compilerConfig.quant_config_sha256,
    `${id} OBB mixed-precision quant_config input SHA mismatch`);
  let quantConfigJson;
  try { quantConfigJson = JSON.parse(quantConfigBytes.toString('utf8')); } catch (error) {
    throw new Error(`Rebuild selection: ${id} quant_config input is invalid JSON: ${error.message}`);
  }
  fail(stableJson(quantConfigJson) === stableJson(campaignQuantConfig),
    `${id} OBB mixed-precision quant_config bytes differ from the campaign`);
  fail(actual?.precision_policy_pass === true
    && actual?.onnx_path === ptqModel.path
    && actual?.onnx_sha256 === ptqModel.sha256
    && actual?.onnx_size_bytes === ptqModel.size_bytes,
  `${id} OBB mixed-precision proof does not validate the exact compiled PTQ graph`);
  const qtypes = actual?.actual_qtype_counts;
  fail(qtypes && qtypes.int8 === 4 && qtypes.int16 === 639
    && Object.entries(qtypes).every(([qtype, count]) => ['int8', 'int16'].includes(qtype)
      && Number.isInteger(count) && count > 0),
  `${id} OBB mixed-precision proof does not show real int8 and int16 PTQ nodes`);
  fail(actual.conv_count === 175 && actual.matmul_count === 6 && actual.gemm_count === 0
    && actual.configured_node_count === 181 && actual.conv_input_qtypes_pass === true
    && actual.conv_weight_scales_pass === true && actual.allowed_int8_exceptions_pass === true
    && actual.configured_node_type_counts?.Conv === 175
    && actual.configured_node_type_counts?.MatMul === 6
    && actual.configured_node_type_counts?.Gemm === undefined
    && actual.quant_info?.output_count === 9 && actual.quant_info?.all_outputs_present === true,
  `${id} OBB mixed-precision proof does not pass the expected graph/qparam/output checks`);
}

function jsonPointer(value, pointer, label) {
  fail(typeof pointer === 'string' && (pointer === '' || pointer.startsWith('/')),
    `${label} must be an RFC 6901 JSON pointer`);
  if (!pointer) return value;
  return pointer.slice(1).split('/').map(token => token.replace(/~1/g, '/').replace(/~0/g, '~'))
    .reduce((current, key) => current?.[key], value);
}

async function loadWindow(path, key) {
  const bytes = await readRegularFile(path, path);
  const context = { window: {} };
  runInNewContext(bytes.toString('utf8'), context, { timeout: 5000 });
  fail(context.window[key] && typeof context.window[key] === 'object', `${path} did not define window.${key}`);
  return { value: plain(context.window[key]), bytes };
}

function assertComparison(row, record) {
  const modelTask = row.catalogId.split('/').at(-1);
  const accuracyRows = row.benchmark?.accuracy || [];
  const expectedMetric = modelTask === 'pose' ? 'keypoints-all-map-50-95' : 'obb-map-50';
  const field = modelTask === 'pose' ? 'keypoints_ap' : 'map_50';
  const float = accuracyRows.find(item => item.metric === expectedMetric && item.model_stage === 'float');
  const board = accuracyRows.find(item => item.metric === expectedMetric && item.model_stage === 'quantized');
  fail(float?.unit === 'ratio' && board?.unit === 'ratio', `${record.release_id} must have paired float and board accuracy`);
  fail(Number.isFinite(float.value) && Number.isFinite(board.value)
    && float.value >= 0 && float.value <= 1 && board.value >= 0 && board.value <= 1,
  `${record.release_id} accuracy values must be finite ratios`);
  fail(float.dataset && board.dataset && float.dataset === board.dataset,
    `${record.release_id} float/board metrics must name one dataset`);
  fail(float.comparison_scope?.comparable === true && board.comparison_scope?.comparable === true
    && float.comparison_scope?.kind === 'end-to-end-metric-comparison'
    && float.comparison_scope?.not_a_pure_quantization_loss_estimate === true,
  `${record.release_id} comparison scope must be reviewed end-to-end evidence`);

  const comparison = record.evidence.comparison_receipt.json;
  const direct = comparison.direct_metric_comparison
    || comparison.comparison?.direct_metric_comparison
    || comparison.accuracy?.comparison_status;
  fail(direct === 'valid', `${record.release_id} comparison receipt does not say direct_metric_comparison=valid`);
  const scope = comparison.comparison_scope || comparison.comparison?.comparison_scope;
  if (scope) {
    fail(scope.comparable === true && scope.kind === 'end-to-end-metric-comparison'
      && scope.not_a_pure_quantization_loss_estimate === true,
    `${record.release_id} comparison receipt has a mismatched comparison scope`);
  }
  const metrics = comparison.metrics || comparison.accuracy?.comparison || {};
  const comparisonMetric = metrics[field] || metrics[modelTask === 'pose' ? 'keypoints/AP' : 'mAP50'];
  fail(comparisonMetric, `${record.release_id} comparison receipt is missing its displayed metric`);
  fail(Math.abs(Number(comparisonMetric.float) - float.value) <= 1e-9
    && Math.abs(Number(comparisonMetric.board) - board.value) <= 1e-9
    && Math.abs(Number(comparisonMetric.board_minus_float) - (board.value - float.value)) <= 1e-9,
  `${record.release_id} web accuracy does not match its comparison receipt`);

  const binding = record.metric_bindings?.[field];
  fail(binding && binding.float && binding.runtime,
    `${record.release_id} must bind float and runtime metric JSON pointers`);
  for (const [stage, sourceRole, expected] of [
    ['float', binding.float.source, float.value], ['runtime', binding.runtime.source, board.value],
  ]) {
    const source = record.evidence[sourceRole];
    fail(source, `${record.release_id} metric binding references missing evidence ${sourceRole}`);
    const value = jsonPointer(source.json, binding[stage].pointer, `${record.release_id}.${stage} metric binding`);
    fail(typeof value === 'number' && Math.abs(value - expected) <= 1e-9,
      `${record.release_id} ${stage} metric differs from cited evidence`);
  }
}

function assertPerformance(row, record) {
  const perf = row.benchmark || {};
  const native = record.evidence.native_performance.json;
  const nativeBinding = record.evidence.native_perf_binding.json;
  const artifact = row.assets?.[0];
  const nativeBoundSha = nativeBinding.model?.sha256
    || nativeBinding.model_sha256
    || nativeBinding.artifact?.sha256;
  fail(nativeBoundSha === artifact.sha256,
    `${record.release_id} native perf binding does not identify this HBM SHA`);
  fail(perf.timing?.tool === native.tool && perf.timing?.implementation === native.implementation,
    `${record.release_id} native perf tool/implementation differs from performance.json`);

  const expectedRuntime = (native.measurements || []).flatMap(item => [
    {
      metric: 'latency', value: Number(item.average_latency_ms), unit: 'ms',
      concurrency: Number(item.threads), observedMin: Number(item.observed_min_latency_ms),
      observedMax: Number(item.observed_max_latency_ms),
    },
    { metric: 'throughput', value: Number(item.aggregate_fps), unit: 'fps', concurrency: Number(item.threads) },
  ]);
  fail(JSON.stringify(perf.performance) === JSON.stringify(expectedRuntime),
    `${record.release_id} native runtime metrics differ from performance.json`);

  const e2e = perf.endToEnd || [];
  fail(Array.isArray(e2e) && e2e.length === 2, `${record.release_id} must contain measured C++ single/dual pipeline results`);
  const sources = [record.evidence.cpp_e2e_single.json, record.evidence.cpp_e2e_dual.json];
  for (const source of sources) {
    const streams = Number(source.pipeline_streams);
    const entry = e2e.find(item => Number(item.timing?.pipelineStreams) === streams);
    fail(entry && streams === Number(source.runtime_submission_threads),
      `${record.release_id} is missing C++ end-to-end pipeline ${streams}`);
    for (const metric of ['preprocess', 'runtime', 'postprocess', 'end_to_end']) {
      const targetMetric = entry.metrics?.[metric === 'end_to_end' ? 'endToEnd' : metric];
      const sourceMetric = source.metrics_ms?.[metric];
      fail(targetMetric && sourceMetric && ['mean', 'p50', 'p95', 'min', 'max']
        .every(key => Math.abs(Number(targetMetric[key === 'mean' ? 'value' : key]) - Number(sourceMetric[key])) <= 1e-9),
      `${record.release_id} C++ ${streams}-stream ${metric} differs from source JSON`);
    }
    fail(Math.abs(Number(entry.throughputFps) - Number(source.throughput_fps)) <= 1e-9,
      `${record.release_id} C++ ${streams}-stream throughput differs from source JSON`);
  }
  const cppProvenance = record.evidence.cpp_e2e_provenance.json;
  fail((cppProvenance.model?.sha256 || cppProvenance.model_sha256) === artifact.sha256,
    `${record.release_id} C++ provenance does not bind this HBM SHA`);
}

async function validateRecord(record, model) {
  const releaseIdExpected = `${model.catalogId}/${model.modelSize}/${model.releasePlatform.toLowerCase()}`;
  fail(record.release_id === releaseIdExpected, `record ID does not match UI model ${model.id}`);
  fail(model.id === expectedUiId(model.catalogId.split('/').at(-1), model.modelSize, model.releasePlatform),
    `${record.release_id} has an unexpected UI model ID`);
  fail(model.releaseStatus === 'released', `${record.release_id} is not finalized as released`);

  const files = {};
  for (const role of REQUIRED_FILES) files[role] = await verifyRef(record.files?.[role], `${record.release_id}.${role}`);
  const evidence = {};
  for (const role of REQUIRED_EVIDENCE) evidence[role] = await readJsonRef(record.evidence?.[role], `${record.release_id}.${role}`);
  record.evidence = evidence;

  fail(record.source_evidence_index && typeof record.source_evidence_index === 'object'
    && !Array.isArray(record.source_evidence_index),
  `${record.release_id} must preserve the finalization source evidence index`);
  const sourceEvidence = {};
  for (const [role, ref] of Object.entries(record.source_evidence_index)) {
    sourceEvidence[role] = await verifyRef(ref, `${record.release_id}.source.${role}`);
  }
  for (const [evidenceRole, sourceRole] of [
    ['build_receipt', 'conversion_receipt'], ['board_receipt', 'board_receipt'],
    ['float_reference', 'float_result_reference'], ['comparison_receipt', 'comparison_receipt'],
    ['native_performance', 'runtime_performance'], ['native_perf_binding', 'runtime_performance_binding'],
    ['cpp_e2e_single', 'cpp_e2e_single'], ['cpp_e2e_dual', 'cpp_e2e_dual'],
    ['cpp_e2e_provenance', 'cpp_e2e_provenance'],
  ]) {
    fail(sourceEvidence[sourceRole]
      && sourceEvidence[sourceRole].sha256 === evidence[evidenceRole].sha256
      && sourceEvidence[sourceRole].path === evidence[evidenceRole].path,
    `${record.release_id} source evidence index does not preserve ${sourceRole}`);
  }
  record.source_evidence_index = Object.fromEntries(Object.entries(sourceEvidence)
    .map(([role, ref]) => [role, { path: ref.path, sha256: ref.sha256 }]));

  const uploads = {};
  for (const role of REQUIRED_UPLOADS) uploads[role] = await readJsonRef(record.upload_receipts?.[role], `${record.release_id}.upload.${role}`);
  record.upload_receipts = uploads;

  const [task, size, platform] = [model.catalogId.split('/').at(-1), model.modelSize, model.releasePlatform.toLowerCase()];
  const asset = model.assets?.[0];
  fail(model.assets?.length === 1 && asset?.format === 'hbm', `${record.release_id} must expose exactly one HBM download`);
  fail(basename(files.artifact_file.path) === asset.filename,
    `${record.release_id} local HBM filename differs from the download name`);
  fail(asset.sha256 === digest(files.artifact_file.bytes) && asset.sizeBytes === files.artifact_file.bytes.length,
    `${record.release_id} HBM UI metadata differs from the local artifact`);

  const manifest = JSON.parse(files.release_manifest.bytes.toString('utf8'));
  fail(manifest.status === 'released' && manifest.publication?.status === 'published',
    `${record.release_id} release manifest is not finalized/published`);
  fail(manifest.model?.family === 'yolo26' && manifest.model?.task === task
    && manifest.model?.size === size && manifest.target?.platform === platform,
  `${record.release_id} release manifest identity mismatch`);
  const releaseBase = `models/ultralytics_yolo/yolo26/${task}/${size}/${platform}`;
  const objectPrefix = manifest.publication?.object_prefix;
  fail(objectPrefix === `${releaseBase}/rebuilds/${record.build_id}`,
    `${record.release_id} OSS object prefix must be the build-versioned rebuild path`);
  const artifactObject = manifest.publication?.uploaded_objects?.artifact;
  const htmlObject = manifest.publication?.uploaded_objects?.oe_report_html;
  const dataObject = manifest.publication?.uploaded_objects?.oe_report_data;
  fail(artifactObject?.key === `${objectPrefix}/${asset.filename}`
    && htmlObject?.key === `${objectPrefix}/oe_report.html`
    && dataObject?.key === `${objectPrefix}/oe_report_data.json`,
  `${record.release_id} published object keys do not match the versioned prefix`);
  for (const object of [artifactObject, htmlObject, dataObject]) {
    fail(object?.url === `${OSS_BASE}/${object.key}`,
      `${record.release_id} manifest has a noncanonical public object URL`);
  }
  const expectedUrl = artifactObject.url;
  fail(asset.url === expectedUrl && model.reportSourceUrl === htmlObject.url,
    `${record.release_id} Web download/report URL differs from the finalized release`);
  fail(manifest.artifact?.sha256 === asset.sha256 && manifest.artifact?.size_bytes === asset.sizeBytes
    && manifest.artifact?.url === asset.url,
  `${record.release_id} release manifest HBM differs from UI download`);
  const releaseManifestKey = `${objectPrefix}/release.json`;
  const checksumsKey = `${objectPrefix}/SHA256SUMS`;
  fail(manifest.artifact?.release_manifest_url === `${OSS_BASE}/${releaseManifestKey}`
    && manifest.artifact?.checksums_url === `${OSS_BASE}/${checksumsKey}`,
  `${record.release_id} release manifest/checksum URLs do not use the versioned prefix`);
  fail(manifest.reports?.oe_report_html_url === model.reportSourceUrl,
    `${record.release_id} release manifest OE HTML URL differs from UI`);
  fail(manifest.reports?.oe_report_data_url === dataObject.url,
    `${record.release_id} release manifest OE data URL is not canonical`);
  const expectedPrecision = manifest.conversion?.precision_policy || manifest.conversion?.quantization;
  const displayedPrecision = typeof expectedPrecision === 'string'
    ? expectedPrecision : JSON.stringify(expectedPrecision);
  fail(typeof displayedPrecision === 'string' && model.benchmark?.precision === displayedPrecision
    && model.benchmark?.precisionPolicy === (manifest.conversion?.precision_policy ?? null)
    && JSON.stringify(model.benchmark?.quantization) === JSON.stringify(manifest.conversion?.quantization),
  `${record.release_id} displayed precision does not match finalized conversion metadata`);

  const oeData = JSON.parse(files.oe_report_data.bytes.toString('utf8'));
  fail(oeData.provenance?.artifact_sha256 === asset.sha256,
    `${record.release_id} OE report data does not bind this HBM SHA`);
  const sourceDataPath = resolve(record.directory, model.reportDataUrl || '');
  const sourceDataRelative = relative(record.directory, sourceDataPath);
  fail(sourceDataRelative && !sourceDataRelative.startsWith('..') && !sourceDataRelative.startsWith('/'),
    `${record.release_id} reportDataUrl escapes rebuild selection directory`);
  const sourceDataBytes = await readRegularFile(sourceDataPath, `${record.release_id}.preview OE data`);
  fail(digest(sourceDataBytes) === digest(files.oe_report_data.bytes),
    `${record.release_id} preview OE report does not match finalized bundle`);

  const sumLines = files.checksums.bytes.toString('utf8').trim().split(/\r?\n/);
  const sums = new Map();
  for (const line of sumLines) {
    const match = /^([a-f0-9]{64})  ([A-Za-z0-9_.-]+)$/.exec(line);
    fail(match && !sums.has(match[2]), `${record.release_id} SHA256SUMS has a malformed/duplicate entry`);
    sums.set(match[2], match[1]);
  }
  const expectedSums = new Map([
    [asset.filename, asset.sha256],
    ['oe_report.html', digest(files.oe_report_html.bytes)],
    ['oe_report_data.json', digest(files.oe_report_data.bytes)],
    ['release.json', digest(files.release_manifest.bytes)],
  ]);
  fail(sums.size === expectedSums.size && [...expectedSums].every(([name, sha]) => sums.get(name) === sha),
    `${record.release_id} SHA256SUMS does not match the finalized package`);

  const expectedObjects = {
    artifact: { key: artifactObject.key, sha256: asset.sha256, size_bytes: asset.sizeBytes, url: artifactObject.url },
    oe_report_html: { key: htmlObject.key, sha256: digest(files.oe_report_html.bytes), size_bytes: files.oe_report_html.bytes.length, url: htmlObject.url },
    oe_report_data: { key: dataObject.key, sha256: digest(files.oe_report_data.bytes), size_bytes: files.oe_report_data.bytes.length, url: dataObject.url },
    release_manifest: { key: releaseManifestKey, sha256: digest(files.release_manifest.bytes), size_bytes: files.release_manifest.bytes.length, url: `${OSS_BASE}/${releaseManifestKey}` },
    checksums: { key: checksumsKey, sha256: digest(files.checksums.bytes), size_bytes: files.checksums.bytes.length, url: `${OSS_BASE}/${checksumsKey}` },
  };
  for (const [role, embedded] of Object.entries({
    artifact: artifactObject, oe_report_html: htmlObject, oe_report_data: dataObject,
  })) {
    const expected = expectedObjects[role];
    fail(embedded?.sha256 === expected.sha256 && embedded?.size_bytes === expected.size_bytes
      && embedded?.acl === 'public-read',
    `${record.release_id} release manifest does not preserve exact ${role} checksum/ACL`);
  }
  for (const [role, expected] of Object.entries(expectedObjects)) {
    const receipt = uploads[role].json;
    fail(receipt.status === 'verified' && receipt.acl === 'public-read'
      && Object.entries(expected).every(([field, value]) => receipt[field] === value),
    `${record.release_id} ${role} upload receipt does not verify the exact public object`);
  }

  const build = evidence.build_receipt.json;
  const buildModel = build.artifacts?.model;
  fail(build.status === 'host-compiled' && build.build_id === record.build_id
    && build.model?.family === 'yolo26' && build.model?.task === task && build.model?.size === size
    && build.target?.platform === platform,
  `${record.release_id} build receipt identity/status mismatch`);
  fail(buildModel?.sha256 === asset.sha256 && buildModel?.size_bytes === asset.sizeBytes,
    `${record.release_id} build receipt does not bind the published HBM`);
  if (task === 'obb') {
    const finalizationSources = evidence.finalization_receipt.json.evidence_sources || {};
    if (manifest.conversion?.precision_policy === 'fp16') {
      const graph = build.ptq_graph_adaptation;
      fail(graph && graph.source_ptq && graph.adapted_ptq && graph.adaptor && graph.proof,
        `${record.release_id} FP16 Swish build receipt lacks PTQ graph-adaptation refs`);
      for (const [role, expected] of Object.entries({
        ptq_source_original: graph.source_ptq,
        ptq_source_adapted: graph.adapted_ptq,
        ptq_graph_adaptation_proof: graph.proof,
        ptq_graph_adaptation_code: graph.adaptor,
      })) {
        const source = record.source_evidence_index[role];
        const finalized = finalizationSources[role];
        fail(source?.path === expected.path && source?.sha256 === expected.sha256
          && finalized?.path === expected.path && finalized?.sha256 === expected.sha256,
        `${record.release_id} finalization source evidence ${role} differs from FP16 build receipt`);
      }
    } else {
      fail(manifest.conversion?.precision_policy === 'int8-int16' && !build.ptq_graph_adaptation,
        `${record.release_id} native OBB integer build must not claim PTQ graph adaptation`);
      const ptqModel = build.provenance?.ptq_model;
      const indexed = record.source_evidence_index.ptq_model;
      const finalized = finalizationSources.ptq_model;
      fail(ptqModel && indexed?.path === ptqModel.path && indexed?.sha256 === ptqModel.sha256
        && finalized?.path === ptqModel.path && finalized?.sha256 === ptqModel.sha256,
      `${record.release_id} native PTQ provenance differs from finalization evidence`);
    }
    fail(typeof record.campaign_sha256 === 'string' && HASH_RE.test(record.campaign_sha256),
      `${record.release_id} is missing its selected OBB campaign SHA`);
    const campaignFile = await verifyRef(record.campaign_manifest,
      `${record.release_id}.campaign_manifest`);
    fail(campaignFile.sha256 === record.campaign_sha256,
      `${record.release_id} selected campaign file differs from its recorded SHA`);
    const rebuildSelectionRef = record.source_evidence_index.rebuild_selection;
    const rebuildSelection = JSON.parse((await verifyRef(rebuildSelectionRef,
      `${record.release_id}.source.rebuild_selection`)).bytes.toString('utf8'));
    const campaignSha = rebuildSelection.lineage?.rebuild_campaign_sha256
      || rebuildSelection.rebuild?.campaign?.sha256 || rebuildSelection.campaign?.sha256;
    fail(campaignSha === record.campaign_sha256,
      `${record.release_id} rebuild-selection campaign SHA differs from the selected runs manifest`);
    const selectedConversion = rebuildSelection.rebuild?.conversion_receipt
      || rebuildSelection.conversion_receipt;
    fail(selectedConversion?.path === evidence.build_receipt.path
      && selectedConversion?.sha256 === evidence.build_receipt.sha256,
    `${record.release_id} rebuild selection does not bind the selected conversion receipt`);
    const selectedCampaign = rebuildSelection.rebuild?.campaign || rebuildSelection.campaign;
    fail(selectedCampaign?.path === campaignFile.path && selectedCampaign?.sha256 === record.campaign_sha256,
      `${record.release_id} rebuild selection does not bind the selected campaign file`);
    await validateObbPrecisionEvidence({
      releaseId: record.release_id,
      manifestConversion: manifest.conversion,
      build,
      buildReceiptRef: evidence.build_receipt,
      campaignRef: campaignFile,
      finalizationEvidence: evidence.finalization_receipt.json.evidence_sources,
      sourceEvidenceIndex: record.source_evidence_index,
    });
  }

  const board = evidence.board_receipt.json;
  fail(board.model?.family === 'yolo26' && board.model?.task === task && board.model?.size === size
    && board.model?.platform === platform && board.model?.sha256 === asset.sha256
    && board.model?.board_sha256_verified === true,
  `${record.release_id} board evaluation does not bind this HBM`);
  fail(board.conversion?.receipt_sha256 === evidence.build_receipt.sha256,
    `${record.release_id} board receipt points to another conversion receipt`);

  const finalization = evidence.finalization_receipt.json;
  fail(finalization.kind === 'yolo26-task-release-finalization'
    && finalization.model?.task === task && finalization.model?.size === size
    && finalization.target?.platform === platform,
  `${record.release_id} finalization receipt identity mismatch`);
  fail(finalization.finalized_bundle?.release_manifest_sha256 === digest(files.release_manifest.bytes)
    && finalization.finalized_bundle?.sha256sums_sha256 === digest(files.checksums.bytes),
  `${record.release_id} finalization receipt does not bind this release bundle`);

  const embeddedUploads = manifest.publication?.uploaded_objects || {};
  for (const role of ['artifact', 'oe_report_html', 'oe_report_data']) {
    const embedded = embeddedUploads[role];
    const receipt = uploads[role].json;
    fail(embedded?.key === receipt.key && embedded?.url === receipt.url
      && embedded?.sha256 === receipt.sha256 && embedded?.size_bytes === receipt.size_bytes
      && embedded?.acl === receipt.acl,
    `${record.release_id} release manifest does not preserve the verified ${role} receipt`);
  }

  assertComparison(model, record);
  assertPerformance(model, record);
  return {
    releaseId: releaseId(model), model, record, files, evidence,
    sourceEvidence: record.source_evidence_index, uploads,
  };
}

export async function validateSingleRebuildRecord({ record, model, directory }) {
  fail(typeof directory === 'string' && isAbsolute(directory), 'single-record directory must be absolute');
  const normalized = { ...record, directory };
  return validateRecord(normalized, model);
}

export async function loadVerifiedRebuildSelection({ directory, snapshotId, auditSha, expectedReleaseIds = null }) {
  fail(typeof directory === 'string' && isAbsolute(directory), 'selection directory must be an absolute path');
  const dirInfo = await lstat(directory);
  fail(dirInfo.isDirectory() && !dirInfo.isSymbolicLink(), 'selection directory must be a real directory');
  const manifestPath = resolve(directory, 'rebuild-selection.json');
  const manifestBytes = await readRegularFile(manifestPath, 'rebuild-selection.json');
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  fail(manifest.schema_version === 1 && manifest.kind === 'yolo26-rebuild-selection'
    && manifest.status === 'ready-for-merge', 'unsupported or not-ready rebuild-selection.json');
  fail(manifest.base_snapshot_id === snapshotId && manifest.base_authoritative_audit_sha256 === auditSha,
    'rebuild selection must bind the exact 56-row frozen candidate snapshot and audit');
  fail(Array.isArray(manifest.records), 'records must be an array');
  const selectedIds = expectedReleaseIds || manifest.excluded_release_ids;
  fail(Array.isArray(selectedIds) && selectedIds.length > 0
    && selectedIds.every(id => EXPECTED_RELEASES.includes(id)),
  'rebuild selection must name one or more of the four authorized repaired releases');
  const sortedSelectedIds = [...selectedIds].sort();
  fail(new Set(sortedSelectedIds).size === sortedSelectedIds.length,
    'rebuild selection target list contains duplicates');
  fail(JSON.stringify(manifest.excluded_release_ids) === JSON.stringify(selectedIds),
    'excluded_release_ids must exactly identify the selected rebuild rows in order');
  const releaseIds = manifest.records.map(record => record.release_id).sort();
  fail(JSON.stringify(releaseIds) === JSON.stringify(sortedSelectedIds),
    `records must exactly match the selected targets: ${sortedSelectedIds.join(', ')}`);
  const inventory = manifest.rebuild_input_inventory;
  fail(inventory && typeof inventory === 'object' && !Array.isArray(inventory),
    'rebuild input inventory is missing');
  const poseId = 'ultralytics_yolo/yolo26/pose/s/s100p';
  const obbIds = sortedSelectedIds.filter(id => id.includes('/obb/'));
  fail(Boolean(inventory.pose_conversion_receipt) === sortedSelectedIds.includes(poseId),
    'Pose conversion input must be present only when Pose is selected');
  let poseBuild;
  if (inventory.pose_conversion_receipt) {
    poseBuild = await readJsonRef(inventory.pose_conversion_receipt, 'Pose rebuild input conversion receipt');
    fail(poseBuild.json.status === 'host-compiled' && poseBuild.json.build_id,
      'Pose rebuild input conversion receipt is not compiled');
  }
  fail(Boolean(inventory.obb_runs_manifest) === (obbIds.length > 0)
    && Boolean(inventory.obb_conversion_receipts) === (obbIds.length > 0),
  'OBB run manifest and receipts must be present exactly when OBB targets are selected');
  let obbRuns;
  const obbBuildReceipts = {};
  if (obbIds.length) {
    const receiptIds = Object.keys(inventory.obb_conversion_receipts).sort();
    fail(JSON.stringify(receiptIds) === JSON.stringify(obbIds),
      'OBB conversion receipt inventory must contain exactly the selected targets');
    obbRuns = await readJsonRef(inventory.obb_runs_manifest, 'OBB rebuild runs manifest');
    fail(Array.isArray(obbRuns.json.runs), 'OBB runs manifest is missing its run rows');
    const obbRows = new Map();
    for (const runRow of obbRuns.json.runs) {
      const platform = String(runRow.platform || '').toLowerCase();
      fail(['s600', 's100p', 's100'].includes(platform) && !obbRows.has(platform),
        `OBB runs manifest has a duplicate/unsupported platform: ${platform}`);
      obbRows.set(platform, runRow);
    }
    const selectedPlatforms = obbIds.map(id => id.split('/').at(-1));
    fail(selectedPlatforms.every(platform => obbRows.has(platform)),
      'OBB runs manifest is missing a selected target');
    for (const releaseId of obbIds) {
      const ref = inventory.obb_conversion_receipts[releaseId];
      const receiptRef = { path: ref.path, sha256: ref.sha256 };
      const receipt = await readJsonRef(receiptRef, `${releaseId} rebuild input conversion receipt`);
      const platform = releaseId.split('/').at(-1);
      const manifestRow = obbRows.get(platform);
      const expectedBuildId = manifestRow?.build_id || obbRuns.json.build_id;
      const expectedCampaignSha = manifestRow?.campaign_sha256 || obbRuns.json.campaign_sha256;
      fail(manifestRow && expectedBuildId === ref.build_id
        && expectedCampaignSha === ref.campaign_sha256
        && ref.campaign_manifest?.sha256 === expectedCampaignSha
        && ref.campaign_manifest?.path === resolve(manifestRow.conversion || '', 'input/campaign.json')
        && ref.runs_manifest?.path === inventory.obb_runs_manifest.path
        && ref.runs_manifest?.sha256 === inventory.obb_runs_manifest.sha256
        && Boolean(ref.source_manifest) === Boolean(manifestRow.source_manifest)
        && (!manifestRow.source_manifest || (ref.source_manifest.path === manifestRow.source_manifest.path
          && ref.source_manifest.sha256 === manifestRow.source_manifest.sha256)),
      `${releaseId} input inventory does not preserve its exact target run/campaign binding`);
      const campaignManifest = await readJsonRef(ref.campaign_manifest, `${releaseId} exact campaign manifest`);
      fail(campaignManifest.sha256 === expectedCampaignSha,
        `${releaseId} campaign manifest SHA differs from the target runs-manifest row`);
      fail(receipt.json.build_id === expectedBuildId && HASH_RE.test(expectedCampaignSha || '')
        && receipt.json.status === 'host-compiled'
        && receipt.json.model?.task === 'obb' && receipt.json.model?.size === 'x'
        && receipt.json.target?.platform === platform
        && receipt.path === resolve(manifestRow.conversion || '', 'output/build_receipt.json'),
      `${releaseId} conversion receipt differs from its exact runs-manifest target row`);
      if (ref.source_manifest) {
        const sourceManifest = await readJsonRef(ref.source_manifest, `${releaseId} source run manifest`);
        const sourceRows = Array.isArray(sourceManifest.json.runs)
          ? sourceManifest.json.runs.filter(item => String(item.platform || '').toLowerCase() === platform)
          : [sourceManifest.json];
        fail(sourceRows.length === 1, `${releaseId} source manifest must identify exactly one target row`);
        const sourceRow = sourceRows[0];
        fail((sourceRow.build_id || sourceManifest.json.build_id) === expectedBuildId
          && (sourceRow.campaign_sha256 || sourceManifest.json.campaign_sha256) === expectedCampaignSha
          && (sourceRow.conversion || sourceManifest.json.conversion) === manifestRow.conversion,
        `${releaseId} source manifest does not identify the selected build/campaign/conversion`);
      }
      obbBuildReceipts[releaseId] = { ...receipt, build_id: expectedBuildId,
        campaign_sha256: expectedCampaignSha, manifest_row: manifestRow };
    }
  }
  for (const row of manifest.records) {
    const isPose = row.release_id === poseId;
    const expectedBuild = isPose ? poseBuild?.json : obbBuildReceipts[row.release_id];
    const expectedBuildRef = isPose
      ? inventory.pose_conversion_receipt : inventory.obb_conversion_receipts[row.release_id];
    fail(expectedBuild && expectedBuild.build_id === row.build_id
      && row.evidence?.build_receipt?.path === expectedBuildRef.path
      && row.evidence?.build_receipt?.sha256 === expectedBuildRef.sha256,
    `${row.release_id} rebuild row does not bind its authorized run inventory`);
    if (row.release_id.includes('/obb/')) {
      const platform = row.release_id.split('/').at(-1);
      const inputRef = inventory.obb_conversion_receipts[row.release_id];
      fail(expectedBuild.build_id === row.build_id
        && expectedBuild.campaign_sha256 === row.campaign_sha256
        && row.campaign_manifest?.path === inputRef.campaign_manifest.path
        && row.campaign_manifest?.sha256 === inputRef.campaign_manifest.sha256
        && row.runs_manifest?.path === inventory.obb_runs_manifest.path
        && row.runs_manifest?.sha256 === inventory.obb_runs_manifest.sha256
        && Boolean(row.source_manifest) === Boolean(inputRef.source_manifest)
        && (!inputRef.source_manifest || (row.source_manifest.path === inputRef.source_manifest.path
          && row.source_manifest.sha256 === inputRef.source_manifest.sha256))
        && expectedBuild.json.model?.task === 'obb' && expectedBuild.json.model?.size === 'x'
        && expectedBuild.json.target?.platform === platform,
      `${row.release_id} conversion receipt does not match its OBB manifest target`);
    }
  }

  const dataPath = resolve(directory, 'data.js');
  const propertiesPath = resolve(directory, 'model-properties.js');
  const reportsPath = resolve(directory, 'reports/reports-data.js');
  const sourceIndexPath = resolve(directory, 'source-evidence-index.json');
  const [{ value: dataWindow }, { value: propertiesWindow }, { value: reportsWindow }, sourceIndexFile] = await Promise.all([
    loadWindow(dataPath, 'MODEL_DATA'),
    loadWindow(propertiesPath, 'MODEL_PROPERTIES'),
    loadWindow(reportsPath, 'OE_REPORTS'),
    readRegularFile(sourceIndexPath, 'source-evidence-index.json'),
  ]);
  fail(HASH_RE.test(manifest.source_evidence_index_sha256 || '')
    && digest(sourceIndexFile) === manifest.source_evidence_index_sha256,
  'source-evidence-index.json SHA-256 does not match rebuild-selection.json');
  const sourceIndex = JSON.parse(sourceIndexFile.toString('utf8'));
  fail(sourceIndex.schema_version === 1 && sourceIndex.kind === 'yolo26-rebuild-source-evidence-index',
    'unsupported source-evidence-index.json');
  const expectedSourceRows = manifest.records.map(record => ({
    release_id: record.release_id,
    build_id: record.build_id,
    evidence_sources: record.source_evidence_index,
  }));
  fail(JSON.stringify(sourceIndex.records) === JSON.stringify(expectedSourceRows),
    'source-evidence-index.json does not preserve the exact release evidence refs');
  const models = dataWindow.models || [];
  const properties = propertiesWindow;
  const reports = reportsWindow;
  fail(dataWindow.catalog?.status === 'published', 'rebuild output must be a normal release build, not a candidate catalog');
  fail(models.length === sortedSelectedIds.length && reports.length === sortedSelectedIds.length,
    'rebuild output must contain exactly one model row and OE report per selected target');
  fail(Object.keys(properties).length === sortedSelectedIds.length,
    'rebuild output must contain exactly one model-property row per selected target');
  fail(new Set(models.map(model => model.id)).size === sortedSelectedIds.length,
    'duplicate UI model IDs in rebuild output');

  const checkedRecords = [];
  for (const record of manifest.records) {
    record.directory = directory;
    const model = models.find(item => releaseId(item) === record.release_id);
    fail(model, `missing rendered UI row for ${record.release_id}`);
    fail(Object.hasOwn(properties, model.id), `missing model properties for ${record.release_id}`);
    const report = reports.find(item => item.modelIds?.length === 1 && item.modelIds[0] === model.id);
    fail(report, `missing OE report registry row for ${record.release_id}`);
    const checked = await validateRecord(record, model);
    fail(model.reportSourceUrl === report.path || model.reportSourceUrl === report.sources?.[0]
      || report.path === model.reportSourceUrl,
    `${record.release_id} report registry does not link the published OE report`);
    checkedRecords.push(checked);
  }
  return {
    directory,
    manifest,
    manifestPath,
    manifestSha256: digest(manifestBytes),
    data: dataWindow,
    properties,
    reports,
    models,
    records: checkedRecords,
  };
}

export function expectedRebuildReleaseIds() {
  return [...EXPECTED_RELEASES];
}
