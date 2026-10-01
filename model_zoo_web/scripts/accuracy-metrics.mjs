import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptRoot, '..');
export const accuracyMetricsSchema = JSON.parse(
  await readFile(resolve(webRoot, 'accuracy-metrics.json'), 'utf8'),
);

export function canonicalAccuracyTask(task) {
  return accuracyMetricsSchema.task_aliases?.[task] || task;
}

export function accuracyMetricLabels() {
  return Object.fromEntries(
    Object.values(accuracyMetricsSchema.tasks)
      .flatMap(profile => profile.display)
      .map(({ metric, labels }) => [metric, labels]),
  );
}

export function accuracyRecords(task, floatAccuracy, runtimeAccuracy, accuracyMetadata = {}) {
  const canonicalTask = canonicalAccuracyTask(task);
  const profile = accuracyMetricsSchema.tasks[canonicalTask];
  if (!profile) throw new Error(`Unsupported accuracy task: ${task}`);
  if (profile.dataset && accuracyMetadata.dataset !== profile.dataset) {
    throw new Error(`${task} accuracy dataset must be ${profile.dataset}`);
  }
  const evaluationScope = accuracyMetadata.evaluation_scope;
  if (profile.evaluation_scope && evaluationScope !== profile.evaluation_scope.value) {
    throw new Error(`${task} accuracy evaluation_scope must be ${profile.evaluation_scope.value}`);
  }

  const records = [];
  for (const { field, metric } of profile.display) {
    for (const [source, stage] of [[floatAccuracy, 'float'], [runtimeAccuracy, 'quantized']]) {
      const value = source?.[field];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error(`${task} accuracy ${stage}.${field} must be a finite ratio between 0 and 1`);
      }
      records.push({
        metric,
        value,
        unit: 'ratio',
        model_stage: stage,
        ...(profile.evaluation_scope ? { scope: profile.evaluation_scope.labels } : {}),
      });
    }
  }
  return records;
}

// A paired task result is published only when the catalog's comparison receipt
// explicitly validates the shared evaluation scope. Keep the receipt metadata
// on each metric so the detail view can preserve that comparison context.
export function comparableAccuracyRecords(task, floatAccuracy, runtimeAccuracy, accuracyMetadata = {}, comparison) {
  const canonicalTask = canonicalAccuracyTask(task);
  if (!['cls', 'seg', 'pose', 'obb'].includes(canonicalTask)) {
    throw new Error(`${task} does not support a float/runtime end-to-end comparison`);
  }
  if (comparison?.status !== 'valid-evidence'
      || comparison?.direct_metric_comparison !== 'valid'
      || comparison?.comparison_scope?.comparable !== true
      || comparison?.comparison_scope?.kind !== 'end-to-end-metric-comparison'
      || comparison?.comparison_scope?.not_a_pure_quantization_loss_estimate !== true) {
    throw new Error(`${task} accuracy comparison must have valid comparable end-to-end evidence`);
  }
  if (typeof accuracyMetadata.dataset !== 'string' || !accuracyMetadata.dataset.trim()) {
    throw new Error(`${task} accuracy comparison requires a dataset`);
  }

  const records = accuracyRecords(task, floatAccuracy, runtimeAccuracy, accuracyMetadata);
  const profile = accuracyMetricsSchema.tasks[canonicalTask];
  const closeEnough = (actual, expected) => Number.isFinite(actual)
    && Math.abs(actual - expected) <= 1e-9;
  for (const { field } of profile.display) {
    const evidence = comparison.metrics?.[field];
    const expectedDelta = runtimeAccuracy[field] - floatAccuracy[field];
    if (!evidence
        || !closeEnough(evidence.float, floatAccuracy[field])
        || !closeEnough(evidence.board, runtimeAccuracy[field])
        || !closeEnough(evidence.board_minus_float, expectedDelta)) {
      throw new Error(`${task} accuracy comparison metrics do not match float/runtime receipts for ${field}`);
    }
  }

  return records.map(record => ({
    ...record,
    dataset: accuracyMetadata.dataset,
    ...(accuracyMetadata.evaluation_scope
      ? { evaluation_scope: accuracyMetadata.evaluation_scope }
      : {}),
    comparison_scope: { ...comparison.comparison_scope },
  }));
}

// A board receipt records standalone runtime accuracy. Keep it separate from
// the float/quantized pair so the detail view cannot infer an accuracy delta.
export function boardAccuracyRecords(task, boardAccuracy, accuracyMetadata = {}) {
  const canonicalTask = canonicalAccuracyTask(task);
  const profile = accuracyMetricsSchema.tasks[canonicalTask];
  if (!profile) throw new Error(`Unsupported accuracy task: ${task}`);
  if (profile.dataset && accuracyMetadata.dataset !== profile.dataset) {
    throw new Error(`${task} accuracy dataset must be ${profile.dataset}`);
  }
  if (profile.evaluation_scope && accuracyMetadata.evaluation_scope !== profile.evaluation_scope.value) {
    throw new Error(`${task} accuracy evaluation_scope must be ${profile.evaluation_scope.value}`);
  }

  const records = [];
  for (const { field, metric } of profile.display) {
    const value = boardAccuracy?.[field];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
      throw new Error(`${task} accuracy board.${field} must be a finite ratio between 0 and 1`);
    }
    records.push({
      metric,
      value,
      unit: 'ratio',
      model_stage: 'board',
      source_scope: accuracyMetadata.source_scope || {
        zh: '板端独立评测',
        en: 'Standalone board evaluation',
      },
      ...(profile.evaluation_scope ? { scope: profile.evaluation_scope.labels } : {}),
    });
  }
  return records;
}
