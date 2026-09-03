import type { BenchmarkRecord, MetricRecord, ModelRecord } from "./types";

export interface CatalogQuery {
  text: string;
  tasks: string[];
  formats: string[];
  precisions: string[];
  benchmark: "all" | "performance" | "accuracy" | "none";
  sort: "name" | "latency" | "fps" | "accuracy";
}

export interface QueryResult {
  models: ModelRecord[];
  sortApplied: boolean;
  reason?: "incomparable-benchmarks" | "missing-benchmarks";
}

const ACCURACY_UNITS = new Set<MetricRecord["unit"]>([
  "percent",
  "ratio",
  "mae",
  "rmse"
]);

const EMPTY_COMPARISON_VALUE = "<unspecified>";

function normalize(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase();
}

function compareNormalized(left: string, right: string): number {
  const normalizedLeft = normalize(left);
  const normalizedRight = normalize(right);
  if (normalizedLeft < normalizedRight) {
    return -1;
  }
  if (normalizedLeft > normalizedRight) {
    return 1;
  }
  return 0;
}

function compareModelsByName(left: ModelRecord, right: ModelRecord): number {
  return compareNormalized(left.name, right.name) || compareNormalized(left.id, right.id);
}

function hasText(value: string | undefined): value is string {
  return value !== undefined && value.trim().length > 0;
}

function hasAnySelectedValue(values: string[], selected: string[]): boolean {
  if (selected.length === 0) {
    return true;
  }
  const normalizedValues = new Set(values.map(normalize));
  return selected.some((value) => normalizedValues.has(normalize(value)));
}

function benchmarkValues(model: ModelRecord): BenchmarkRecord[] {
  return model.benchmarks;
}

function modelSearchValues(model: ModelRecord): string[] {
  return [
    model.name,
    model.id,
    model.sample_path,
    ...model.tasks,
    ...model.assets.map((asset) => asset.filename),
    ...benchmarkValues(model).flatMap((benchmark) => [
      benchmark.id,
      benchmark.variant_id,
      benchmark.display_name,
      ...(benchmark.asset_filename === undefined ? [] : [benchmark.asset_filename])
    ])
  ];
}

function hasPerformance(model: ModelRecord): boolean {
  return benchmarkValues(model).some((benchmark) => (benchmark.performance?.length ?? 0) > 0);
}

function hasAccuracy(model: ModelRecord): boolean {
  return benchmarkValues(model).some((benchmark) => (benchmark.accuracy?.length ?? 0) > 0);
}

function hasPublishedBenchmark(model: ModelRecord): boolean {
  return hasPerformance(model) || hasAccuracy(model);
}

function matchesBenchmarkFilter(model: ModelRecord, filter: CatalogQuery["benchmark"]): boolean {
  switch (filter) {
    case "performance":
      return hasPerformance(model);
    case "accuracy":
      return hasAccuracy(model);
    case "none":
      return !hasPublishedBenchmark(model);
    case "all":
      return true;
  }
}

function matchesSearch(model: ModelRecord, text: string): boolean {
  const normalizedText = normalize(text.trim());
  if (normalizedText.length === 0) {
    return true;
  }
  return modelSearchValues(model).some((value) => normalize(value).includes(normalizedText));
}

function modelFormats(model: ModelRecord): string[] {
  return [
    ...model.assets.map((asset) => asset.format),
    ...benchmarkValues(model)
      .map((benchmark) => benchmark.model_format)
      .filter((format): format is string => format !== undefined)
  ];
}

function modelPrecisions(model: ModelRecord): string[] {
  return benchmarkValues(model)
    .map((benchmark) => benchmark.precision)
    .filter((precision): precision is string => precision !== undefined);
}

function matchesFilters(model: ModelRecord, query: CatalogQuery): boolean {
  return matchesSearch(model, query.text)
    && hasAnySelectedValue(model.tasks, query.tasks)
    && hasAnySelectedValue(modelFormats(model), query.formats)
    && hasAnySelectedValue(modelPrecisions(model), query.precisions)
    && matchesBenchmarkFilter(model, query.benchmark);
}

function isAccuracyMetric(record: BenchmarkRecord, metric: MetricRecord): boolean {
  if (ACCURACY_UNITS.has(metric.unit)) {
    return true;
  }
  return record.accuracy?.some((candidate) => candidate === metric) ?? false;
}

function canonicalShape(shape: number[] | undefined): string | null {
  if (shape === undefined || shape.length === 0 || shape.some((value) => !Number.isFinite(value))) {
    return null;
  }
  return JSON.stringify(shape);
}

function canonicalOptional(value: string | undefined): string {
  return hasText(value) ? normalize(value) : EMPTY_COMPARISON_VALUE;
}

function canonicalPositiveInteger(value: number | undefined): string | null {
  return value !== undefined && Number.isInteger(value) && value > 0 ? String(value) : null;
}

/**
 * Builds the full set of conditions that must match before numeric values can
 * be compared. Performance records require the timing and input conditions;
 * accuracy records require the dataset and model stage instead.
 */
export function comparisonKey(record: BenchmarkRecord, metric: MetricRecord): string | null {
  if (!hasText(record.environment.hardware) || !hasText(metric.metric) || !hasText(metric.unit)) {
    return null;
  }

  const shape = canonicalShape(record.input?.shape);
  const concurrency = canonicalPositiveInteger(metric.concurrency);
  const bpuCores = canonicalPositiveInteger(record.environment.bpu_cores);
  const accuracy = isAccuracyMetric(record, metric);

  if (accuracy) {
    if (!hasText(metric.dataset) || !hasText(metric.model_stage)) {
      return null;
    }
    return JSON.stringify([
      normalize(record.environment.hardware),
      normalize(metric.metric),
      normalize(metric.unit),
      canonicalOptional(metric.statistic),
      canonicalOptional(metric.scope),
      concurrency ?? EMPTY_COMPARISON_VALUE,
      shape ?? EMPTY_COMPARISON_VALUE,
      canonicalOptional(record.input?.layout),
      canonicalOptional(record.input?.format),
      bpuCores ?? EMPTY_COMPARISON_VALUE,
      normalize(metric.dataset),
      normalize(metric.model_stage)
    ]);
  }

  if (
    !hasText(metric.scope)
    || concurrency === null
    || shape === null
    || !hasText(record.input?.layout)
    || !hasText(record.input?.format)
    || bpuCores === null
  ) {
    return null;
  }

  return JSON.stringify([
    normalize(record.environment.hardware),
    normalize(metric.metric),
    normalize(metric.unit),
    canonicalOptional(metric.statistic),
    normalize(metric.scope),
    concurrency,
    shape,
    normalize(record.input.layout),
    normalize(record.input.format),
    bpuCores
  ]);
}

function metricsForSort(model: ModelRecord, sort: Exclude<CatalogQuery["sort"], "name">): MetricRecord[] {
  const metrics = benchmarkValues(model).flatMap((benchmark) => {
    if (sort === "accuracy") {
      return benchmark.accuracy ?? [];
    }
    return benchmark.performance ?? [];
  });

  switch (sort) {
    case "latency":
      return metrics.filter((metric) => normalize(metric.metric) === "latency");
    case "fps":
      return metrics.filter((metric) =>
        normalize(metric.metric) === "throughput" || normalize(metric.metric) === "fps"
      );
    case "accuracy":
      return metrics;
  }
}

interface SortCandidate {
  model: ModelRecord;
  metric: MetricRecord;
  key: string;
}

function sortByNumericMetric(
  models: ModelRecord[],
  sort: Exclude<CatalogQuery["sort"], "name">
): QueryResult {
  const candidates: SortCandidate[] = [];
  let missingMetric = false;
  let ambiguousMetric = false;

  for (const model of models) {
    const metrics = metricsForSort(model, sort);
    if (metrics.length === 0) {
      missingMetric = true;
      continue;
    }
    if (metrics.length !== 1) {
      ambiguousMetric = true;
      continue;
    }
    const metric = metrics[0];
    if (metric === undefined) {
      missingMetric = true;
      continue;
    }
    const benchmark = benchmarkValues(model).find((candidate) =>
      (sort === "accuracy" ? candidate.accuracy : candidate.performance)?.some((value) => value === metric)
    );
    if (benchmark === undefined) {
      ambiguousMetric = true;
      continue;
    }
    const key = comparisonKey(benchmark, metric);
    if (key === null) {
      ambiguousMetric = true;
      continue;
    }
    candidates.push({ model, metric, key });
  }

  if (missingMetric) {
    return {
      models,
      sortApplied: false,
      reason: "missing-benchmarks"
    };
  }
  if (ambiguousMetric || candidates.length === 0) {
    return {
      models,
      sortApplied: false,
      reason: "incomparable-benchmarks"
    };
  }

  const comparisonKeyValue = candidates[0]?.key;
  if (comparisonKeyValue === undefined || candidates.some((candidate) => candidate.key !== comparisonKeyValue)) {
    return {
      models,
      sortApplied: false,
      reason: "incomparable-benchmarks"
    };
  }

  const descending = sort === "fps" || sort === "accuracy";
  const candidateById = new Map(candidates.map((candidate) => [candidate.model.id, candidate]));
  const sorted = [...models].sort((left, right) => {
    const leftCandidate = candidateById.get(left.id);
    const rightCandidate = candidateById.get(right.id);
    if (leftCandidate === undefined || rightCandidate === undefined) {
      return compareModelsByName(left, right);
    }
    const valueDifference = leftCandidate.metric.value - rightCandidate.metric.value;
    if (valueDifference !== 0) {
      return descending ? -valueDifference : valueDifference;
    }
    return compareModelsByName(left, right);
  });

  return { models: sorted, sortApplied: true };
}

export function queryModels(models: ModelRecord[], query: CatalogQuery): QueryResult {
  const filtered = models.filter((model) => matchesFilters(model, query)).sort(compareModelsByName);
  if (query.sort === "name") {
    return { models: filtered, sortApplied: true };
  }
  if (filtered.length === 0) {
    return { models: filtered, sortApplied: false };
  }
  return sortByNumericMetric(filtered, query.sort);
}
