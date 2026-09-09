import type { BenchmarkRecord, Locale, MetricRecord, MetricUnit } from "./types";
import { canonicalMetricName, isRetentionMetricName } from "./metric-identity";

/** A metric together with the benchmark record that supplied its conditions. */
export interface MetricEntry {
  record: BenchmarkRecord;
  metric: MetricRecord;
  kind: "performance" | "accuracy";
  artifact: string;
}

export interface PerformanceThread {
  /** Undefined means that the source did not record a thread count. */
  concurrency?: number;
  latency?: MetricEntry;
  throughput?: MetricEntry;
  metrics: MetricEntry[];
}

export interface PerformanceGroup {
  key: string;
  variantId: string;
  artifact: string;
  scope?: string;
  statistic?: MetricRecord["statistic"];
  threads: PerformanceThread[];
  metrics: MetricEntry[];
}

export interface AccuracyPair {
  key: string;
  metric: string;
  /** Canonical identity, so `TOP1` and `top-1` are one measurement. */
  canonicalMetric: string;
  dataset?: string;
  unit: MetricUnit;
  scope?: string;
  statistic?: MetricRecord["statistic"];
  artifact: string;
  float?: MetricEntry;
  quantized?: MetricEntry;
  /**
   * A published value whose stage the source does not label (or labels as
   * compiled/runtime). It must stay visible: an unlabelled stage is not the
   * same statement as “not yet measured”.
   */
  other?: MetricEntry;
  retention?: MetricEntry;
  retentionValue?: number;
  retentionSource?: "explicit" | "derived";
  retentionStatus: "value" | "not-measured" | "not-comparable" | "not-applicable";
}

export interface RetentionDisplay {
  status: AccuracyPair["retentionStatus"];
  value?: number;
  source?: "explicit" | "derived";
  metric?: MetricEntry;
}

function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
}

function optionalKey(value: string | undefined): string {
  return normalized(value) || "<unspecified>";
}

function statisticKey(value: MetricRecord["statistic"]): string {
  return value ?? "<unspecified>";
}

function validConcurrency(value: number | undefined): number | undefined {
  return value !== undefined && Number.isInteger(value) && value > 0 ? value : undefined;
}

/**
 * A benchmark asset is the strongest available artifact identity. A variant
 * id is used for legacy records that predate explicit asset associations.
 */
export function artifactForRecord(record: BenchmarkRecord): string {
  return record.asset_filename?.trim() || record.variant_id;
}

function inputKey(record: BenchmarkRecord): string {
  return JSON.stringify([
    record.input?.shape ?? null,
    optionalKey(record.input?.layout),
    optionalKey(record.input?.format)
  ]);
}

function isLatencyMetric(metric: MetricRecord): boolean {
  return normalized(metric.metric) === "latency";
}

function isThroughputMetric(metric: MetricRecord): boolean {
  const name = normalized(metric.metric);
  return name === "throughput" || name === "fps";
}

function isPrimaryPerformanceMetric(metric: MetricRecord): boolean {
  return isLatencyMetric(metric) || isThroughputMetric(metric);
}

function metricEntry(record: BenchmarkRecord, metric: MetricRecord, kind: MetricEntry["kind"]): MetricEntry {
  return { record, metric, kind, artifact: artifactForRecord(record) };
}

/**
 * Timing-scope equivalence class for grouping.
 *
 * The three release lines label the plain BPU timing measurement differently
 * (`BPU task`, `BPU; 2 threads`, `multi-thread`, `frame rate`, `100 frames`,
 * no scope at all…). Those wordings all describe the same row: the thread
 * configuration is already carried by `concurrency`. Groupring by the raw
 * wording split one measurement into several rows — a latency row, a separate
 * FPS row and one row per thread count — so the wording is folded into one
 * class here. Scopes that describe a genuinely different measurement context
 * (encoder/decoder, pooler vs last hidden state, compiler estimates, post
 * processing) keep their own class and stay separate rows.
 */
const BPU_SCOPE_CLASS_PATTERNS: RegExp[] = [
  /^bpu( task)?([;,.]|$)/,
  /^bpu (single|multi)[- ]?thread/,
  /^bpu throughput( summary)?$/,
  /^(single|multi|two|three|four|eight|twelve)[- ]?thread/,
  /^single-frame, single-thread/,
  /^frame rate$/,
  /^[0-9]+\s*frames?(, core_id [0-9,]+)?$/,
  /^hrt_model_exec perf/,
  /^model execution only$/,
  /^measured latency table$/,
  /^[0-9]+×[0-9]+, single core$/
];

function performanceScopeClass(scope: string | undefined): string {
  if (!scope) return "bpu";
  const value = scope.trim().toLowerCase();
  if (BPU_SCOPE_CLASS_PATTERNS.some((pattern) => pattern.test(value))) {
    return "bpu";
  }
  return `scope:${value}`;
}

/**
 * An unstated statistic is the natural default of the stated one: a latency
 * published without a statistic and the mean latency beside it belong to the
 * same row. Explicit statistics (min/p50/…) still produce their own rows.
 */
function statisticClass(statistic: MetricRecord["statistic"]): string {
  return statistic ?? "mean";
}

/**
 * Groups performance measurements only when their timing scope and statistic
 * are compatible. Unknown concurrency is intentionally kept as its own
 * thread bucket instead of silently becoming one thread. When folding scopes
 * into one class would put two latencies (or two throughputs) of different
 * raw scopes into the same thread bucket, the second entry keeps its own
 * exact-scope group instead of being hidden.
 */
export function groupPerformanceMetrics(records: BenchmarkRecord[]): PerformanceGroup[] {
  const groups = new Map<string, PerformanceGroup>();
  for (const record of records) {
    for (const metric of record.performance ?? []) {
      const artifact = artifactForRecord(record);
      const scope = metric.scope;
      const statistic = metric.statistic;
      const makeKey = (scopeKey: string): string => JSON.stringify([
        record.variant_id,
        artifact,
        inputKey(record),
        scopeKey,
        statisticClass(statistic)
      ]);
      const entry = metricEntry(record, metric, "performance");
      const concurrency = validConcurrency(metric.concurrency);

      const placeInto = (group: PerformanceGroup): void => {
        group.metrics.push(entry);
        let thread = group.threads.find((candidate) => candidate.concurrency === concurrency);
        if (thread === undefined) {
          thread = { concurrency, metrics: [] };
          group.threads.push(thread);
          group.threads.sort((left, right) => {
            if (left.concurrency === undefined) return 1;
            if (right.concurrency === undefined) return -1;
            return left.concurrency - right.concurrency;
          });
        }
        thread.metrics.push(entry);
        const occupied = isLatencyMetric(metric) ? thread.latency : undefined;
        const occupiedThroughput = isThroughputMetric(metric) ? thread.throughput : undefined;
        const incumbent = occupied ?? occupiedThroughput;
        if (incumbent !== undefined && incumbent.metric.scope !== scope) {
          // Collision from a different raw scope: this entry would be hidden.
          // Undo the partial placement; the caller falls back to exact scope.
          group.metrics.pop();
          thread.metrics.pop();
          if (thread.metrics.length === 0) {
            group.threads.splice(group.threads.indexOf(thread), 1);
          }
          return;
        }
        if (isLatencyMetric(metric)) thread.latency ??= entry;
        if (isThroughputMetric(metric)) thread.throughput ??= entry;
      };

      const classKey = makeKey(performanceScopeClass(scope));
      let group = groups.get(classKey);
      if (group === undefined) {
        group = {
          key: classKey,
          variantId: record.variant_id,
          artifact,
          scope,
          statistic,
          threads: [],
          metrics: []
        };
        groups.set(classKey, group);
      }
      const before = group.metrics.length;
      placeInto(group);
      if (group.metrics.length === before) {
        // collision: keep the entry visible in its own exact-scope group
        const exactKey = makeKey(optionalKey(scope));
        let exact = groups.get(exactKey);
        if (exact === undefined) {
          exact = {
            key: exactKey,
            variantId: record.variant_id,
            artifact,
            scope,
            statistic,
            threads: [],
            metrics: []
          };
          groups.set(exactKey, exact);
        }
        const beforeExact = exact.metrics.length;
        placeInto(exact);
        if (exact.metrics.length === beforeExact) {
          // pathological duplicate: keep it as its own single-entry group
          const singleton: PerformanceGroup = {
            key: `${exactKey}#${groups.size}`,
            variantId: record.variant_id,
            artifact,
            scope,
            statistic,
            threads: [],
            metrics: []
          };
          placeInto(singleton);
          groups.set(singleton.key, singleton);
        }
      }
    }
  }
  return [...groups.values()].filter((group) => group.metrics.length > 0);
}

/** Return only groups that have a latency or throughput measurement. */
export function primaryPerformanceGroups(records: BenchmarkRecord[]): PerformanceGroup[] {
  return groupPerformanceMetrics(records).filter((group) =>
    group.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
  );
}

function isRetentionMetric(metric: MetricRecord): boolean {
  return isRetentionMetricName(metric.metric);
}

/** Remove the conventional suffix from an explicit retention metric name. */
export function retentionBaseMetric(metricName: string): string | undefined {
  const value = metricName.normalize("NFKC").trim();
  const match = value.match(/^(.*?)(?:[-_ ]retention|[-_ ]retained|retention)$/i);
  if (match?.[1]?.trim()) return match[1].trim();
  return normalized(value) === "retention" ? undefined : value;
}

function isErrorMetric(metric: MetricRecord): boolean {
  const name = normalized(metric.metric);
  return metric.unit === "mae"
    || metric.unit === "rmse"
    || name === "mae"
    || name === "rmse"
    || name === "wer"
    || name === "cer"
    || name === "error"
    || name === "loss"
    || name.includes("mean absolute error")
    || name.includes("root mean square error")
    || name.includes("word error rate")
    || name.includes("character error rate")
    || name.includes("error-rate")
    || name.includes("error rate")
    || name.includes("loss");
}

/** Whether an accuracy unit can be safely compared as a higher-is-better value. */
export function isHigherBetterMetric(metric: MetricRecord): boolean {
  return !isErrorMetric(metric) && (metric.unit === "ratio" || metric.unit === "percent");
}

/**
 * Derive quantized / float retention as a percentage. Undefined is returned
 * for mismatched units, lower-is-better metrics, non-finite values, and a zero
 * float denominator.
 */
export function deriveRetention(
  floatMetric: MetricRecord | undefined,
  quantizedMetric: MetricRecord | undefined
): number | undefined {
  if (floatMetric === undefined || quantizedMetric === undefined) return undefined;
  if (floatMetric.unit !== quantizedMetric.unit || !isHigherBetterMetric(floatMetric)) return undefined;
  if (!Number.isFinite(floatMetric.value) || !Number.isFinite(quantizedMetric.value) || floatMetric.value === 0) {
    return undefined;
  }
  return quantizedMetric.value / floatMetric.value * 100;
}

/** Compatibility alias for callers that prefer an explicit verb. */
export const calculateRetention = deriveRetention;
export const computeRetention = deriveRetention;
export const calculateAccuracyRetention = deriveRetention;

function accuracyKey(
  record: BenchmarkRecord,
  metric: MetricRecord,
  metricName = metric.metric
): string {
  return JSON.stringify([
    canonicalMetricName(metricName),
    optionalKey(metric.dataset),
    metric.unit,
    optionalKey(metric.scope),
    statisticKey(metric.statistic),
    artifactForRecord(record),
    inputKey(record)
  ]);
}

function explicitRetentionKey(record: BenchmarkRecord, metric: MetricRecord): string {
  return JSON.stringify([
    canonicalMetricName(retentionBaseMetric(metric.metric) ?? metric.metric),
    optionalKey(metric.dataset),
    optionalKey(metric.scope),
    statisticKey(metric.statistic),
    artifactForRecord(record),
    inputKey(record)
  ]);
}

function hasOppositeStage(pair: AccuracyPair, entries: MetricEntry[]): boolean {
  const stage = pair.float ? "quantized" : "float";
  const variantId = pair.float?.record.variant_id ?? pair.quantized?.record.variant_id;
  // A counterpart with the same model variant but a different metric, scope,
  // dataset, statistic or artifact is evidence that this row is present but
  // cannot be compared.  Restricting this check to the metric name would make
  // a float mAP50 and quantized mAP50-95 pair look merely unmeasured.
  return entries.some((entry) => entry.metric.model_stage === stage
    && entry.record.variant_id === variantId);
}

function retentionForPair(pair: AccuracyPair): RetentionDisplay {
  if (pair.retention !== undefined && Number.isFinite(pair.retention.metric.value)) {
    const raw = pair.retention.metric;
    const value = raw.unit === "ratio" ? raw.value * 100 : raw.value;
    return { status: "value", value, source: "explicit", metric: pair.retention };
  }
  const value = deriveRetention(pair.float?.metric, pair.quantized?.metric);
  if (value !== undefined) return { status: "value", value, source: "derived" };
  if (pair.float !== undefined && pair.quantized !== undefined) {
    return { status: isHigherBetterMetric(pair.float.metric) ? "not-comparable" : "not-applicable" };
  }
  return { status: "not-measured" };
}

/** Compute the best available retention for one paired accuracy row. */
export function getRetention(pair: AccuracyPair): RetentionDisplay {
  if (pair.retentionValue !== undefined) {
    return {
      status: "value",
      value: pair.retentionValue,
      source: pair.retentionSource,
      metric: pair.retention
    };
  }
  if (pair.retentionStatus !== "not-measured") {
    return { status: pair.retentionStatus };
  }
  return retentionForPair(pair);
}

/**
 * Pairs float and quantized values only when metric, dataset, unit, scope,
 * statistic, input and artifact all agree. Explicit retention records are
 * attached to that exact pair and always take precedence over derivation.
 */
export function pairAccuracyMetrics(records: BenchmarkRecord[]): AccuracyPair[] {
  const entries: MetricEntry[] = [];
  const explicit = new Map<string, MetricEntry[]>();
  for (const record of records) {
    for (const metric of record.accuracy ?? []) {
      const entry = metricEntry(record, metric, "accuracy");
      if (isRetentionMetric(metric)) {
        const key = explicitRetentionKey(record, metric);
        const values = explicit.get(key) ?? [];
        values.push(entry);
        explicit.set(key, values);
      } else {
        entries.push(entry);
      }
    }
  }

  const pairs = new Map<string, AccuracyPair>();
  for (const entry of entries) {
    const metric = entry.metric;
    const key = accuracyKey(entry.record, metric);
    const existing = pairs.get(key) ?? {
      key,
      metric: metric.metric,
      canonicalMetric: canonicalMetricName(metric.metric),
      dataset: metric.dataset,
      unit: metric.unit,
      scope: metric.scope,
      statistic: metric.statistic,
      artifact: entry.artifact,
      retentionStatus: "not-measured"
    };
    if (metric.model_stage === "float") existing.float ??= entry;
    else if (metric.model_stage === "quantized") existing.quantized ??= entry;
    else existing.other ??= entry;
    pairs.set(key, existing);
  }

  for (const pair of pairs.values()) {
    const pairRecord = pair.float?.record
      ?? pair.quantized?.record
      ?? entries.find((entry) => accuracyKey(entry.record, entry.metric) === pair.key)?.record;
    if (pairRecord === undefined) {
      pair.retentionStatus = "not-measured";
      continue;
    }
    const retentionKey = JSON.stringify([
      pair.canonicalMetric,
      optionalKey(pair.dataset),
      optionalKey(pair.scope),
      statisticKey(pair.statistic),
      pair.artifact,
      inputKey(pairRecord)
    ]);
    const raw = explicit.get(retentionKey)?.[0];
    if (raw !== undefined) pair.retention = raw;
    const display = retentionForPair(pair);
    pair.retentionValue = display.value;
    pair.retentionSource = display.source;
    pair.retentionStatus = display.status;
    if (display.status === "not-measured" && (pair.float !== undefined || pair.quantized !== undefined)) {
      if (hasOppositeStage(pair, entries)) pair.retentionStatus = "not-comparable";
    }
    // A value published without a float/quantized split has no counterpart to
    // compare against. Say so instead of implying it was never measured.
    if (display.status === "not-measured" && pair.float === undefined && pair.quantized === undefined
      && pair.other !== undefined && pair.retention === undefined) {
      pair.retentionStatus = "not-applicable";
    }
  }

  // Retention can be the only accuracy record. Preserve it as a visible row
  // rather than discarding it when no float/quantized pair exists.
  for (const values of explicit.values()) {
    for (const entry of values) {
      const base = retentionBaseMetric(entry.metric.metric);
      if (base === undefined) {
        const candidates = [...pairs.values()].filter((pair) =>
          pair.retention === undefined
          && pair.artifact === entry.artifact
          && optionalKey(pair.dataset) === optionalKey(entry.metric.dataset)
          && optionalKey(pair.scope) === optionalKey(entry.metric.scope)
          && statisticKey(pair.statistic) === statisticKey(entry.metric.statistic)
          && (() => {
            const pairRecord = pair.float?.record ?? pair.quantized?.record ?? pair.other?.record;
            return pairRecord !== undefined && inputKey(pairRecord) === inputKey(entry.record);
          })()
        );
        if (candidates.length === 1) {
          const pair = candidates[0]!;
          pair.retention = entry;
          const display = retentionForPair(pair);
          pair.retentionValue = display.value;
          pair.retentionSource = display.source;
          pair.retentionStatus = display.status;
          continue;
        }
        const key = `explicit:${entry.record.id}:${entry.metric.metric}`;
        pairs.set(key, {
          key,
          metric: entry.metric.metric,
          canonicalMetric: canonicalMetricName(retentionBaseMetric(entry.metric.metric) ?? entry.metric.metric),
          dataset: entry.metric.dataset,
          unit: entry.metric.unit,
          scope: entry.metric.scope,
          statistic: entry.metric.statistic,
          artifact: entry.artifact,
          retention: entry,
          retentionValue: entry.metric.unit === "ratio" ? entry.metric.value * 100 : entry.metric.value,
          retentionSource: "explicit",
          retentionStatus: "value"
        });
        continue;
      }
      const hasPair = [...pairs.values()].some((pair) => pair.canonicalMetric === canonicalMetricName(base)
        && pair.artifact === entry.artifact
        && optionalKey(pair.dataset) === optionalKey(entry.metric.dataset)
        && optionalKey(pair.scope) === optionalKey(entry.metric.scope)
        && statisticKey(pair.statistic) === statisticKey(entry.metric.statistic)
        && (() => {
          const pairRecord = pair.float?.record ?? pair.quantized?.record ?? pair.other?.record;
          return pairRecord !== undefined && inputKey(pairRecord) === inputKey(entry.record);
        })());
      if (!hasPair) {
        const key = `explicit:${entry.record.id}:${entry.metric.metric}`;
        const value = entry.metric.unit === "ratio" ? entry.metric.value * 100 : entry.metric.value;
        pairs.set(key, {
          key,
          metric: base,
          canonicalMetric: canonicalMetricName(base),
          dataset: entry.metric.dataset,
          unit: entry.metric.unit,
          scope: entry.metric.scope,
          statistic: entry.metric.statistic,
          artifact: entry.artifact,
          retention: entry,
          retentionValue: value,
          retentionSource: "explicit",
          retentionStatus: "value"
        });
      }
    }
  }

  return [...pairs.values()];
}

/** Compatibility alias for callers that use the shorter noun. */
export const pairAccuracy = pairAccuracyMetrics;
export const groupPerformance = groupPerformanceMetrics;

function formatNumber(value: number, locale: Locale, maximumFractionDigits = 6): string {
  return new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits }).format(value);
}

export interface FormatMetricOptions {
  /** Convert ratio values to percentage display for accuracy metrics. */
  asPercentage?: boolean;
  maximumFractionDigits?: number;
}

/** Format a metric value with an unambiguous unit. */
export function formatMetricValue(
  metric: Pick<MetricRecord, "value" | "unit" | "metric">,
  locale: Locale = "en",
  options: FormatMetricOptions = {}
): string {
  const asPercentage = options.asPercentage ?? metric.unit === "ratio";
  const value = metric.unit === "ratio" && asPercentage ? metric.value * 100 : metric.value;
  const fractionDigits = options.maximumFractionDigits ?? (metric.unit === "ratio" || metric.unit === "percent" ? 2 : 6);
  const number = formatNumber(value, locale, fractionDigits);
  const unit = metric.unit === "percent" || (metric.unit === "ratio" && asPercentage)
    ? "%"
    : metric.unit;
  return `${number}${unit}`;
}

/** Format values that are shown in a performance thread cell. */
export function formatPerformanceMetric(metric: MetricRecord, locale: Locale = "en"): string {
  return formatMetricValue(metric, locale, { asPercentage: false });
}

/** Format an already-computed retention percentage without changing its scale. */
export function formatRetention(value: number, locale: Locale = "en"): string {
  return `${formatNumber(value, locale, 2)}%`;
}

/** Compatibility alias for callers that use a shorter formatter name. */
export const formatMetric = formatMetricValue;
