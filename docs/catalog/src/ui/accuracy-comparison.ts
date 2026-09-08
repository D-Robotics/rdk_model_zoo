import type { Locale, MetricRecord, MetricUnit } from "../catalog/types";
import {
  formatRetention,
  getRetention,
  type AccuracyPair,
  type MetricEntry
} from "../catalog/metric-display";
import { metricDisplayLabel, unitScaleLabel } from "../catalog/metric-identity";
import { detailLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";

/**
 * Accuracy columns are derived from the measurements a model actually
 * publishes, so every model gets its own table shape instead of a shared
 * three-column template. A classification model shows Top-1/Top-5, a detector
 * shows the bbox mAP columns of its source table, a pose model shows the
 * keypoint columns, and an embedding model shows cosine similarity.
 */

export type AccuracyStage = "float" | "quantized" | "other";

export interface AccuracyColumnGroup {
  key: string;
  canonicalMetric: string;
  label: string;
  units: MetricUnit[];
  datasets: string[];
  stages: AccuracyStage[];
  showRetention: boolean;
  pairs: AccuracyPair[];
}

const STAGE_ORDER: AccuracyStage[] = ["float", "quantized", "other"];

function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
}

function stageEntry(pair: AccuracyPair, stage: AccuracyStage): MetricEntry | undefined {
  if (stage === "float") return pair.float;
  if (stage === "quantized") return pair.quantized;
  return pair.other;
}

function unique(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value && value.trim())))];
}

/**
 * Groups paired accuracy measurements by canonical metric, preserving the
 * order in which the source tables publish them.
 */
export function accuracyColumnGroups(pairs: AccuracyPair[], locale: Locale): AccuracyColumnGroup[] {
  const groups = new Map<string, AccuracyColumnGroup>();
  for (const pair of pairs) {
    const existing = groups.get(pair.canonicalMetric);
    if (existing) {
      existing.pairs.push(pair);
      continue;
    }
    groups.set(pair.canonicalMetric, {
      key: pair.canonicalMetric,
      canonicalMetric: pair.canonicalMetric,
      label: metricDisplayLabel(pair.canonicalMetric, locale),
      units: [],
      datasets: [],
      stages: [],
      showRetention: false,
      pairs: [pair]
    });
  }

  for (const group of groups.values()) {
    const stages = new Set<AccuracyStage>();
    const units = new Set<MetricUnit>();
    let retention = false;
    for (const pair of group.pairs) {
      if (pair.float) stages.add("float");
      if (pair.quantized) stages.add("quantized");
      if (pair.other) stages.add("other");
      units.add(pair.unit);
      const display = getRetention(pair);
      if (display.status === "value") retention = true;
    }
    group.stages = STAGE_ORDER.filter((stage) => stages.has(stage));
    group.units = [...units];
    group.datasets = unique(group.pairs.map((pair) => pair.dataset));
    // A retention column only carries information when something is comparable.
    group.showRetention = retention || group.stages.length > 1;
  }
  return [...groups.values()];
}

/** Whether an accuracy measurement belongs to a performance row's timing scope. */
export function scopeCompatible(rowScope: string | undefined, pairScope: string | undefined): boolean {
  const row = normalized(rowScope);
  const pair = normalized(pairScope);
  if (!row || !pair) return true;
  return pair.includes(row) || row.includes(pair);
}

/** The measurements of one column group that apply to a single table row. */
export function pairsForRow(group: AccuracyColumnGroup, rowPairs: AccuracyPair[], rowScope?: string): AccuracyPair[] {
  return group.pairs.filter((candidate) => {
    const rowPair = rowPairs.find((pair) => pair.key === candidate.key);
    return rowPair !== undefined && scopeCompatible(rowScope, rowPair.scope);
  });
}

function digitsFor(unit: MetricUnit): number {
  return unit === "ratio" || unit === "percent" ? 4 : 6;
}

/**
 * Accuracy values are shown exactly as the source records them, with no `%`
 * suffix: the scale lives in the column header. Retention is the one value
 * that keeps its `%` because it is defined as a percentage.
 */
export function accuracyValueText(metric: MetricRecord, locale: Locale): string {
  return new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en-US", {
    maximumFractionDigits: digitsFor(metric.unit)
  }).format(metric.value);
}

export function stageLabel(stage: AccuracyStage, locale: Locale): string {
  if (stage === "float") return detailLabel(locale, "floatAccuracy");
  if (stage === "quantized") return detailLabel(locale, "quantizedAccuracy");
  return locale === "zh" ? "实测值" : "Reported value";
}

/** Header condition line: dataset and value scale, as published by the source. */
export function columnConditionText(group: AccuracyColumnGroup, locale: Locale): string {
  const parts = [
    group.datasets.length > 0 ? group.datasets.join(" / ") : undefined,
    group.units.map((unit) => unitScaleLabel(unit, locale)).join(" / ")
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

function appendValueLine(
  element: HTMLElement,
  text: string,
  className: string,
  title?: string,
  dataset?: Record<string, string>
): void {
  if (element.childNodes.length > 0) element.append(document.createElement("br"));
  const item = document.createElement("span");
  item.className = className;
  item.textContent = text;
  if (title) item.title = title;
  for (const [key, value] of Object.entries(dataset ?? {})) item.dataset[key] = value;
  element.append(item);
}

/**
 * Renders one stage column for one row. A measurement that belongs to a
 * different timing scope is marked not applicable rather than being dropped or
 * reported as missing, so the reader can tell “other row” from “never measured”.
 */
export function renderStageCell(
  element: HTMLElement,
  group: AccuracyColumnGroup,
  rowPairs: AccuracyPair[],
  stage: AccuracyStage,
  context: DetailContext,
  rowScope?: string
): void {
  const applicable = pairsForRow(group, rowPairs, rowScope);
  const values = applicable
    .map((pair) => ({ pair, entry: stageEntry(pair, stage) }))
    .filter((value): value is { pair: AccuracyPair; entry: MetricEntry } => value.entry !== undefined);
  if (values.length === 0) {
    const anyStageForScope = applicable.some((pair) => STAGE_ORDER.some((candidate) => stageEntry(pair, candidate)));
    const scopedOut = group.pairs.some((pair) => rowPairs.some((candidate) => candidate.key === pair.key)
      && !scopeCompatible(rowScope, pair.scope));
    element.textContent = detailLabel(context.locale, scopedOut && !anyStageForScope
      ? "notApplicable"
      : "noAccuracy");
    element.dataset.empty = "true";
    return;
  }
  const ambiguousUnits = group.units.length > 1;
  for (const { pair, entry } of values) {
    const suffix = ambiguousUnits ? ` ${unitScaleLabel(entry.metric.unit, context.locale)}` : "";
    appendValueLine(
      element,
      `${accuracyValueText(entry.metric, context.locale)}${suffix}`,
      "model-detail-accuracy-value",
      [pair.dataset, pair.scope, pair.artifact].filter(Boolean).join(" · "),
      { metric: pair.canonicalMetric, stage }
    );
  }
}

/** Renders the retention column for one row. */
export function renderRetentionCell(
  element: HTMLElement,
  group: AccuracyColumnGroup,
  rowPairs: AccuracyPair[],
  context: DetailContext,
  rowScope?: string
): void {
  const applicable = pairsForRow(group, rowPairs, rowScope);
  if (applicable.length === 0) {
    element.textContent = detailLabel(context.locale, "notApplicable");
    element.dataset.empty = "true";
    return;
  }
  const rendered = new Set<string>();
  let wrote = false;
  for (const pair of applicable) {
    const retention = getRetention(pair);
    const text = retention.status === "value" && retention.value !== undefined
      ? formatRetention(retention.value, context.locale)
      : detailLabel(context.locale, retention.status === "not-comparable"
        ? "notComparable"
        : retention.status === "not-applicable" ? "notApplicable" : "notMeasured");
    if (rendered.has(text)) continue;
    rendered.add(text);
    appendValueLine(element, text, "model-detail-retention-value", retention.source === "derived"
      ? (context.locale === "zh" ? "由原始精度计算" : "Derived from source accuracies")
      : undefined, { retentionSource: retention.source ?? "none" });
    wrote = true;
  }
  if (!wrote) {
    element.textContent = detailLabel(context.locale, "notMeasured");
    element.dataset.empty = "true";
  }
}
