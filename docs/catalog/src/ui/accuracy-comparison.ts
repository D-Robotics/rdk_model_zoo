import type { Locale } from "../catalog/types";
import {
  formatRetention,
  getRetention,
  type AccuracyPair,
  type MetricEntry
} from "../catalog/metric-display";
import { detailLabel, metricLabel, unitLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";
import { accuracyCellText } from "./detail-utils";

function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
}

export function readableAccuracyMetric(metric: string, locale: Locale): string {
  const normalizedMetric = normalized(metric);
  const map = normalizedMetric.match(/^(bbox|mask|keypoints?)-all-map-50-95$/);
  if (map?.[1]) return `${map[1]} mAP@0.5:0.95`;
  if (normalizedMetric === "top-1") return "Top-1";
  if (normalizedMetric === "top-5") return "Top-5";
  return metricLabel(locale, metric);
}

export function accuracyMetricName(pair: AccuracyPair, locale: Locale): string {
  return readableAccuracyMetric(pair.metric, locale);
}

export function accuracyDescriptor(pair: AccuracyPair, locale: Locale): string {
  const dataset = pair.dataset ?? detailLabel(locale, "notRecorded");
  const metric = readableAccuracyMetric(pair.metric, locale);
  const unit = pair.unit === "ratio" || pair.unit === "percent" ? "%" : unitLabel(locale, pair.unit);
  return `${dataset} · ${metric} (${unit})`;
}

export function accuracyHeader(pairs: AccuracyPair[], locale: Locale): { text: string; shared: boolean } {
  const descriptors = [...new Set(pairs.map((pair) => accuracyDescriptor(pair, locale)))];
  if (descriptors.length === 0) return { text: detailLabel(locale, "accuracy"), shared: true };
  return {
    text: `${detailLabel(locale, "accuracy")} · ${descriptors.join(" / ")}`,
    shared: descriptors.length === 1
  };
}

export function renderAccuracyValues(
  element: HTMLElement,
  pairs: AccuracyPair[],
  stage: "float" | "quantized",
  context: DetailContext,
  showMetricLabels: boolean
): void {
  const values = pairs
    .map((pair) => ({ pair, entry: pair[stage] }))
    .filter((value): value is { pair: AccuracyPair; entry: MetricEntry } => value.entry !== undefined);
  if (values.length === 0) {
    element.textContent = detailLabel(context.locale, "noAccuracy");
    return;
  }
  for (const [index, value] of values.entries()) {
    if (index > 0) element.append(document.createElement("br"));
    const item = document.createElement("span");
    item.className = "model-detail-accuracy-value";
    item.dataset.metric = value.pair.metric;
    item.textContent = showMetricLabels
      ? `${accuracyMetricName(value.pair, context.locale)}: ${accuracyCellText(value.entry.metric, context.locale)}`
      : accuracyCellText(value.entry.metric, context.locale);
    item.title = [value.pair.dataset, value.pair.scope, value.pair.artifact].filter(Boolean).join(" · ");
    element.append(item);
  }
}

export function renderRetentionValues(
  element: HTMLElement,
  pairs: AccuracyPair[],
  context: DetailContext,
  showMetricLabels: boolean
): void {
  if (pairs.length === 0) {
    element.textContent = detailLabel(context.locale, "notMeasured");
    return;
  }
  const rendered = new Set<string>();
  for (const pair of pairs) {
    const retention = getRetention(pair);
    const valueText = retention.status === "value" && retention.value !== undefined
      ? formatRetention(retention.value, context.locale)
      : detailLabel(context.locale, retention.status === "not-comparable"
        ? "notComparable"
        : retention.status === "not-applicable" ? "notApplicable" : "notMeasured");
    const text = showMetricLabels && retention.status === "value"
      ? `${accuracyMetricName(pair, context.locale)}: ${valueText}`
      : valueText;
    if (rendered.has(text)) continue;
    rendered.add(text);
    if (element.childNodes.length > 0) element.append(document.createElement("br"));
    const item = document.createElement("span");
    item.className = "model-detail-retention-value";
    item.dataset.retentionSource = retention.source ?? "none";
    item.textContent = text;
    if (retention.source === "derived") {
      item.title = context.locale === "zh" ? "由原始精度计算" : "Derived from source accuracies";
    }
    element.append(item);
  }
}
