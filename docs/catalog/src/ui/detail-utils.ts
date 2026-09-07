import type { BenchmarkRecord, Locale, MetricRecord, ModelVariant } from "../catalog/types";
import { formatMetricValue } from "../catalog/metric-display";
import { detailLabel, unitLabel } from "./detail-labels";

export function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
}

export function sourceUrl(record: BenchmarkRecord, repositoryUrl: string): string {
  return `${repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(record.source.ref)}/${record.source.path}`;
}

export function cell(value: string, header = false): HTMLTableCellElement {
  const element = document.createElement(header ? "th" : "td");
  element.textContent = value;
  if (header) element.scope = "col";
  return element;
}

export function wrapper(tableElement: HTMLTableElement, label: string): HTMLElement {
  const region = document.createElement("div");
  region.className = "table-scroll model-detail-table-scroll";
  region.tabIndex = 0;
  region.setAttribute("role", "region");
  region.setAttribute("aria-label", label);
  region.append(tableElement);
  return region;
}

export function table(captionText: string): HTMLTableElement {
  const result = document.createElement("table");
  const caption = document.createElement("caption");
  caption.textContent = captionText;
  result.append(caption, document.createElement("thead"), document.createElement("tbody"));
  return result;
}

export function appendEmptyRow(tableElement: HTMLTableElement, text: string, columns: number): void {
  const row = document.createElement("tr");
  const value = cell(text);
  value.colSpan = columns;
  row.append(value);
  tableElement.tBodies[0]!.append(row);
}

export function inputDescription(input: BenchmarkRecord["input"] | undefined): string {
  const parts = [
    input?.shape?.join("×"),
    input?.layout,
    input?.format
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

export function inputForVariant(variant: ModelVariant): BenchmarkRecord["input"] {
  return variant.input ?? variant.benchmarks.find((record) => record.input !== undefined)?.input;
}

export function metricCellText(metric: MetricRecord, locale: Locale): string {
  const value = formatMetricValue(metric, locale, { asPercentage: false });
  return metric.unit === "fps" || metric.unit === "ms" || metric.unit === "us"
    ? value.replace(metric.unit, ` ${unitLabel(locale, metric.unit)}`)
    : value;
}

export function accuracyCellText(metric: MetricRecord, locale: Locale): string {
  const value = formatMetricValue(metric, locale, { asPercentage: true });
  // Keep the existing catalog's readable number/unit spacing for raw values;
  // retention remains compact (for example 95.42%) in the comparison column.
  return value.endsWith("%") ? `${value.slice(0, -1)} %` : `${value} ${unitLabel(locale, metric.unit)}`;
}

export function formatThreadLabel(concurrency: number | undefined, locale: Locale): string {
  return concurrency === undefined
    ? detailLabel(locale, "unknownConcurrency")
    : `${concurrency} ${locale === "zh" ? "线程" : "thread"}`;
}

export function threadKey(concurrency: number | undefined): string {
  return concurrency === undefined ? "unknown" : String(concurrency);
}
