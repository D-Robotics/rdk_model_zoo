import type { BenchmarkRecord, Locale, MetricRecord, ModelRecord } from "../catalog/types";
import { t, type TranslationKey } from "../i18n/translations";

export interface DetailContext {
  locale: Locale;
  repositoryUrl: string;
  releaseTag: string;
}

const qualifierKeys: Record<NonNullable<MetricRecord["qualifier"]>, TranslationKey> = {
  exact: "qualifier.exact",
  "lower-bound": "qualifier.lowerBound",
  "upper-bound": "qualifier.upperBound",
  approximate: "qualifier.approximate"
};

export function readModelId(url: URL): string | null {
  const value = url.searchParams.get("model")?.trim();
  return value ? value : null;
}

export function writeModelId(url: URL, modelId: string | null): URL {
  const next = new URL(url.href);
  const value = modelId?.trim();
  if (value) next.searchParams.set("model", value);
  else next.searchParams.delete("model");
  return next;
}

function cell(value: string, header = false): HTMLTableCellElement {
  const element = document.createElement(header ? "th" : "td");
  element.textContent = value;
  if (header) element.scope = "col";
  return element;
}

function tableWithCaption(captionText: string, headers: string[]): HTMLTableElement {
  const table = document.createElement("table");
  const caption = document.createElement("caption");
  caption.textContent = captionText;
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const header of headers) row.append(cell(header, true));
  head.append(row);
  table.append(caption, head, document.createElement("tbody"));
  return table;
}

function wrapTable(table: HTMLTableElement, label: string): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "table-scroll";
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", label);
  wrapper.append(table);
  return wrapper;
}

function inputDescription(record: BenchmarkRecord): string {
  const parts = [
    record.input?.shape?.join("×"),
    record.input?.layout,
    record.input?.format
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

function sourceUrl(record: BenchmarkRecord, repositoryUrl: string): string {
  return `${repositoryUrl}/blob/${encodeURIComponent(record.source.ref)}/${record.source.path}`;
}

function sourceCell(record: BenchmarkRecord, context: DetailContext): HTMLTableCellElement {
  const value = document.createElement("td");
  const link = document.createElement("a");
  link.dataset.testid = "benchmark-source";
  link.href = sourceUrl(record, context.repositoryUrl);
  link.textContent = record.source.section;
  value.append(link);
  return value;
}

function metricValue(metric: MetricRecord, locale: Locale): string {
  return new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en-US", {
    maximumFractionDigits: 6
  }).format(metric.value);
}

function metricConditions(record: BenchmarkRecord, metric: MetricRecord, locale: Locale): string {
  const parts = [
    `${t(locale, "details.hardware")}: ${record.environment.hardware}`,
    record.environment.runtime ? `${t(locale, "details.runtime")}: ${record.environment.runtime}` : undefined,
    record.environment.cpu_mode ? `${t(locale, "details.cpuMode")}: ${record.environment.cpu_mode}` : undefined,
    record.environment.bpu_cores !== undefined
      ? `${t(locale, "details.bpuCores")}: ${record.environment.bpu_cores}`
      : undefined,
    inputDescription(record) || undefined,
    metric.scope,
    metric.concurrency !== undefined
      ? `${t(locale, "details.concurrency")}: ${metric.concurrency}`
      : undefined
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

function appendMetricRows(
  table: HTMLTableElement,
  records: BenchmarkRecord[],
  kind: "performance" | "accuracy",
  context: DetailContext
): number {
  const body = table.tBodies[0]!;
  let count = 0;
  for (const record of records) {
    for (const metric of record[kind] ?? []) {
      const row = document.createElement("tr");
      row.append(
        cell(record.display_name),
        cell(metric.metric),
        cell(metricValue(metric, context.locale)),
        cell(metric.unit),
        cell(t(context.locale, qualifierKeys[metric.qualifier ?? "exact"])),
        cell(metricConditions(record, metric, context.locale)),
        cell(metric.dataset ?? t(context.locale, "missing.unspecified")),
        cell(metric.model_stage ?? t(context.locale, "missing.unspecified")),
        sourceCell(record, context)
      );
      body.append(row);
      count += 1;
    }
  }
  return count;
}

function emptyTableRow(table: HTMLTableElement, message: string, columns: number): void {
  const row = document.createElement("tr");
  const value = cell(message);
  value.colSpan = columns;
  row.append(value);
  table.tBodies[0]!.append(row);
}

function variantsTable(model: ModelRecord, context: DetailContext): HTMLElement {
  const table = tableWithCaption(t(context.locale, "details.variants"), [
    t(context.locale, "details.variants"),
    t(context.locale, "details.modelFormat"),
    t(context.locale, "details.precision"),
    t(context.locale, "details.input"),
    t(context.locale, "details.hardware"),
    t(context.locale, "details.runtime")
  ]);
  for (const record of model.benchmarks) {
    const row = document.createElement("tr");
    row.append(
      cell(record.display_name),
      cell(record.model_format ?? t(context.locale, "missing.unspecified")),
      cell(record.precision ?? t(context.locale, "missing.unspecified")),
      cell(inputDescription(record) || t(context.locale, "missing.unspecified")),
      cell(record.environment.hardware),
      cell(record.environment.runtime ?? t(context.locale, "missing.unspecified"))
    );
    table.tBodies[0]!.append(row);
  }
  if (model.benchmarks.length === 0) emptyTableRow(table, t(context.locale, "details.noPublishedData"), 6);
  return wrapTable(table, t(context.locale, "details.variants"));
}

function metricTable(model: ModelRecord, kind: "performance" | "accuracy", context: DetailContext): HTMLElement {
  const captionKey = kind === "performance" ? "details.performance" : "details.accuracy";
  const table = tableWithCaption(t(context.locale, captionKey), [
    t(context.locale, "details.variants"),
    t(context.locale, "details.metric"),
    t(context.locale, "details.value"),
    t(context.locale, "details.unit"),
    t(context.locale, "details.qualifier"),
    t(context.locale, "details.conditions"),
    t(context.locale, "details.dataset"),
    t(context.locale, "details.modelStage"),
    t(context.locale, "details.source")
  ]);
  const count = appendMetricRows(table, model.benchmarks, kind, context);
  if (count === 0) {
    emptyTableRow(
      table,
      t(context.locale, kind === "performance" ? "details.noPerformanceData" : "details.noAccuracyData"),
      9
    );
  }
  return wrapTable(table, t(context.locale, captionKey));
}

function assetsTable(model: ModelRecord, context: DetailContext): HTMLElement {
  const table = tableWithCaption(t(context.locale, "details.assets"), [
    t(context.locale, "model.asset"),
    t(context.locale, "details.modelFormat"),
    t(context.locale, "model.availability"),
    t(context.locale, "details.checksum")
  ]);
  for (const asset of model.assets) {
    const row = document.createElement("tr");
    const assetCell = document.createElement("td");
    if (asset.url) {
      const link = document.createElement("a");
      link.href = asset.url;
      link.download = asset.filename;
      link.textContent = asset.filename;
      assetCell.append(link);
    } else {
      assetCell.textContent = asset.filename;
    }
    row.append(
      assetCell,
      cell(asset.format),
      cell(t(context.locale, asset.url ? "details.available" : "details.manualDownload")),
      cell(asset.sha256 ?? t(context.locale, "details.checksumUnknown"))
    );
    table.tBodies[0]!.append(row);
  }
  if (model.assets.length === 0) emptyTableRow(table, t(context.locale, "details.noAssets"), 4);
  return wrapTable(table, t(context.locale, "details.assets"));
}

export function renderModelDetails(model: ModelRecord, context: DetailContext): HTMLElement {
  const supportsDialog = typeof HTMLDialogElement !== "undefined"
    && typeof HTMLDialogElement.prototype.showModal === "function";
  const root = document.createElement(supportsDialog ? "dialog" : "section");
  root.className = "model-details";
  root.dataset.modelId = model.id;
  if (!supportsDialog) {
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
  }

  const heading = document.createElement("h2");
  heading.id = `model-details-${model.id}`;
  heading.textContent = t(context.locale, "details.modelTitle", { name: model.name });
  root.setAttribute("aria-labelledby", heading.id);
  const close = document.createElement("button");
  close.type = "button";
  close.dataset.action = "close-details";
  close.textContent = t(context.locale, "details.close");
  close.setAttribute("aria-label", t(context.locale, "details.close"));

  const sample = document.createElement("a");
  sample.href = `${context.repositoryUrl}/blob/${encodeURIComponent(context.releaseTag)}/${model.sample_path}/README.md`;
  sample.textContent = t(context.locale, "details.viewSample");

  root.append(
    heading,
    close,
    sample,
    variantsTable(model, context),
    metricTable(model, "performance", context),
    metricTable(model, "accuracy", context),
    assetsTable(model, context)
  );
  return root;
}
