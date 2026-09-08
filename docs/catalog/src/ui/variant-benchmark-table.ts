import type { BenchmarkRecord, HardwareId, ModelVariant } from "../catalog/types";
import {
  groupPerformanceMetrics,
  pairAccuracyMetrics,
  type AccuracyPair,
  type PerformanceGroup,
  type PerformanceThread
} from "../catalog/metric-display";
import { normalizeHardware } from "../catalog/variants";
import { stripHardwareSuffix } from "../catalog/model-naming";
import {
  accuracyColumnGroups,
  columnConditionText,
  renderRetentionCell,
  renderStageCell,
  stageLabel,
  type AccuracyColumnGroup
} from "./accuracy-comparison";
import { renderDownloadCell, renderAssetDetails, runnableAssets } from "./artifact-downloads";
import type { DetailContext } from "./detail-types";
import { renderEvidenceDetails } from "./evidence-details";
import { detailLabel, unitLabel } from "./detail-labels";
import {
  appendEmptyRow,
  cell,
  formatThreadLabel,
  inputDescription,
  inputForVariant,
  metricCellText,
  normalized,
  threadKey,
  table,
  wrapper
} from "./detail-utils";

function taskFromVariant(variantId: string, fallback: string): string {
  const id = normalized(variantId);
  if (id.includes("-seg-") || id.includes("_seg_")) return "instance-segmentation";
  if (id.includes("-pose-") || id.includes("_pose_")) return "pose-estimation";
  if (id.includes("-cls-") || id.includes("_cls_")) return "image-classification";
  if (id.includes("-detect-") || id.includes("_detect_")) return "object-detection";
  return fallback;
}

export function variantRecords(variant: ModelVariant, hardware: HardwareId): BenchmarkRecord[] {
  return variant.benchmarks.filter((record) => {
    const recordHardware = normalizeHardware(record.environment.hardware);
    return recordHardware === undefined || recordHardware === hardware;
  });
}

function allThreadCounts(groups: PerformanceGroup[]): Array<number | undefined> {
  const values = new Set<number | undefined>();
  for (const group of groups) for (const thread of group.threads) values.add(thread.concurrency);
  return [...values].sort((left, right) => {
    if (left === undefined) return 1;
    if (right === undefined) return -1;
    return left - right;
  });
}

function primaryThreadCounts(all: Array<number | undefined>): Array<number | undefined> {
  const preferred = all.filter((concurrency) => concurrency === undefined || concurrency === 1 || concurrency === 2);
  if (preferred.length > 0) return preferred;
  return all.slice(0, 1);
}

function metricUnit(
  groups: PerformanceGroup[],
  concurrency: number | undefined,
  metric: "latency" | "throughput",
  locale: DetailContext["locale"]
): string | undefined {
  for (const group of groups) {
    const thread = group.threads.find((candidate) => candidate.concurrency === concurrency);
    const entry = metric === "latency" ? thread?.latency : thread?.throughput;
    if (entry !== undefined) return unitLabel(locale, entry.metric.unit);
  }
  return undefined;
}

function appendMetricValue(
  element: HTMLElement,
  entry: PerformanceThread["latency"],
  context: DetailContext,
  metricName: "latency" | "throughput",
  rowHasPerformance: boolean
): void {
  if (entry === undefined) {
    // “No performance table was published” and “this table reports latency but
    // no FPS for this thread” are different statements and must not share text.
    element.textContent = detailLabel(context.locale, rowHasPerformance ? "notRecorded" : "noPerformance");
    element.dataset.empty = "true";
    return;
  }
  const value = document.createElement("span");
  value.className = "model-detail-metric-value";
  value.dataset.metric = metricName;
  value.dataset.concurrency = threadKey(entry.metric.concurrency);
  value.textContent = metricCellText(entry.metric, context.locale);
  value.title = entry.metric.scope ?? "";
  element.append(value);
}

function accuracyColumnCount(columns: AccuracyColumnGroup[]): number {
  return columns.reduce((count, column) => count + column.stages.length + (column.showRetention ? 1 : 0), 0);
}

function buildVariantRow(
  variant: ModelVariant,
  records: BenchmarkRecord[],
  group: PerformanceGroup | undefined,
  threadCounts: Array<number | undefined>,
  columns: AccuracyColumnGroup[],
  rowPairs: AccuracyPair[],
  detailId: string,
  context: DetailContext
): HTMLTableRowElement {
  const row = document.createElement("tr");
  row.className = "model-detail-spec-row";
  row.dataset.variantId = variant.id;
  row.dataset.hardware = variant.hardware;
  row.dataset.task = variant.task;
  if (group?.scope) row.dataset.scope = group.scope;
  if (group?.statistic) row.dataset.statistic = group.statistic;

  const specification = document.createElement("th");
  specification.scope = "row";
  specification.className = "model-detail-specification";
  // The hardware tab already states the board, so `on RDK S100` is dropped
  // instead of being repeated on every row.
  specification.textContent = stripHardwareSuffix(variant.name || variant.id);
  if (group !== undefined) {
    const conditions = [
      group.scope ? `${detailLabel(context.locale, "scope")}: ${group.scope}` : undefined,
      group.statistic ? `${detailLabel(context.locale, "statistic")}: ${group.statistic}` : undefined
    ].filter((value): value is string => Boolean(value));
    if (conditions.length > 0) {
      const note = document.createElement("small");
      note.className = "model-detail-measurement-scope";
      note.textContent = conditions.join(" · ");
      specification.append(document.createElement("br"), note);
    }
  }
  const rowToggle = document.createElement("button");
  rowToggle.type = "button";
  rowToggle.className = "model-detail-row-toggle";
  rowToggle.dataset.action = "toggle-row-details";
  rowToggle.textContent = detailLabel(context.locale, "details");
  rowToggle.setAttribute("aria-expanded", "false");
  rowToggle.setAttribute("aria-controls", detailId);
  specification.append(document.createElement("br"), rowToggle);
  row.append(specification);

  const input = document.createElement("td");
  input.className = "model-detail-input";
  input.textContent = inputDescription(inputForVariant(variant)) || detailLabel(context.locale, "notRecorded");
  row.append(input);

  const byConcurrency = new Map<number | undefined, PerformanceThread>();
  for (const thread of group?.threads ?? []) byConcurrency.set(thread.concurrency, thread);
  for (const concurrency of threadCounts) {
    const thread = byConcurrency.get(concurrency);
    const latency = document.createElement("td");
    latency.className = "model-detail-thread-value model-detail-latency";
    if (thread?.latency) latency.dataset.metric = "latency";
    latency.dataset.concurrency = threadKey(concurrency);
    appendMetricValue(latency, thread?.latency, context, "latency", group !== undefined);
    const throughput = document.createElement("td");
    throughput.className = "model-detail-thread-value model-detail-throughput";
    if (thread?.throughput) throughput.dataset.metric = "throughput";
    throughput.dataset.concurrency = threadKey(concurrency);
    appendMetricValue(throughput, thread?.throughput, context, "throughput", group !== undefined);
    row.append(latency, throughput);
  }

  for (const column of columns) {
    for (const stage of column.stages) {
      const stageCell = document.createElement("td");
      stageCell.className = "model-detail-accuracy-cell";
      stageCell.dataset.metric = column.canonicalMetric;
      stageCell.dataset.stage = stage;
      renderStageCell(stageCell, column, rowPairs, stage, context, group?.scope);
      row.append(stageCell);
    }
    if (column.showRetention) {
      const retentionCell = document.createElement("td");
      retentionCell.className = "model-detail-retention-cell";
      retentionCell.dataset.metric = column.canonicalMetric;
      renderRetentionCell(retentionCell, column, rowPairs, context, group?.scope);
      row.append(retentionCell);
    }
  }

  const downloads = document.createElement("td");
  downloads.className = "model-detail-download-cell";
  renderDownloadCell(downloads, runnableAssets(variant), context);
  row.append(downloads);
  return row;
}

function expandedVariantRow(
  variant: ModelVariant,
  records: BenchmarkRecord[],
  columnCount: number,
  detailId: string,
  context: DetailContext
): HTMLTableRowElement {
  const row = document.createElement("tr");
  row.className = "model-detail-expanded-row";
  row.hidden = true;
  row.id = detailId;
  row.dataset.variantId = variant.id;
  row.dataset.hardware = variant.hardware;
  row.dataset.task = variant.task;
  const detailCell = document.createElement("td");
  detailCell.colSpan = columnCount;

  const details = document.createElement("details");
  details.className = "model-detail-row-details";
  const summary = document.createElement("summary");
  summary.textContent = detailLabel(context.locale, "details");
  details.append(summary);
  details.append(renderEvidenceDetails(records, context), renderAssetDetails(variant, context));
  detailCell.append(details);
  row.append(detailCell);
  return row;
}

function appendAccuracyHeader(
  firstRow: HTMLTableRowElement,
  secondRow: HTMLTableRowElement,
  columns: AccuracyColumnGroup[],
  context: DetailContext
): void {
  for (const column of columns) {
    const span = column.stages.length + (column.showRetention ? 1 : 0);
    const group = document.createElement("th");
    group.scope = "colgroup";
    group.colSpan = span;
    group.className = "model-detail-accuracy-group";
    group.dataset.metric = column.canonicalMetric;
    const label = document.createElement("span");
    label.className = "model-detail-accuracy-group-label";
    label.textContent = column.label;
    group.append(label);
    const condition = columnConditionText(column, context.locale);
    if (condition) {
      const note = document.createElement("small");
      note.className = "model-detail-metric-scale";
      note.textContent = condition;
      group.append(document.createElement("br"), note);
    }
    firstRow.append(group);
    for (const stage of column.stages) {
      const head = cell(stageLabel(stage, context.locale), true);
      head.dataset.metric = column.canonicalMetric;
      head.dataset.stage = stage;
      secondRow.append(head);
    }
    if (column.showRetention) {
      const head = cell(detailLabel(context.locale, "retention"), true);
      head.dataset.metric = column.canonicalMetric;
      head.dataset.retention = "true";
      secondRow.append(head);
    }
  }
}

function buildTable(
  variants: ModelVariant[],
  hardware: HardwareId,
  task: string,
  threadCounts: Array<number | undefined>,
  performanceGroups: PerformanceGroup[],
  columns: AccuracyColumnGroup[],
  context: DetailContext
): HTMLTableElement {
  const selected = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const result = table(detailLabel(context.locale, "performance"));
  result.className = "model-detail-specifications-table";
  const head = result.tHead!;
  const firstRow = document.createElement("tr");
  const specification = cell(detailLabel(context.locale, "specification"), true);
  specification.rowSpan = 2;
  const input = cell(detailLabel(context.locale, "input"), true);
  input.rowSpan = 2;
  firstRow.append(specification, input);
  for (const concurrency of threadCounts) {
    const latencyUnit = metricUnit(performanceGroups, concurrency, "latency", context.locale);
    const group = cell(`${formatThreadLabel(concurrency, context.locale)} ${context.locale === "zh" ? "延迟 / FPS" : "latency / FPS"}`, true);
    group.colSpan = 2;
    group.scope = "colgroup";
    group.dataset.concurrency = threadKey(concurrency);
    if (latencyUnit) group.dataset.latencyUnit = latencyUnit;
    firstRow.append(group);
  }
  const secondRow = document.createElement("tr");
  for (const concurrency of threadCounts) {
    const latencyUnit = metricUnit(performanceGroups, concurrency, "latency", context.locale);
    const latency = cell(`${context.locale === "zh" ? "延迟" : "Latency"}${latencyUnit ? ` (${latencyUnit})` : ""}`, true);
    latency.dataset.metric = "latency";
    latency.dataset.concurrency = threadKey(concurrency);
    const throughput = cell("FPS", true);
    throughput.dataset.metric = "throughput";
    throughput.dataset.concurrency = threadKey(concurrency);
    secondRow.append(latency, throughput);
  }
  appendAccuracyHeader(firstRow, secondRow, columns, context);
  const download = cell(detailLabel(context.locale, "download"), true);
  download.rowSpan = 2;
  firstRow.append(download);
  head.append(firstRow, secondRow);

  const columnCount = 2 + threadCounts.length * 2 + accuracyColumnCount(columns) + 1;
  for (const variant of selected) {
    const variantRecordsList = variantRecords(variant, hardware);
    const rowPairs = pairAccuracyMetrics(variantRecordsList);
    const groups = groupPerformanceMetrics(variantRecordsList).filter((candidate) =>
      candidate.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
    );
    if (groups.length === 0) {
      const detailId = `model-detail-row-${variant.id}-default`.replace(/[^a-zA-Z0-9_-]+/g, "-");
      const mainRow = buildVariantRow(variant, variantRecordsList, undefined, threadCounts, columns, rowPairs, detailId, context);
      const expandedRow = expandedVariantRow(variant, variantRecordsList, columnCount, detailId, context);
      result.tBodies[0]!.append(mainRow, expandedRow);
      wireRowToggle(mainRow, expandedRow, context);
      continue;
    }
    for (const [groupIndex, group] of groups.entries()) {
      const groupKey = `${groupIndex}-${group.scope ?? "scope"}-${group.statistic ?? "statistic"}`;
      const detailId = `model-detail-row-${variant.id}-${groupKey}`.replace(/[^a-zA-Z0-9_-]+/g, "-");
      const mainRow = buildVariantRow(variant, variantRecordsList, group, threadCounts, columns, rowPairs, detailId, context);
      const expandedRow = expandedVariantRow(variant, variantRecordsList, columnCount, detailId, context);
      result.tBodies[0]!.append(mainRow, expandedRow);
      wireRowToggle(mainRow, expandedRow, context);
    }
  }
  if (selected.length === 0) appendEmptyRow(result, detailLabel(context.locale, "noPerformance"), columnCount);
  return result;
}

function wireRowToggle(
  mainRow: HTMLTableRowElement,
  expandedRow: HTMLTableRowElement,
  context: DetailContext
): void {
  const toggle = mainRow.querySelector<HTMLButtonElement>('[data-action="toggle-row-details"]');
  const details = expandedRow.querySelector<HTMLDetailsElement>(".model-detail-row-details");
  if (!toggle || !details) return;
  toggle.addEventListener("click", () => {
    const expanded = expandedRow.hidden;
    expandedRow.hidden = !expanded;
    details.open = expanded;
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.textContent = expanded
      ? (context.locale === "zh" ? "收起详情" : "Hide details")
      : detailLabel(context.locale, "details");
  });
}

export interface VariantBenchmarkTableOptions {
  variants: ModelVariant[];
  hardware: HardwareId;
  task: string;
  context: DetailContext;
}

/**
 * Render the platform comparison as one grouped table. The primary view keeps
 * 1/2-thread columns (plus explicitly unknown concurrency) compact; the
 * complete source thread set remains one click away in the same table.
 *
 * Accuracy columns come from the measurements this model actually publishes, so
 * a detector, a classifier and an embedding model each get their own table
 * shape rather than sharing one fixed template.
 */
export function renderVariantBenchmarkTable(options: VariantBenchmarkTableOptions): HTMLElement {
  const { variants, hardware, task, context } = options;
  const selected = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const records = selected.flatMap((variant) => variantRecords(variant, hardware));
  const allGroups = groupPerformanceMetrics(records);
  const primaryGroups = allGroups.filter((group) =>
    group.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
  );
  const columns = accuracyColumnGroups(pairAccuracyMetrics(records), context.locale);
  const allThreads = allThreadCounts(primaryGroups);
  let showAllThreads = false;
  const primaryThreads = primaryThreadCounts(allThreads);
  if (primaryThreads.length === 0 && selected.length > 0) primaryThreads.push(undefined);
  const tableHost = document.createElement("div");
  tableHost.className = "model-detail-benchmark-table";
  const initialThreadMode = typeof window !== "undefined"
    && new URL(window.location.href).searchParams.get("threads") === "all";
  showAllThreads = initialThreadMode && allThreads.length > 0;
  tableHost.dataset.threadMode = showAllThreads ? "all" : "primary";
  const toolbar = document.createElement("div");
  toolbar.className = "model-detail-benchmark-toolbar";
  const extraThreads = allThreads.filter((concurrency) => !primaryThreads.includes(concurrency));
  toolbar.hidden = extraThreads.length === 0;
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "model-detail-thread-toggle";
  toggle.dataset.action = "toggle-threads";
  toggle.hidden = extraThreads.length === 0;
  toolbar.append(toggle);
  const scrollHost = document.createElement("div");

  const render = (): void => {
    const threadCounts = showAllThreads ? allThreads : primaryThreads;
    tableHost.dataset.threadMode = showAllThreads ? "all" : "primary";
    toggle.textContent = showAllThreads
      ? (context.locale === "zh" ? "收起其他线程" : "Hide additional threads")
      : (context.locale === "zh" ? `显示其他线程（${extraThreads.length}）` : `Show additional threads (${extraThreads.length})`);
    scrollHost.replaceChildren(wrapper(
      buildTable(variants, hardware, task, threadCounts, primaryGroups, columns, context),
      detailLabel(context.locale, "specifications")
    ));
  };
  toggle.addEventListener("click", () => {
    showAllThreads = !showAllThreads;
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      if (showAllThreads) url.searchParams.set("threads", "all");
      else url.searchParams.delete("threads");
      window.history.replaceState({}, "", url);
    }
    render();
  });
  render();
  tableHost.append(toolbar, scrollHost);
  return tableHost;
}
