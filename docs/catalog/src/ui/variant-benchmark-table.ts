import type { BenchmarkRecord, HardwareId, ModelVariant } from "../catalog/types";
import {
  groupPerformanceMetrics,
  pairAccuracyMetrics,
  type AccuracyPair,
  type PerformanceGroup,
  type MetricEntry
} from "../catalog/metric-display";
import { normalizeHardware } from "../catalog/variants";
import { stripHardwareSuffix } from "../catalog/model-naming";
import { canonicalMetricName, metricDisplayLabel } from "../catalog/metric-identity";
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
import {
  conditionText,
  hasConditions,
  renderTableConditions,
  summarizeConditions,
  sharedAccuracyScope,
  uniformColumnConditions,
  visibleParts,
  type ConditionsSummary,
  type UniformConditions
} from "./table-conditions";

/** The glyph that replaces repeated verbose missing-value wording in a cell. */
const MISSING_DASH = "—";

function taskFromVariant(variantId: string, fallback: string): string {
  const id = normalized(variantId);
  if (id.includes("-seg-") || id.includes("_seg_")) return "instance-segmentation";
  if (id.includes("-pose-") || id.includes("_pose-")) return "pose-estimation";
  if (id.includes("-cls-") || id.includes("_cls-")) return "image-classification";
  if (id.includes("-detect-") || id.includes("_detect-")) return "object-detection";
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

/**
 * The comparison defaults to single-thread whenever the source measured it,
 * for every model family. Additional threads are not dropped; they stay one
 * click away in the same table. A table that never measured one thread keeps
 * its smallest comparable set instead of inventing a column.
 */
function primaryThreadCounts(all: Array<number | undefined>): Array<number | undefined> {
  if (all.includes(1)) return [1];
  const preferred = all.filter((concurrency) => concurrency === undefined || concurrency === 2);
  if (preferred.length > 0) return preferred;
  return all.slice(0, 1);
}

function metricUnit(
  groups: PerformanceGroup[],
  concurrency: number | undefined,
  metric: "latency" | "throughput",
  locale: DetailContext["locale"]
): string | undefined {
  const units = new Set<string>();
  for (const group of groups) {
    const thread = group.threads.find((candidate) => candidate.concurrency === concurrency);
    const entry = metric === "latency" ? thread?.latency : thread?.throughput;
    if (entry !== undefined) units.add(unitLabel(locale, entry.metric.unit));
  }
  return units.size === 1 ? [...units][0] : undefined;
}

function perfColumnKey(concurrency: number | undefined, kind: "latency" | "throughput"): string {
  return `perf:${threadKey(concurrency)}:${kind}`;
}

function stageColumnKey(stage: string): string {
  return `stage:${stage}`;
}

/** Compact placeholder carrying the full wording for hover and screen readers. */
function renderMissingValue(
  element: HTMLElement,
  context: DetailContext,
  labelKey: "notRecorded" | "noPerformance" | "noAccuracy",
  dimension: "accuracy" | "performance"
): void {
  const label = detailLabel(context.locale, labelKey);
  element.textContent = MISSING_DASH;
  element.title = label;
  element.setAttribute("aria-label", label);
  element.dataset.empty = "true";
  element.dataset.missing = dimension;
}

/** Different observations share a configuration row, never a calculated value. */
function appendMetricValues(
  element: HTMLElement,
  entries: MetricEntry[],
  context: DetailContext,
  hasPerformance: boolean,
  uniform?: UniformConditions
): void {
  const unique = [...new Map(entries.map(entry => [JSON.stringify([entry.metric, entry.record.environment]), entry])).values()];
  if (!unique.length) {
    renderMissingValue(element, context, hasPerformance ? "notRecorded" : "noPerformance", "performance");
    return;
  }
  for (const entry of unique) {
    const item = document.createElement("div");
    item.className = "model-detail-observation";
    const value = document.createElement("span");
    value.className = "model-detail-metric-value";
    value.dataset.metric = entry.metric.metric;
    value.dataset.concurrency = threadKey(entry.metric.concurrency);
    value.textContent = metricCellText(entry.metric, context.locale);
    // Only the wording this column does not already state stays beside the
    // value. A column-wide condition is stated once in the conditions block,
    // while a stage, runtime or statistic that differs inside one cell keeps
    // its label so the two numbers remain distinguishable.
    const conditions = conditionText(visibleParts(entry, uniform));
    value.title = conditions;
    item.append(value);
    if (conditions) {
      const note = document.createElement("small");
      note.className = "model-detail-measurement-scope";
      note.textContent = conditions;
      note.title = conditions;
      item.append(document.createElement("br"), note);
    }
    element.append(item);
  }
}

function stageKey(group: PerformanceGroup): string | undefined {
  if (!group.measurement) return undefined;
  if (/^post[-_]?process(?:ing)?[-_]latency$/i.test(group.measurement)) return "post_process_latency";
  return group.measurement.replaceAll("_", "-");
}

function stageLabelFor(key: string, context: DetailContext): string {
  return key === "post_process_latency"
    ? (context.locale === "zh" ? "CPU 后处理" : "CPU post-processing")
    : metricDisplayLabel(canonicalMetricName(key), context.locale);
}

function accuracyColumnCount(columns: AccuracyColumnGroup[]): number {
  return columns.reduce((count, column) => count + column.stages.length + (column.showRetention ? 1 : 0), 0);
}

/** One rendered configuration, with its per-column measurements precomputed. */
interface PlannedRow {
  variant: ModelVariant;
  records: BenchmarkRecord[];
  groups: PerformanceGroup[];
  rowPairs: AccuracyPair[];
  cells: Map<string, MetricEntry[]>;
}

function planRow(
  variant: ModelVariant,
  hardware: HardwareId,
  threadCounts: Array<number | undefined>,
  stages: string[]
): PlannedRow {
  const records = variantRecords(variant, hardware);
  const groups = groupPerformanceMetrics(records);
  const primary = groups.filter(group => !group.measurement);
  const cells = new Map<string, MetricEntry[]>();
  for (const concurrency of threadCounts) {
    for (const kind of ["latency", "throughput"] as const) {
      cells.set(perfColumnKey(concurrency, kind), primary
        .flatMap(group => group.threads.filter(thread => thread.concurrency === concurrency)
          .flatMap(thread => thread[kind] ? [thread[kind]!] : [])));
    }
  }
  for (const stage of stages) {
    cells.set(stageColumnKey(stage), groups.filter(group => stageKey(group) === stage).flatMap(group => group.metrics));
  }
  return { variant, records, groups, rowPairs: pairAccuracyMetrics(records), cells };
}

function buildVariantRow(
  planned: PlannedRow,
  threadCounts: Array<number | undefined>,
  columns: AccuracyColumnGroup[],
  detailId: string,
  context: DetailContext,
  stages: string[],
  uniform: Map<string, UniformConditions>,
  accuracyScope?: string
): HTMLTableRowElement {
  const { variant, records, groups, rowPairs, cells } = planned;
  const row = document.createElement("tr");
  row.className = "model-detail-spec-row";
  row.dataset.variantId = variant.id;
  row.dataset.hardware = variant.hardware;
  row.dataset.task = variant.task;
  if (groups.length === 1 && groups[0]?.scope) row.dataset.scope = groups[0].scope;

  const specification = document.createElement("th");
  specification.scope = "row";
  specification.className = "model-detail-specification";
  specification.textContent = stripHardwareSuffix(variant.name || variant.id)
    .replace(/\s+(?:(?:Single|One|Two|Eight|Multi)[- ]Threads?|Statistics)$/i, "");
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

  const hasPerformance = records.some(record => (record.performance?.length ?? 0) > 0);
  for (const concurrency of threadCounts) {
    for (const kind of ["latency", "throughput"] as const) {
      const key = perfColumnKey(concurrency, kind);
      const target = document.createElement("td");
      target.className = "model-detail-thread-value model-detail-" + kind;
      target.dataset.concurrency = threadKey(concurrency);
      const entries = cells.get(key) ?? [];
      if (entries.length) target.dataset.metric = kind;
      appendMetricValues(target, entries, context, hasPerformance, uniform.get(key));
      row.append(target);
    }
  }
  for (const stage of stages) {
    const key = stageColumnKey(stage);
    const target = document.createElement("td");
    target.className = stage === "post_process_latency" ? "model-detail-postprocess" : "model-detail-stage-value";
    target.dataset.metric = stage;
    appendMetricValues(target, cells.get(key) ?? [], context, hasPerformance, uniform.get(key));
    row.append(target);
  }

  for (const column of columns) {
    for (const stage of column.stages) {
      const stageCell = document.createElement("td");
      stageCell.className = "model-detail-accuracy-cell";
      stageCell.classList.toggle("benchmark-group-start", stage === column.stages[0]);
      stageCell.dataset.metric = column.canonicalMetric;
      stageCell.dataset.stage = stage;
      renderStageCell(stageCell, column, rowPairs, stage, context, undefined, accuracyScope);
      row.append(stageCell);
    }
    if (column.showRetention) {
      const retentionCell = document.createElement("td");
      retentionCell.className = "model-detail-retention-cell";
      retentionCell.dataset.metric = column.canonicalMetric;
      renderRetentionCell(retentionCell, column, rowPairs, context);
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

/**
 * The table legend states what the compact placeholders mean. Missing accuracy
 * and missing performance get one entry each so a dash in an accuracy column
 * is never read as an absent benchmark.
 */
function renderTableLegend(context: DetailContext): HTMLElement {
  const legend = document.createElement("dl");
  legend.className = "model-detail-table-legend";
  const entries: Array<[string, string]> = [
    [MISSING_DASH, detailLabel(context.locale, "legendMissingAccuracy")],
    [MISSING_DASH, detailLabel(context.locale, "legendMissingPerformance")],
    [detailLabel(context.locale, "notApplicable"), detailLabel(context.locale, "legendNotApplicable")]
  ];
  for (const [term, description] of entries) {
    const item = document.createElement("div");
    const key = document.createElement("dt");
    key.textContent = term;
    const value = document.createElement("dd");
    value.textContent = description;
    item.append(key, value);
    legend.append(item);
  }
  return legend;
}

interface BuiltTable {
  table: HTMLTableElement;
  summary: ConditionsSummary;
  hasMissingCells: boolean;
}

function buildTable(
  variants: ModelVariant[],
  hardware: HardwareId,
  task: string,
  threadCounts: Array<number | undefined>,
  performanceGroups: PerformanceGroup[],
  columns: AccuracyColumnGroup[],
  context: DetailContext
): BuiltTable {
  const selected = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const stages = [...new Set(selected.flatMap(variant => groupPerformanceMetrics(variantRecords(variant, hardware)).map(stageKey)).filter((key): key is string => key !== undefined))];
  const result = table(detailLabel(context.locale, "performance"));
  result.className = "model-detail-specifications-table";
  const head = result.tHead!;
  const firstRow = document.createElement("tr");
  const specification = cell(detailLabel(context.locale, "specification"), true);
  specification.rowSpan = 2;
  const input = cell(detailLabel(context.locale, "input"), true);
  input.rowSpan = 2;
  firstRow.append(specification, input);

  const columnLabels = new Map<string, string>();
  for (const concurrency of threadCounts) {
    const latencyUnit = metricUnit(performanceGroups, concurrency, "latency", context.locale);
    const unknownMetrics = performanceGroups.flatMap(group => group.threads.filter(thread => thread.concurrency === undefined).flatMap(thread => thread.metrics));
    const unspecifiedMultithread = concurrency === undefined && unknownMetrics.length > 0
      && unknownMetrics.every(entry => /^multi[- ]thread/i.test(entry.metric.scope ?? ""));
    const threadLabel = unspecifiedMultithread
      ? (context.locale === "zh" ? "多线程（数量未记录）" : "Multi-thread (count not recorded)")
      : formatThreadLabel(concurrency, context.locale);
    const group = cell(`${threadLabel} ${context.locale === "zh" ? "延迟 / FPS" : "latency / FPS"}`, true);
    group.colSpan = 2;
    group.scope = "colgroup";
    group.dataset.concurrency = threadKey(concurrency);
    if (latencyUnit) group.dataset.latencyUnit = latencyUnit;
    firstRow.append(group);
    columnLabels.set(perfColumnKey(concurrency, "latency"), `${threadLabel} ${context.locale === "zh" ? "延迟" : "latency"}`);
    columnLabels.set(perfColumnKey(concurrency, "throughput"), `${threadLabel} FPS`);
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
  for (const stage of stages) {
    const header = cell(stageLabelFor(stage, context), true);
    header.rowSpan = 2;
    header.dataset.metric = stage;
    firstRow.append(header);
    columnLabels.set(stageColumnKey(stage), stageLabelFor(stage, context));
  }
  appendAccuracyHeader(firstRow, secondRow, columns, context);
  const download = cell(detailLabel(context.locale, "download"), true);
  download.rowSpan = 2;
  firstRow.append(download);
  head.append(firstRow, secondRow);

  const planned = selected.map(variant => planRow(variant, hardware, threadCounts, stages));
  const cellsByColumn = new Map([...new Set(planned.flatMap(row => [...row.cells.keys()]))]
    .map(key => [key, planned.map(row => row.cells.get(key) ?? [])]));
  // A wording that repeats in every populated cell of one column describes the
  // column, so it moves to the conditions block instead of printing each time.
  const uniform = uniformColumnConditions(cellsByColumn);

  const columnCount = 2 + threadCounts.length * 2 + accuracyColumnCount(columns) + stages.length + 1;
  for (const row of planned) {
    const detailId = ("model-detail-row-" + row.variant.id).replace(/[^a-zA-Z0-9_-]+/g, "-");
    const mainRow = buildVariantRow(row, threadCounts, columns, detailId, context, stages, uniform, sharedAccuracyScope(planned.flatMap(item => item.records)));
    const expandedRow = expandedVariantRow(row.variant, row.records, columnCount, detailId, context);
    result.tBodies[0]!.append(mainRow, expandedRow);
    wireRowToggle(mainRow, expandedRow, context);
  }
  if (selected.length === 0) appendEmptyRow(result, detailLabel(context.locale, "noPerformance"), columnCount);

  const summary = summarizeConditions({
    records: planned.flatMap(row => row.records),
    siblingRecords: variants.flatMap(variant => variant.benchmarks),
    columns: cellsByColumn,
    columnLabels,
    threadCounts,
    sourceRef: selected[0]?.release_tag ?? context.releaseTag,
    locale: context.locale
  });
  return {
    table: result,
    summary,
    hasMissingCells: result.querySelector('[data-empty="true"]') !== null
  };
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
  preferredThreads?: number[];
  variants: ModelVariant[];
  hardware: HardwareId;
  task: string;
  context: DetailContext;
}

/**
 * Render the platform comparison as one grouped table. Single-thread is the
 * default comparison whenever the source measured it; the complete source
 * thread set remains one click away in the same table.
 *
 * Accuracy columns come from the measurements this model actually publishes, so
 * a detector, a classifier and an embedding model each get their own table
 * shape rather than sharing one fixed template. Conditions that every cell of a
 * column repeats are stated once below the table, taken from the records
 * themselves.
 */
export function renderVariantBenchmarkTable(options: VariantBenchmarkTableOptions): HTMLElement {
  const { variants, hardware, task, context } = options;
  const selected = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const records = selected.flatMap((variant) => variantRecords(variant, hardware));
  const allGroups = groupPerformanceMetrics(records);
  const primaryGroups = allGroups.filter((group) =>
    !group.measurement && group.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
  );
  const columns = accuracyColumnGroups(pairAccuracyMetrics(records), context.locale);
  const allThreads = allThreadCounts(primaryGroups);
  let showAllThreads = false;
  const preferred = options.preferredThreads;
  const matching = preferred ? allThreads.filter(count => count === undefined || preferred.includes(count)) : [];
  const primaryThreads = matching.length ? matching : primaryThreadCounts(allThreads);
  if (primaryThreads.length === 0 && selected.length > 0 && !allGroups.some(group => group.measurement)) primaryThreads.push(undefined);
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
  const conditionsHost = document.createElement("div");
  conditionsHost.className = "model-detail-conditions-host";

  const render = (): void => {
    const threadCounts = showAllThreads ? allThreads : primaryThreads;
    tableHost.dataset.threadMode = showAllThreads ? "all" : "primary";
    toggle.textContent = showAllThreads
      ? (context.locale === "zh" ? "收起其他线程" : "Hide additional threads")
      : (context.locale === "zh" ? `显示其他线程（${extraThreads.length}）` : `Show additional threads (${extraThreads.length})`);
    const built = buildTable(variants, hardware, task, threadCounts, primaryGroups, columns, context);
    scrollHost.replaceChildren(wrapper(built.table, detailLabel(context.locale, "specifications")));
    const reference: HTMLElement[] = [];
    if (built.hasMissingCells) reference.push(renderTableLegend(context));
    if (hasConditions(built.summary)) reference.push(renderTableConditions(built.summary, context));
    conditionsHost.replaceChildren(...reference);
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
  tableHost.append(toolbar, scrollHost, conditionsHost);

  return tableHost;
}
