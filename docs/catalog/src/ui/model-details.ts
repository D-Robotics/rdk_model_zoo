import type { BenchmarkRecord, HardwareId, Locale, MetricRecord, ModelRecord, ModelVariant } from "../catalog/types";
import {
  formatMetricValue,
  formatRetention,
  getRetention,
  groupPerformanceMetrics,
  pairAccuracyMetrics,
  type AccuracyPair,
  type MetricEntry,
  type PerformanceGroup,
  type PerformanceThread
} from "../catalog/metric-display";
import { getHardwareIds, getModelVariants, HARDWARE_IDS, isRunnableAsset, normalizeHardware } from "../catalog/variants";
import {
  detailLabel,
  hardwareLabel,
  metricLabel,
  taskLabel,
  unitLabel
} from "./detail-labels";

export interface DetailContext {
  locale: Locale;
  repositoryUrl: string;
  releaseTag: string;
  hardware?: HardwareId;
  task?: string;
  onSelectionChange?: (hardware: HardwareId, task: string) => void;
  /** Kept for links and integrations written before hardware tabs were added. */
  platform?: string;
}

const taskPriority = [
  "object-detection",
  "instance-segmentation",
  "pose-estimation",
  "image-classification"
];

function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
}

function sourceUrl(record: BenchmarkRecord, repositoryUrl: string): string {
  return `${repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(record.source.ref)}/${record.source.path}`;
}

function cell(value: string, header = false): HTMLTableCellElement {
  const element = document.createElement(header ? "th" : "td");
  element.textContent = value;
  if (header) element.scope = "col";
  return element;
}

function inputDescription(input: BenchmarkRecord["input"] | undefined): string {
  const parts = [
    input?.shape?.join("×"),
    input?.layout,
    input?.format
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

function inputForVariant(variant: ModelVariant): BenchmarkRecord["input"] {
  return variant.input ?? variant.benchmarks.find((record) => record.input !== undefined)?.input;
}

function displayMetricNumber(metric: MetricRecord, locale: Locale, accuracy = false): string {
  return formatMetricValue(metric, locale, { asPercentage: accuracy });
}

function metricCellText(metric: MetricRecord, locale: Locale): string {
  const value = displayMetricNumber(metric, locale);
  return metric.unit === "fps" || metric.unit === "ms" || metric.unit === "us"
    ? value.replace(metric.unit, ` ${unitLabel(locale, metric.unit)}`)
    : value;
}

function accuracyCellText(metric: MetricRecord, locale: Locale): string {
  const value = displayMetricNumber(metric, locale, true);
  // Keep the existing catalog's readable number/unit spacing for raw values;
  // retention remains compact (for example 95.42%) below.
  return value.endsWith("%") ? `${value.slice(0, -1)} %` : `${value} ${unitLabel(locale, metric.unit)}`;
}

function sourceLink(record: BenchmarkRecord, context: DetailContext): HTMLAnchorElement {
  const link = document.createElement("a");
  link.dataset.testid = "benchmark-source";
  link.href = sourceUrl(record, context.repositoryUrl);
  link.textContent = record.source.section;
  link.title = `${record.source.ref}: ${record.source.path}`;
  return link;
}

function wrapper(table: HTMLTableElement, label: string): HTMLElement {
  const region = document.createElement("div");
  region.className = "table-scroll model-detail-table-scroll";
  region.tabIndex = 0;
  region.setAttribute("role", "region");
  region.setAttribute("aria-label", label);
  region.append(table);
  return region;
}

function table(captionText: string): HTMLTableElement {
  const result = document.createElement("table");
  const caption = document.createElement("caption");
  caption.textContent = captionText;
  result.append(caption, document.createElement("thead"), document.createElement("tbody"));
  return result;
}

function appendEmptyRow(tableElement: HTMLTableElement, text: string, columns: number): void {
  const row = document.createElement("tr");
  const value = cell(text);
  value.colSpan = columns;
  row.append(value);
  tableElement.tBodies[0]!.append(row);
}

function variantRecords(variant: ModelVariant, hardware: HardwareId): BenchmarkRecord[] {
  return variant.benchmarks.filter((record) => {
    const recordHardware = normalizeHardware(record.environment.hardware);
    return recordHardware === undefined || recordHardware === hardware;
  });
}

function availableVariants(model: ModelRecord, context: DetailContext): ModelVariant[] {
  const shared = getModelVariants(model);
  if (shared.length > 0) return shared;

  // This fallback keeps old hand-authored fixtures and ?model= links usable
  // while a generated catalog is upgraded to explicit variants.
  const platformModels = model.platforms ?? [{
    ...model,
    platform: "x5" as const,
    release_tag: context.releaseTag
  }];
  const result: ModelVariant[] = [];
  for (const platform of platformModels) {
    const grouped = new Map<string, BenchmarkRecord[]>();
    for (const record of platform.benchmarks) {
      const records = grouped.get(record.variant_id) ?? [];
      records.push(record);
      grouped.set(record.variant_id, records);
    }
    const platformHardware = normalizeHardware(platform.platform);
    for (const [id, benchmarks] of grouped) {
      const hardware = normalizeHardware(benchmarks[0]?.environment.hardware ?? "") ?? platformHardware;
      if (hardware === undefined) continue;
      result.push({
        id,
        name: benchmarks[0]?.display_name ?? id,
        hardware,
        task: taskFromVariant(benchmarks[0]?.variant_id ?? id, model.tasks[0] ?? ""),
        input: benchmarks.find((record) => record.input !== undefined)?.input,
        assets: platform.assets,
        benchmarks,
        sample_path: platform.sample_path,
        release_tag: platform.release_tag
      });
    }
  }
  return result;
}

function taskFromVariant(variantId: string, fallback: string): string {
  const id = normalized(variantId);
  if (id.includes("-seg-") || id.includes("_seg_")) return "instance-segmentation";
  if (id.includes("-pose-") || id.includes("_pose_")) return "pose-estimation";
  if (id.includes("-cls-") || id.includes("_cls_")) return "image-classification";
  if (id.includes("-detect-") || id.includes("_detect_")) return "object-detection";
  return fallback;
}

function orderedTasks(variants: ModelVariant[], hardware: HardwareId): string[] {
  const tasks = [...new Set(variants.filter((variant) => variant.hardware === hardware)
    .map((variant) => variant.task || taskFromVariant(variant.id, ""))
    .filter(Boolean))];
  return tasks.sort((left, right) => {
    const leftPriority = taskPriority.indexOf(left);
    const rightPriority = taskPriority.indexOf(right);
    if (leftPriority >= 0 && rightPriority >= 0) return leftPriority - rightPriority;
    if (leftPriority >= 0) return -1;
    if (rightPriority >= 0) return 1;
    return left.localeCompare(right);
  });
}

function variantHasMeasurements(variant: ModelVariant): boolean {
  return variant.benchmarks.some((record) =>
    (record.performance?.length ?? 0) > 0 || (record.accuracy?.length ?? 0) > 0
  );
}

function variantIsAuxiliary(variant: ModelVariant): boolean {
  const value = normalized(`${variant.id} ${variant.name}`);
  return value.includes("-cls-") || value.includes("_cls_") || value.includes("classify");
}

function variantSizeRank(variant: ModelVariant): number {
  const value = normalized(`${variant.id} ${variant.name}`);
  const size = value.match(/(?:yolov?\d+|yolo\d+)[-_ ]*([nsm lx])(?:[-_ ]|$)/)?.[1];
  if (size) return ["n", "s", "m", "l", "x"].indexOf(size);
  const mobile = value.match(/mobilenetv?(\d+)/)?.[1];
  if (mobile) return Number(mobile) + 10;
  return 99;
}

function orderedVariants(variants: ModelVariant[]): ModelVariant[] {
  return [...variants].sort((left, right) => {
    const measured = Number(variantHasMeasurements(right)) - Number(variantHasMeasurements(left));
    if (measured !== 0) return measured;
    const auxiliary = Number(variantIsAuxiliary(left)) - Number(variantIsAuxiliary(right));
    if (auxiliary !== 0) return auxiliary;
    return variantSizeRank(left) - variantSizeRank(right)
      || left.name.localeCompare(right.name)
      || left.id.localeCompare(right.id);
  });
}

function preferredHardware(model: ModelRecord, variants: ModelVariant[], context: DetailContext): HardwareId {
  const supported = new Set(getHardwareIds(model).filter((hardware) => variants.some((variant) => variant.hardware === hardware)));
  const fromContext = context.hardware ?? (context.platform ? normalizeHardware(context.platform) : undefined);
  if (fromContext !== undefined && supported.has(fromContext)) return fromContext;
  for (const hardware of HARDWARE_IDS) if (supported.has(hardware)) return hardware;
  return variants[0]?.hardware ?? "x5";
}

function preferredTask(tasks: string[], context: DetailContext): string {
  if (context.task !== undefined && tasks.includes(context.task)) return context.task;
  return tasks[0] ?? "";
}

function formatThreadLabel(concurrency: number | undefined, locale: Locale): string {
  return concurrency === undefined
    ? detailLabel(locale, "unknownConcurrency")
    : `${concurrency} ${locale === "zh" ? "线程" : "thread"}`;
}

function threadKey(concurrency: number | undefined): string {
  return concurrency === undefined ? "unknown" : String(concurrency);
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

function appendMetricValue(
  element: HTMLElement,
  entry: MetricEntry | undefined,
  context: DetailContext,
  kind: "performance" | "accuracy",
  metricName: "latency" | "throughput" | "accuracy"
): void {
  if (entry === undefined) {
    element.textContent = detailLabel(context.locale, kind === "performance" ? "noPerformance" : "noAccuracy");
    return;
  }
  const value = document.createElement("span");
  value.className = "model-detail-metric-value";
  value.dataset.metric = metricName;
  value.dataset.concurrency = threadKey(entry.metric.concurrency);
  value.textContent = kind === "accuracy"
    ? accuracyCellText(entry.metric, context.locale)
    : metricCellText(entry.metric, context.locale);
  value.title = entry.metric.scope ?? "";
  element.append(value);
}

function readableAccuracyMetric(metric: string, locale: Locale): string {
  const normalizedMetric = normalized(metric);
  const map = normalizedMetric.match(/^(bbox|mask|keypoints?)-all-map-50-95$/);
  if (map?.[1]) return `${map[1]} mAP@0.5:0.95`;
  if (normalizedMetric === "top-1") return "Top-1";
  if (normalizedMetric === "top-5") return "Top-5";
  return metricLabel(locale, metric);
}

function accuracyMetricName(pair: AccuracyPair, locale: Locale): string {
  return readableAccuracyMetric(pair.metric, locale);
}

function accuracyDescriptor(pair: AccuracyPair, locale: Locale): string {
  const dataset = pair.dataset ?? detailLabel(locale, "notRecorded");
  const metric = readableAccuracyMetric(pair.metric, locale);
  const unit = pair.unit === "ratio" || pair.unit === "percent" ? "%" : unitLabel(locale, pair.unit);
  return `${dataset} · ${metric} (${unit})`;
}

function accuracyHeader(pairs: AccuracyPair[], locale: Locale): { text: string; shared: boolean } {
  const descriptors = [...new Set(pairs.map((pair) => accuracyDescriptor(pair, locale)))];
  if (descriptors.length === 0) return { text: detailLabel(locale, "accuracy"), shared: true };
  return {
    text: `${detailLabel(locale, "accuracy")} · ${descriptors.join(" / ")}`,
    shared: descriptors.length === 1
  };
}

function accuracyValues(
  cellElement: HTMLElement,
  pairs: AccuracyPair[],
  stage: "float" | "quantized",
  context: DetailContext,
  showMetricLabels: boolean
): void {
  const values = pairs
    .map((pair) => ({ pair, entry: pair[stage] }))
    .filter((value): value is { pair: AccuracyPair; entry: MetricEntry } => value.entry !== undefined);
  if (values.length === 0) {
    cellElement.textContent = detailLabel(context.locale, "noAccuracy");
    return;
  }
  for (const [index, value] of values.entries()) {
    if (index > 0) cellElement.append(document.createElement("br"));
    const item = document.createElement("span");
    item.className = "model-detail-accuracy-value";
    item.dataset.metric = value.pair.metric;
    item.textContent = showMetricLabels
      ? `${accuracyMetricName(value.pair, context.locale)}: ${accuracyCellText(value.entry.metric, context.locale)}`
      : accuracyCellText(value.entry.metric, context.locale);
    item.title = [value.pair.dataset, value.pair.scope, value.pair.artifact].filter(Boolean).join(" · ");
    cellElement.append(item);
  }
}

function retentionValues(
  cellElement: HTMLElement,
  pairs: AccuracyPair[],
  context: DetailContext,
  showMetricLabels: boolean
): void {
  if (pairs.length === 0) {
    cellElement.textContent = detailLabel(context.locale, "notMeasured");
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
    if (cellElement.childNodes.length > 0) cellElement.append(document.createElement("br"));
    const item = document.createElement("span");
    item.className = "model-detail-retention-value";
    item.dataset.retentionSource = retention.source ?? "none";
    item.textContent = text;
    if (retention.source === "derived") item.title = context.locale === "zh" ? "由原始精度计算" : "Derived from source accuracies";
    cellElement.append(item);
  }
}

function appendAssetList(
  container: HTMLElement,
  assets: ModelRecord["assets"],
  context: DetailContext,
  compactLinks = false
): void {
  const list = document.createElement("ul");
  list.className = "model-detail-download-list";
  for (const asset of assets) {
    const item = document.createElement("li");
    const value = asset.url ? document.createElement("a") : document.createElement("span");
    const extension = asset.filename.split(".").pop()?.toLowerCase() || asset.format;
    value.textContent = compactLinks && asset.url
      ? `${context.locale === "zh" ? "下载" : "Download"} .${extension}`
      : asset.filename;
    value.setAttribute("aria-label", `${context.locale === "zh" ? "下载" : "Download"} ${asset.filename}`);
    value.title = asset.filename;
    if (asset.url && value instanceof HTMLAnchorElement) {
      value.href = asset.url;
      value.download = asset.filename;
      value.dataset.action = "download-model";
    }
    item.append(value);
    if (!asset.url) {
      const missing = document.createElement("small");
      missing.className = "missing-data";
      missing.textContent = ` — ${detailLabel(context.locale, "downloadNotRecorded")}`;
      item.append(missing);
    }
    list.append(item);
  }
  container.append(list);
}

function runnableAssets(variant: ModelVariant): ModelRecord["assets"] {
  return variant.assets.filter(isRunnableAsset);
}

function appendMetricDetails(
  container: HTMLElement,
  records: BenchmarkRecord[],
  variant: ModelVariant,
  context: DetailContext
): void {
  const details = document.createElement("details");
  details.className = "model-detail-row-details";
  const summary = document.createElement("summary");
  summary.textContent = detailLabel(context.locale, "details");
  details.append(summary);

  const conditions = document.createElement("div");
  conditions.className = "model-detail-conditions";
  conditions.textContent = detailLabel(context.locale, "conditions");
  for (const [index, record] of records.entries()) {
    const block = document.createElement("p");
    block.className = "model-detail-condition";
    const environment = [
      record.environment.hardware,
      record.environment.runtime ? `${detailLabel(context.locale, "runtime")}: ${record.environment.runtime}` : undefined,
      record.environment.cpu_mode ? `${detailLabel(context.locale, "cpuMode")}: ${record.environment.cpu_mode}` : undefined,
      record.environment.bpu_cores !== undefined ? `${detailLabel(context.locale, "bpuCores")}: ${record.environment.bpu_cores}` : undefined,
      inputDescription(record.input) || undefined
    ].filter((value): value is string => Boolean(value));
    block.textContent = environment.join(" · ");
    if (index === 0) block.append(" · ");
    block.append(sourceLink(record, context));
    conditions.append(block);
  }
  details.append(conditions);

  const metrics = document.createElement("table");
  metrics.className = "model-detail-metrics-table";
  const caption = document.createElement("caption");
  caption.textContent = detailLabel(context.locale, "details");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const header of [
    detailLabel(context.locale, "metric"),
    detailLabel(context.locale, "value"),
    detailLabel(context.locale, "unit"),
    detailLabel(context.locale, "scope"),
    detailLabel(context.locale, "concurrency"),
    detailLabel(context.locale, "dataset"),
    detailLabel(context.locale, "modelStage")
  ]) headRow.append(cell(header, true));
  head.append(headRow);
  metrics.append(caption, head, document.createElement("tbody"));
  const body = metrics.tBodies[0]!;
  for (const record of records) {
    for (const kind of ["performance", "accuracy"] as const) {
      for (const metric of record[kind] ?? []) {
        const row = document.createElement("tr");
        row.dataset.metric = metric.metric;
        row.append(
          cell(metric.metric),
          cell(kind === "accuracy" ? accuracyCellText(metric, context.locale) : metricCellText(metric, context.locale)),
          cell(unitLabel(context.locale, metric.unit)),
          cell(metric.scope ?? detailLabel(context.locale, "notRecorded")),
          cell(metric.concurrency === undefined ? detailLabel(context.locale, "unknownConcurrency") : String(metric.concurrency)),
          cell(metric.dataset ?? detailLabel(context.locale, "notRecorded")),
          cell(metric.model_stage ?? detailLabel(context.locale, "notRecorded"))
        );
        body.append(row);
      }
    }
  }
  if (body.childElementCount > 0) details.append(wrapper(metrics, detailLabel(context.locale, "details")));

  const assets = document.createElement("table");
  assets.className = "model-detail-assets-table";
  const assetsCaption = document.createElement("caption");
  assetsCaption.textContent = detailLabel(context.locale, "assets");
  const assetsHead = document.createElement("thead");
  const assetsRow = document.createElement("tr");
  for (const header of [detailLabel(context.locale, "downloads"), detailLabel(context.locale, "modelFormat"), detailLabel(context.locale, "checksum")]) {
    assetsRow.append(cell(header, true));
  }
  assetsHead.append(assetsRow);
  assets.append(assetsCaption, assetsHead, document.createElement("tbody"));
  const runnable = runnableAssets(variant);
  for (const asset of runnable) {
    const row = document.createElement("tr");
    const filename = document.createElement("td");
    if (asset.url) {
      const link = document.createElement("a");
      link.href = asset.url;
      link.download = asset.filename;
      link.textContent = asset.filename;
      link.title = asset.filename;
      link.setAttribute("aria-label", `${context.locale === "zh" ? "下载" : "Download"} ${asset.filename}`);
      link.dataset.action = "download-model";
      filename.append(link);
    } else {
      filename.textContent = `${asset.filename} — ${detailLabel(context.locale, "downloadNotRecorded")}`;
    }
    row.append(filename, cell(asset.format), cell(asset.sha256 ?? detailLabel(context.locale, "notRecorded")));
    assets.tBodies[0]!.append(row);
  }
  if (runnable.length === 0) appendEmptyRow(assets, detailLabel(context.locale, "noAssets"), 3);
  details.append(wrapper(assets, detailLabel(context.locale, "assets")));
  container.append(details);
}

function appendDownloadCell(cellElement: HTMLElement, assets: ModelRecord["assets"], context: DetailContext): void {
  const runnable = assets.filter((asset) => {
    const format = normalized(asset.format).replace(/^\./, "");
    const extension = normalized(asset.filename).split(".").pop() ?? "";
    const knownRunnable = new Set(["bin", "hbm", "nb", "bpu", "bc", "bmodel", "elf"]);
    return format ? knownRunnable.has(format) : knownRunnable.has(extension);
  });
  if (runnable.length === 0) {
    cellElement.textContent = detailLabel(context.locale, "noAssets");
    return;
  }
  appendAssetList(cellElement, runnable, context, true);
}

function buildVariantRow(
  variant: ModelVariant,
  records: BenchmarkRecord[],
  group: PerformanceGroup | undefined,
  threadCounts: Array<number | undefined>,
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
  specification.textContent = variant.name || variant.id;
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
    appendMetricValue(latency, thread?.latency, context, "performance", "latency");
    const throughput = document.createElement("td");
    throughput.className = "model-detail-thread-value model-detail-throughput";
    if (thread?.throughput) throughput.dataset.metric = "throughput";
    throughput.dataset.concurrency = threadKey(concurrency);
    appendMetricValue(throughput, thread?.throughput, context, "performance", "throughput");
    row.append(latency, throughput);
  }

  const pairs = pairAccuracyMetrics(records);
  const accuracyDisplay = accuracyHeader(pairs, context.locale);
  const floatCell = document.createElement("td");
  floatCell.className = "model-detail-accuracy-cell";
  accuracyValues(floatCell, pairs, "float", context, !accuracyDisplay.shared);
  row.append(floatCell);
  const quantizedCell = document.createElement("td");
  quantizedCell.className = "model-detail-accuracy-cell";
  accuracyValues(quantizedCell, pairs, "quantized", context, !accuracyDisplay.shared);
  row.append(quantizedCell);
  const retentionCell = document.createElement("td");
  retentionCell.className = "model-detail-retention-cell";
  retentionValues(retentionCell, pairs, context, !accuracyDisplay.shared);
  row.append(retentionCell);
  const downloads = document.createElement("td");
  downloads.className = "model-detail-download-cell";
  appendDownloadCell(downloads, runnableAssets(variant), context);
  row.append(downloads);

  const detailCell = document.createElement("td");
  detailCell.className = "model-detail-row-detail-cell";
  detailCell.colSpan = 1;
  appendMetricDetails(detailCell, records, variant, context);
  row.append(detailCell);
  return row;
}

function specificationsTable(
  variants: ModelVariant[],
  hardware: HardwareId,
  task: string,
  context: DetailContext
): HTMLElement {
  const selected = orderedVariants(variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task));
  const records = selected.flatMap((variant) => variantRecords(variant, hardware));
  const allGroups = groupPerformanceMetrics(records);
  const primaryGroups = allGroups.filter((group) =>
    group.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
  );
  const threadCounts = allThreadCounts(primaryGroups);
  if (threadCounts.length === 0 && selected.length > 0) threadCounts.push(undefined);
  const accuracyDisplay = accuracyHeader(pairAccuracyMetrics(records), context.locale);
  const result = table(`${detailLabel(context.locale, "performance")}${accuracyDisplay.shared && accuracyDisplay.text !== detailLabel(context.locale, "accuracy") ? ` · ${accuracyDisplay.text}` : ""}`);
  result.className = "model-detail-specifications-table";
  const head = result.tHead!;
  const firstRow = document.createElement("tr");
  const specification = cell(detailLabel(context.locale, "specification"), true);
  specification.rowSpan = 2;
  const input = cell(detailLabel(context.locale, "input"), true);
  input.rowSpan = 2;
  firstRow.append(specification, input);
  for (const concurrency of threadCounts) {
    const group = cell(`${formatThreadLabel(concurrency, context.locale)} ${context.locale === "zh" ? "延迟 / FPS" : "latency / FPS"}`, true);
    group.colSpan = 2;
    group.scope = "colgroup";
    group.dataset.concurrency = threadKey(concurrency);
    firstRow.append(group);
  }
  const accuracy = cell(accuracyDisplay.text, true);
  accuracy.colSpan = 3;
  accuracy.scope = "colgroup";
  firstRow.append(accuracy);
  const download = cell(detailLabel(context.locale, "download"), true);
  download.rowSpan = 2;
  firstRow.append(download);
  const details = cell(detailLabel(context.locale, "details"), true);
  details.rowSpan = 2;
  firstRow.append(details);
  head.append(firstRow);

  const secondRow = document.createElement("tr");
  for (const concurrency of threadCounts) {
    const latency = cell(`${formatThreadLabel(concurrency, context.locale)} ${context.locale === "zh" ? "延迟" : "latency"}`, true);
    latency.dataset.metric = "latency";
    latency.dataset.concurrency = threadKey(concurrency);
    const throughput = cell(`${formatThreadLabel(concurrency, context.locale)} FPS`, true);
    throughput.dataset.metric = "throughput";
    throughput.dataset.concurrency = threadKey(concurrency);
    secondRow.append(latency, throughput);
  }
  secondRow.append(
    cell(detailLabel(context.locale, "floatAccuracy"), true),
    cell(detailLabel(context.locale, "quantizedAccuracy"), true),
    cell(detailLabel(context.locale, "retention"), true)
  );
  head.append(secondRow);

  for (const variant of selected) {
    const variantRecordsList = variantRecords(variant, hardware);
    const groups = groupPerformanceMetrics(variantRecordsList).filter((candidate) =>
      candidate.threads.some((thread) => thread.latency !== undefined || thread.throughput !== undefined)
    );
    if (groups.length === 0) {
      result.tBodies[0]!.append(buildVariantRow(variant, variantRecordsList, undefined, threadCounts, context));
      continue;
    }
    for (const group of groups) {
      result.tBodies[0]!.append(buildVariantRow(variant, variantRecordsList, group, threadCounts, context));
    }
  }
  if (selected.length === 0) appendEmptyRow(result, detailLabel(context.locale, "noPerformance"), 2 + threadCounts.length * 2 + 5);
  return wrapper(result, detailLabel(context.locale, "specifications"));
}

function renderTaskContent(
  model: ModelRecord,
  variants: ModelVariant[],
  hardware: HardwareId,
  task: string,
  context: DetailContext
): HTMLElement {
  const section = document.createElement("section");
  section.className = "model-detail-content";
  section.dataset.hardware = hardware;
  section.dataset.task = task;
  const heading = document.createElement("h2");
  heading.className = "model-detail-specifications-heading";
  heading.textContent = detailLabel(context.locale, "specifications");
  section.append(heading);
  const selectedVariants = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const sample = selectedVariants[0];
  if (sample) {
    const sampleLink = document.createElement("a");
    sampleLink.className = "model-detail-sample-link";
    sampleLink.href = `${context.repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(sample.release_tag || context.releaseTag)}/${sample.sample_path}/README.md`;
    sampleLink.textContent = detailLabel(context.locale, "source");
    section.append(sampleLink);
  }
  section.append(specificationsTable(variants, hardware, task, context));
  return section;
}

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

export function renderModelDetails(model: ModelRecord, context: DetailContext): HTMLElement {
  const variants = availableVariants(model, context).filter((variant) => HARDWARE_IDS.includes(variant.hardware));
  const hardwareIds = HARDWARE_IDS.filter((hardware) => variants.some((variant) => variant.hardware === hardware));
  const initialHardware = preferredHardware(model, variants, context);
  let selectedHardware = hardwareIds.includes(initialHardware) ? initialHardware : hardwareIds[0] ?? "x5";
  let selectedTasks = orderedTasks(variants, selectedHardware);
  let selectedTask = preferredTask(selectedTasks, context);

  const root = document.createElement("section");
  root.className = "model-details";
  root.dataset.modelId = model.id;

  const heading = document.createElement("h1");
  heading.id = `model-details-${model.id}`;
  heading.textContent = model.name;
  root.setAttribute("aria-labelledby", heading.id);

  const close = document.createElement("button");
  close.type = "button";
  close.dataset.action = "close-details";
  close.className = "model-details-back";
  close.textContent = detailLabel(context.locale, "back");
  close.setAttribute("aria-label", detailLabel(context.locale, "back"));

  const summary = document.createElement("p");
  summary.className = "model-detail-task-summary";
  summary.textContent = `${detailLabel(context.locale, "tasks")}: ${model.tasks.map((task) => taskLabel(context.locale, task)).join(" · ")}`;

  const tabs = document.createElement("div");
  tabs.className = "model-detail-hardware-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", detailLabel(context.locale, "hardware"));

  const taskControl = document.createElement("div");
  taskControl.className = "model-detail-task-control";
  const taskLabelElement = document.createElement("label");
  taskLabelElement.htmlFor = `model-detail-task-${model.id}`;
  taskLabelElement.textContent = detailLabel(context.locale, "task");
  const taskSelect = document.createElement("select");
  taskSelect.id = taskLabelElement.htmlFor;
  taskSelect.dataset.control = "task";
  taskControl.append(taskLabelElement, taskSelect);

  const content = document.createElement("div");
  content.className = "model-detail-content-host";
  content.id = `model-detail-panel-${model.id}`;
  content.setAttribute("role", "tabpanel");

  const updateRootState = (): void => {
    root.dataset.hardware = selectedHardware;
    root.dataset.task = selectedTask;
  };

  const selectHardware = (hardware: HardwareId, notify: boolean): void => {
    if (!hardwareIds.includes(hardware)) return;
    selectedHardware = hardware;
    selectedTasks = orderedTasks(variants, selectedHardware);
    if (!selectedTasks.includes(selectedTask)) selectedTask = selectedTasks[0] ?? "";
    renderLocal();
    if (notify) context.onSelectionChange?.(selectedHardware, selectedTask);
  };

  const selectTask = (task: string, notify: boolean): void => {
    if (!selectedTasks.includes(task)) return;
    selectedTask = task;
    renderLocal();
    if (notify) context.onSelectionChange?.(selectedHardware, selectedTask);
  };

  const tabElements = new Map<HardwareId, HTMLButtonElement>();
  const renderTabs = (): void => {
    tabs.replaceChildren();
    tabElements.clear();
    for (const hardware of hardwareIds) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.role = "tab";
      tab.className = "model-detail-hardware-tab";
      tab.dataset.hardware = hardware;
      tab.id = `model-detail-${model.id}-${hardware}`;
      tab.setAttribute("aria-controls", content.id);
      tab.setAttribute("aria-selected", String(hardware === selectedHardware));
      tab.tabIndex = hardware === selectedHardware ? 0 : -1;
      tab.textContent = hardwareLabel(context.locale, hardware);
      const activate = (): void => selectHardware(hardware, true);
      tab.addEventListener("click", activate);
      tab.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const index = hardwareIds.indexOf(hardware);
        const nextIndex = event.key === "Home" ? 0
          : event.key === "End" ? hardwareIds.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : -1) + hardwareIds.length) % hardwareIds.length;
        const nextHardware = hardwareIds[nextIndex];
        if (nextHardware !== undefined) {
          selectHardware(nextHardware, true);
          tabElements.get(nextHardware)?.focus();
        }
      });
      tabElements.set(hardware, tab);
      tabs.append(tab);
    }
  };

  const renderTaskControl = (): void => {
    taskSelect.replaceChildren();
    for (const task of selectedTasks) {
      const option = document.createElement("option");
      option.value = task;
      option.textContent = taskLabel(context.locale, task);
      option.selected = task === selectedTask;
      taskSelect.append(option);
    }
    taskControl.hidden = selectedTasks.length <= 1;
  };

  const renderLocal = (): void => {
    updateRootState();
    renderTabs();
    renderTaskControl();
    content.replaceChildren(renderTaskContent(model, variants, selectedHardware, selectedTask, context));
    content.setAttribute("aria-labelledby", `model-detail-${model.id}-${selectedHardware}`);
  };
  taskSelect.addEventListener("change", () => selectTask(taskSelect.value, true));

  root.append(close, heading, summary, tabs, taskControl, content);
  renderLocal();
  return root;
}
