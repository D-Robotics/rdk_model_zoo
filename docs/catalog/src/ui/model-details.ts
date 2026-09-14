import { renderRepresentativeDetails } from "./representative-details";
import "./detail-layout.css";
import "./readability-tables.css";
import type { BenchmarkRecord, HardwareId, Locale, ModelRecord, ModelVariant } from "../catalog/types";
import { getHardwareIds, getModelVariants, HARDWARE_IDS, isRunnableAsset, normalizeHardware } from "../catalog/variants";
import { detailLabel, taskLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";
import { createHardwareTabs } from "./hardware-tabs";
import { createTaskSelector } from "./task-selector";
import { renderVariantBenchmarkTable } from "./variant-benchmark-table";
import { renderEvidenceDetails } from "./evidence-details";
import { groupPerformanceMetrics } from "../catalog/metric-display";
import { sourceUrl } from "./detail-utils";

export type { DetailContext } from "./detail-types";

const taskPriority = [
  "object-detection",
  "instance-segmentation",
  "pose-estimation",
  "image-classification"
];

function normalized(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLocaleLowerCase();
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
      const declaredHardware = benchmarks[0]?.environment.hardware?.trim();
      const hardware = declaredHardware ? normalizeHardware(declaredHardware) : platformHardware;
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

function actionLabel(locale: Locale, action: "share" | "copied" | "copyFailed"): string {
  if (locale === "zh") {
    return action === "share" ? "复制当前链接" : action === "copied" ? "链接已复制" : "复制失败";
  }
  return action === "share" ? "Copy link" : action === "copied" ? "Link copied" : "Copy failed";
}

async function copyText(value: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Try the DOM fallback below when clipboard permissions are unavailable.
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  let copied = false;
  try {
    copied = typeof document.execCommand === "function" && document.execCommand("copy");
  } catch {
    copied = false;
  }
  textarea.remove();
  return copied;
}

function detailUrl(modelId: string, hardware: HardwareId, task: string): string {
  if (typeof window === "undefined") return `?model=${encodeURIComponent(modelId)}&hardware=${hardware}&task=${encodeURIComponent(task)}`;
  const url = new URL(window.location.href);
  url.searchParams.set("model", modelId);
  url.searchParams.set("hardware", hardware);
  if (task) url.searchParams.set("task", task);
  else url.searchParams.delete("task");
  return url.href;
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
  section.dataset.model = model.id;
  section.dataset.hardware = hardware;
  section.dataset.task = task;

  const header = document.createElement("div");
  header.className = "model-detail-content-header";
  const heading = document.createElement("h2");
  heading.className = "model-detail-specifications-heading";
  heading.textContent = detailLabel(context.locale, "specifications");
  header.append(heading);
  const selectedVariants = variants.filter((variant) => variant.hardware === hardware
    && (variant.task || taskFromVariant(variant.id, "")) === task);
  const sample = selectedVariants[0];
  if (sample?.sample_path) {
    const sampleLink = document.createElement("a");
    sampleLink.className = "model-detail-sample-link";
    const submoduleSource = sample.benchmarks.find(record => record.source.repository_url);
    sampleLink.href = submoduleSource ? sourceUrl(submoduleSource, context.repositoryUrl)
      : `${context.repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(sample.release_tag || context.releaseTag)}/${sample.sample_path}/README.md`;
    sampleLink.textContent = detailLabel(context.locale, "source");
    header.append(sampleLink);
  }
  const representative = ["paraformer", "himloco", "siglip"].includes(model.id);
  const renderTable = representative ? (options: Parameters<typeof renderVariantBenchmarkTable>[0]) => renderRepresentativeDetails(model.id, options) : renderVariantBenchmarkTable;
  section.append(header);
  const guides: Record<string, [string, string]> = {
    yolov8: ["默认对比单线程推理、CPU 后处理和精度；其他线程可展开查看。完整测试条件见各行详情。", "Compare single-thread inference, CPU post-processing and accuracy. Expand additional threads or row details for more measurements and test conditions."],
    paraformer: ["配置与下载、阶段性能、识别精度分别展示。Python 与 C++ UCP 按原始测量记录对照。", "Configuration, pipeline performance and recognition accuracy are shown separately, with source-reported Python and C++ UCP measurements."],
    himloco: ["按运行时比较延迟分布与吞吐量。编译器估算单独列出。", "Compare latency distributions and throughput by runtime. Compiler estimates are shown separately."],
    siglip: ["先选择具体模型，再查看不同输出的性能，以及各数据集的浮点与量化精度。", "Choose a configuration, then compare performance by output and float/quantized accuracy by dataset."]
  };
  if (guides[model.id]) {
    const guide = document.createElement("p");
    guide.className = "model-detail-reading-guide";
    guide.textContent = guides[model.id]![context.locale === "zh" ? 0 : 1];
    section.append(guide);
  }
  section.append(renderTable({
    variants: orderedVariants(variants),
    hardware,
    task,
    context
  }));
  const selectedRecords = [...new Map(selectedVariants.flatMap(variant => variant.benchmarks)
    .map(record => [JSON.stringify([record.id, record.source.ref, record.source.path]), record])).values()];
  const primaryMetrics = new Set(groupPerformanceMetrics(selectedRecords).flatMap(group => group.threads
    .flatMap(thread => [thread.latency?.metric, thread.throughput?.metric]).filter(Boolean)));
  const additionalRecords = selectedRecords.map(record => ({ ...record, accuracy: [],
    performance: record.performance?.filter(metric => !primaryMetrics.has(metric))
  })).filter(record => (record.performance?.length ?? 0) > 0);
  if (additionalRecords.length > 0 && !representative) {
    const extra = document.createElement("section");
    extra.className = "model-detail-additional-metrics";
    const title = document.createElement("h3");
    title.textContent = context.locale === "zh" ? "其他性能指标" : "Additional performance metrics";
    extra.append(title, renderEvidenceDetails(additionalRecords, context));
    section.append(extra);
  }
  const relatedPlatforms = model.platforms?.filter(platform => platform.sample_path === sample?.sample_path
    && getModelVariants(platform).some(variant => variant.hardware === hardware && variant.task === task));
  const relatedAssets = (relatedPlatforms?.flatMap(platform => platform.assets) ?? model.assets)
    .filter(asset => !isRunnableAsset(asset))
    .filter(asset => !normalizeHardware(asset.filename) || normalizeHardware(asset.filename) === hardware);
  if (relatedAssets.length > 0) {
    const related = document.createElement("p");
    related.className = "model-detail-related-files";
    related.textContent = `${context.locale === "zh" ? "其他模型文件" : "Other model files"}: ${[...new Set(relatedAssets.map(asset => asset.filename))].join(", ")}. `
      + (context.locale === "zh" ? "用途与准备方法见来源文档。" : "See the source documentation for their purpose and setup.");
    section.append(related);
  }
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

  const breadcrumb = document.createElement("p");
  breadcrumb.className = "model-detail-breadcrumb";
  breadcrumb.textContent = `${context.locale === "zh" ? "模型目录" : "Model catalog"} / ${model.name}`;

  const actions = document.createElement("div");
  actions.className = "model-detail-actions";
  const share = document.createElement("button");
  share.type = "button";
  share.dataset.action = "copy-detail-link";
  share.textContent = actionLabel(context.locale, "share");
  share.setAttribute("aria-label", actionLabel(context.locale, "share"));
  const shareStatus = document.createElement("span");
  shareStatus.className = "model-detail-share-status";
  shareStatus.setAttribute("aria-live", "polite");
  const onShare = (): void => {
    void copyText(detailUrl(model.id, selectedHardware, selectedTask)).then((copied) => {
      shareStatus.textContent = actionLabel(context.locale, copied ? "copied" : "copyFailed");
    });
  };
  share.addEventListener("click", onShare);
  actions.append(share, shareStatus);

  const toolbar = document.createElement("div");
  toolbar.className = "model-detail-toolbar";
  toolbar.append(close, breadcrumb, actions);

  const summary = document.createElement("p");
  summary.className = "model-detail-task-summary";
  summary.textContent = `${detailLabel(context.locale, "tasks")}: ${model.tasks.map((task) => taskLabel(context.locale, task)).join(" · ")}`;
  const overviewCopy = document.createElement("div");
  overviewCopy.className = "model-detail-overview-copy";
  overviewCopy.append(heading, summary);
  const overview = document.createElement("div");
  overview.className = "model-detail-overview";
  overview.append(overviewCopy);

  const content = document.createElement("div");
  content.className = "model-detail-content-host";
  content.id = `model-detail-panel-${model.id}`;
  content.setAttribute("role", "tabpanel");

  const updateRootState = (): void => {
    root.dataset.hardware = selectedHardware;
    root.dataset.task = selectedTask;
    content.setAttribute("aria-labelledby", `model-detail-${model.id}-${selectedHardware}`);
  };

  const selectHardware = (hardware: HardwareId): void => {
    if (!hardwareIds.includes(hardware)) return;
    selectedHardware = hardware;
    selectedTasks = orderedTasks(variants, selectedHardware);
    if (!selectedTasks.includes(selectedTask)) selectedTask = selectedTasks[0] ?? "";
    renderLocal();
    context.onSelectionChange?.(selectedHardware, selectedTask);
  };

  const selectTask = (task: string): void => {
    if (!selectedTasks.includes(task)) return;
    selectedTask = task;
    renderLocal();
    context.onSelectionChange?.(selectedHardware, selectedTask);
  };

  const tabs = createHardwareTabs({
    locale: context.locale,
    modelId: model.id,
    panelId: content.id,
    hardwareIds,
    selectedHardware,
    onSelect: selectHardware
  });
  const taskControl = createTaskSelector({
    locale: context.locale,
    modelId: model.id,
    tasks: selectedTasks,
    selectedTask,
    onSelect: selectTask
  });
  const selection = document.createElement("div");
  selection.className = "model-detail-selection";
  selection.append(tabs.element, taskControl.element);

  const renderLocal = (): void => {
    updateRootState();
    tabs.setSelected(selectedHardware);
    taskControl.setTasks(selectedTasks, selectedTask);
    content.replaceChildren(renderTaskContent(model, variants, selectedHardware, selectedTask, context));
  };

  root.append(toolbar, overview);
  if (variants.length > 0) {
    root.append(selection, content);
    renderLocal();
  } else {
    const empty = document.createElement("p");
    empty.textContent = context.locale === "zh"
      ? "尚未发布可对应到具体开发板的模型配置。"
      : "No board-specific model configuration has been published.";
    root.append(empty);
  }

  // Evidence can describe a toolchain, an unspecified board, or a shared
  // measurement. Keep it accessible without assigning it to an inferred board.
  const recordKey = (record: BenchmarkRecord): string =>
    JSON.stringify([record.id, record.source.ref, record.source.path]);
  const assigned = new Set(variants.flatMap(variant => variant.benchmarks).map(recordKey));
  const unassigned = [...new Map(model.benchmarks.filter(record => !assigned.has(recordKey(record)))
    .map(record => [recordKey(record), record])).values()];
  if (unassigned.length > 0) {
    // Families such as SigLIP publish many shared distributions at once. The
    // list stays complete and in the DOM, but starts collapsed with an
    // informative summary so it does not push the assigned configurations off
    // the page. No record is ever attributed to a board it does not name.
    const evidence = document.createElement("details");
    evidence.className = "model-detail-unassigned-evidence";
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.className = "model-detail-unassigned-title";
    title.textContent = detailLabel(context.locale, "unassignedEvidence");
    const count = document.createElement("span");
    count.className = "model-detail-unassigned-count";
    count.textContent = context.locale === "zh"
      ? `共 ${unassigned.length} 条记录 · ${detailLabel(context.locale, "unassignedEvidenceCount")}`
      : `${unassigned.length} records · ${detailLabel(context.locale, "unassignedEvidenceCount")}`;
    summary.append(title, count);
    const explanation = document.createElement("p");
    explanation.className = "model-detail-unassigned-explanation";
    explanation.textContent = context.locale === "zh"
      ? "以下记录按原始测试条件展示，尚未关联到上方的开发板配置。"
      : "These records retain their original test conditions and are not assigned to the board configurations above.";
    evidence.append(summary, explanation);
    for (const record of unassigned) {
      const item = document.createElement("details");
      item.className = "model-detail-unassigned-record";
      item.dataset.unassignedRecord = record.id;
      const recordSummary = document.createElement("summary");
      const name = document.createElement("span");
      name.className = "model-detail-unassigned-record-name";
      name.textContent = record.display_name;
      const conditions = document.createElement("span");
      conditions.className = "model-detail-unassigned-record-conditions";
      conditions.textContent = [record.environment.hardware, record.environment.runtime,
        record.source.section].filter(Boolean).join(" · ");
      recordSummary.append(name, conditions);
      recordSummary.title = detailLabel(context.locale, "unassignedRecordSummary");
      item.append(recordSummary, renderEvidenceDetails([record], context));
      evidence.append(item);
    }
    root.append(evidence);
  }
  return root;
}
