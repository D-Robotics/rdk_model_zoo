import "./detail-layout.css";
import type { BenchmarkRecord, HardwareId, Locale, ModelRecord, ModelVariant } from "../catalog/types";
import { getHardwareIds, getModelVariants, HARDWARE_IDS, normalizeHardware } from "../catalog/variants";
import { detailLabel, taskLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";
import { createHardwareTabs } from "./hardware-tabs";
import { createTaskSelector } from "./task-selector";
import { renderVariantBenchmarkTable } from "./variant-benchmark-table";

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
    sampleLink.href = `${context.repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(sample.release_tag || context.releaseTag)}/${sample.sample_path}/README.md`;
    sampleLink.textContent = detailLabel(context.locale, "source");
    header.append(sampleLink);
  }
  section.append(header, renderVariantBenchmarkTable({
    variants: orderedVariants(variants),
    hardware,
    task,
    context
  }));
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

  root.append(toolbar, overview, selection, content);
  renderLocal();
  return root;
}
