import type { HardwareId, Locale, ModelRecord } from "../catalog/types";
import { getHardwareIds, getModelVariants } from "../catalog/variants";
import { t, taskTranslationKey } from "../i18n/translations";

export interface RenderedModelCard { element: HTMLElement; destroy(): void; }

export function renderModelCard(
  model: ModelRecord, _platform: string, locale: Locale,
  onSelect: (modelId: string, hardware?: HardwareId) => void
): RenderedModelCard {
  const article = document.createElement("article");
  article.className = "model-card";
  article.dataset.modelId = model.id;
  const title = document.createElement("h3");
  const titleLink = document.createElement("a");
  const detailUrl = new URL(window.location.href);
  detailUrl.searchParams.set("model", model.id);
  detailUrl.searchParams.delete("hardware");
  detailUrl.searchParams.delete("task");
  if (detailUrl.searchParams.get("platform")) detailUrl.searchParams.set("hardware", detailUrl.searchParams.get("platform")!);
  titleLink.href = detailUrl.href;
  titleLink.textContent = model.name;
  title.append(titleLink);
  const variants = getModelVariants(model);
  const tasks = document.createElement("ul");
  tasks.className = "task-badges";
  tasks.setAttribute("aria-label", t(locale, "model.tasks"));
  const taskIds = [...new Set(variants.length ? variants.map((variant) => variant.task) : model.tasks)];
  for (const task of taskIds) {
    const item = document.createElement("li");
    item.textContent = t(locale, taskTranslationKey(task));
    tasks.append(item);
  }
  const hardwareLabels = document.createElement("div");
  hardwareLabels.className = "hardware-badges";
  hardwareLabels.setAttribute("role", "group");
  hardwareLabels.setAttribute("aria-label", t(locale, "model.platform"));
  const cleanups: Array<() => void> = [];
  const followTitle = (event: MouseEvent): void => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); onSelect(model.id);
  };
  const followCard = (event: MouseEvent): void => {
    if (!(event.target instanceof Element) || event.target.closest("a, button, input, select, details")) return;
    titleLink.focus({ preventScroll: true });
    onSelect(model.id);
  };
  titleLink.addEventListener("click", followTitle);
  article.addEventListener("click", followCard);
  cleanups.push(() => titleLink.removeEventListener("click", followTitle), () => article.removeEventListener("click", followCard));
  for (const hardware of getHardwareIds(model)) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.hardware = hardware;
    button.textContent = hardware.toUpperCase();
    const select = (): void => onSelect(model.id, hardware);
    button.addEventListener("click", select);
    cleanups.push(() => button.removeEventListener("click", select));
    hardwareLabels.append(button);
  }
  const specifications = document.createElement("p");
  specifications.className = "card-specifications";
  const names = [...new Set(variants.map((variant) => {
    if (/^yolov?\d+$/i.test(model.id)) return /^yolov?\d+[a-z]*/i.exec(variant.name)?.[0] ?? variant.name;
    return variant.name.replace(/\s+(?:on\s+)?RDK\s+.*$/i, "");
  }))];
  if (/^yolov?\d+$/i.test(model.id)) {
    const sizeOrder = ["n", "s", "m", "l", "x", "b", "c", "e"];
    const rank = (name: string): number => { const size = /\d+([a-z])/i.exec(name)?.[1]?.toLowerCase(); return size ? sizeOrder.indexOf(size) : 99; };
    names.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  }
  specifications.textContent = names.slice(0, 5).join(" · ")
    + (names.length > 5 ? (locale === "zh" ? ` 等 ${names.length} 种规格` : ` · ${names.length} specifications`) : "");
  const details = document.createElement("button");
  details.type = "button";
  details.dataset.action = "open-details";
  details.textContent = locale === "zh" ? "查看模型 →" : "View model →";
  details.setAttribute("aria-label", t(locale, "model.openDetails", { name: model.name }));
  const select = (): void => onSelect(model.id);
  details.addEventListener("click", select);
  cleanups.push(() => details.removeEventListener("click", select));
  const actions = document.createElement("div"); actions.className = "model-actions"; actions.append(details);
  article.append(title, tasks, hardwareLabels, specifications, actions);
  return { element: article, destroy() { cleanups.forEach((cleanup) => cleanup()); } };
}
