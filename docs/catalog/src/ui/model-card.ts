import type { HardwareId, Locale, ModelRecord } from "../catalog/types";
import { buildModelCardViewModel } from "../catalog/card-view-model";
import { t, taskTranslationKey } from "../i18n/translations";
import "./card-layout.css";

export interface RenderedModelCard { element: HTMLElement; destroy(): void; }

function hardwareLabel(hardware: HardwareId): string {
  return hardware.toUpperCase();
}

export function renderModelCard(
  model: ModelRecord, platform: string, locale: Locale,
  onSelect: (modelId: string, hardware?: HardwareId) => void
): RenderedModelCard {
  const viewModel = buildModelCardViewModel(model, platform);
  const article = document.createElement("article");
  article.className = "model-card";
  article.dataset.modelId = viewModel.modelId;
  article.dataset.platform = viewModel.platform || "all";
  article.dataset.variantCount = String(viewModel.variantCount);

  const visual = document.createElement("div");
  visual.className = "model-card-visual";
  visual.dataset.visualKind = viewModel.visualKind;
  visual.setAttribute("aria-hidden", "true");
  visual.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));

  const body = document.createElement("div");
  body.className = "model-card-body";

  const heading = document.createElement("div");
  heading.className = "model-card-heading";
  const kicker = document.createElement("p");
  kicker.className = "model-card-kicker";
  kicker.textContent = locale === "zh" ? "模型家族" : "Model family";
  const title = document.createElement("h3");
  const titleLink = document.createElement("a");
  const detailUrl = new URL(window.location.href);
  detailUrl.searchParams.set("model", viewModel.modelId);
  detailUrl.searchParams.delete("hardware");
  detailUrl.searchParams.delete("task");
  if (detailUrl.searchParams.get("platform")) {
    detailUrl.searchParams.set("hardware", detailUrl.searchParams.get("platform")!);
  }
  titleLink.href = detailUrl.href;
  titleLink.textContent = viewModel.name;
  title.append(titleLink);
  heading.append(kicker, title);

  const tasks = document.createElement("ul");
  tasks.className = "task-badges";
  tasks.setAttribute("aria-label", t(locale, "model.tasks"));
  for (const task of viewModel.tasks) {
    const item = document.createElement("li");
    item.textContent = t(locale, taskTranslationKey(task));
    tasks.append(item);
  }

  const specificationsGroup = document.createElement("div");
  specificationsGroup.className = "model-card-specifications";
  const specificationsLabel = document.createElement("span");
  specificationsLabel.className = "model-card-section-label";
  specificationsLabel.textContent = locale === "zh" ? "规格" : "Specifications";
  const specifications = document.createElement("p");
  specifications.className = "card-specifications";
  if (viewModel.specifications.length === 0) {
    specifications.dataset.empty = "true";
    specifications.textContent = t(locale, "missing.notPublished");
  } else {
    const visible = viewModel.specifications.slice(0, 5).join(" · ");
    const remaining = viewModel.specifications.length - 5;
    specifications.textContent = visible
      + (remaining > 0
        ? (locale === "zh" ? ` 等 ${viewModel.specifications.length} 种规格` : ` · ${viewModel.specifications.length} specifications`)
        : "");
  }
  specificationsGroup.append(specificationsLabel, specifications);

  const hardwareGroup = document.createElement("div");
  hardwareGroup.className = "model-card-hardware";
  const hardwareLabelElement = document.createElement("span");
  hardwareLabelElement.className = "model-card-section-label";
  hardwareLabelElement.textContent = t(locale, "model.platform");
  const hardwareLabels = document.createElement("div");
  hardwareLabels.className = "hardware-badges";
  hardwareLabels.setAttribute("role", "group");
  hardwareLabels.setAttribute("aria-label", t(locale, "model.platform"));
  hardwareGroup.append(hardwareLabelElement, hardwareLabels);

  const cleanups: Array<() => void> = [];
  const followTitle = (event: MouseEvent): void => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onSelect(viewModel.modelId);
  };
  const followCard = (event: MouseEvent): void => {
    if (!(event.target instanceof Element) || event.target.closest("a, button, input, select, details")) return;
    titleLink.focus({ preventScroll: true });
    onSelect(viewModel.modelId);
  };
  titleLink.addEventListener("click", followTitle);
  article.addEventListener("click", followCard);
  cleanups.push(
    () => titleLink.removeEventListener("click", followTitle),
    () => article.removeEventListener("click", followCard)
  );

  for (const hardware of viewModel.hardware) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.hardware = hardware;
    button.textContent = hardwareLabel(hardware);
    const select = (): void => onSelect(viewModel.modelId, hardware);
    button.addEventListener("click", select);
    cleanups.push(() => button.removeEventListener("click", select));
    hardwareLabels.append(button);
  }

  const details = document.createElement("button");
  details.type = "button";
  details.dataset.action = "open-details";
  details.textContent = locale === "zh" ? "查看模型 →" : "View model →";
  details.setAttribute("aria-label", t(locale, "model.openDetails", { name: viewModel.name }));
  const select = (): void => onSelect(viewModel.modelId);
  details.addEventListener("click", select);
  cleanups.push(() => details.removeEventListener("click", select));
  const actions = document.createElement("div");
  actions.className = "model-actions";
  actions.append(details);

  body.append(heading, tasks, specificationsGroup, hardwareGroup, actions);
  article.append(visual, body);
  return {
    element: article,
    destroy() {
      cleanups.forEach((cleanup) => cleanup());
    }
  };
}
