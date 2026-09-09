import type { HardwareId, Locale } from "../catalog/types";
import { detailLabel, hardwareLabel } from "./detail-labels";

export interface HardwareTabsOptions {
  locale: Locale;
  modelId: string;
  panelId: string;
  hardwareIds: HardwareId[];
  selectedHardware: HardwareId;
  onSelect: (hardware: HardwareId) => void;
}

export interface HardwareTabsView {
  element: HTMLDivElement;
  setSelected(hardware: HardwareId): void;
  destroy(): void;
}

/** Accessible roving-tab hardware switcher used by the model detail page. */
export function createHardwareTabs(options: HardwareTabsOptions): HardwareTabsView {
  const element = document.createElement("div");
  element.className = "model-detail-hardware-tabs";
  element.setAttribute("role", "tablist");
  element.setAttribute("aria-label", detailLabel(options.locale, "hardware"));

  let selected = options.selectedHardware;
  const tabs = new Map<HardwareId, HTMLButtonElement>();
  const cleanups: Array<() => void> = [];

  const sync = (): void => {
    for (const [hardware, tab] of tabs) {
      const active = hardware === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
  };

  const select = (hardware: HardwareId, focus = false): void => {
    if (!options.hardwareIds.includes(hardware)) return;
    selected = hardware;
    sync();
    options.onSelect(hardware);
    if (focus) tabs.get(hardware)?.focus();
  };

  for (const hardware of options.hardwareIds) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.role = "tab";
    tab.className = "model-detail-hardware-tab";
    tab.dataset.hardware = hardware;
    tab.id = `model-detail-${options.modelId}-${hardware}`;
    tab.setAttribute("aria-controls", options.panelId);
    tab.textContent = hardwareLabel(options.locale, hardware);
    const onClick = (): void => select(hardware);
    const onKeyDown = (event: KeyboardEvent): void => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = options.hardwareIds.indexOf(hardware);
      const nextIndex = event.key === "Home" ? 0
        : event.key === "End" ? options.hardwareIds.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + options.hardwareIds.length) % options.hardwareIds.length;
      const nextHardware = options.hardwareIds[nextIndex];
      if (nextHardware !== undefined) select(nextHardware, true);
    };
    tab.addEventListener("click", onClick);
    tab.addEventListener("keydown", onKeyDown);
    cleanups.push(() => tab.removeEventListener("click", onClick), () => tab.removeEventListener("keydown", onKeyDown));
    tabs.set(hardware, tab);
    element.append(tab);
  }
  sync();

  return {
    element,
    setSelected(hardware: HardwareId): void {
      if (options.hardwareIds.includes(hardware)) {
        selected = hardware;
        sync();
      }
    },
    destroy(): void {
      cleanups.forEach((cleanup) => cleanup());
      cleanups.length = 0;
    }
  };
}
