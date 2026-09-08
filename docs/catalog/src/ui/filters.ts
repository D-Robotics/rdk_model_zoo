import type { Catalog, Locale } from "../catalog/types";
import type { CatalogQuery } from "../catalog/query";
import { t, taskTranslationKey } from "../i18n/translations";
import { HARDWARE_IDS, getModelVariants } from "../catalog/variants";
import { groupTasks } from "../catalog/task-groups";

export const DEFAULT_QUERY: CatalogQuery = {
  text: "", platform: "", tasks: [], formats: [], precisions: [], benchmark: "all", sort: "name"
};
export interface FilterPanel {
  element: HTMLElement; hardware: HTMLElement; toolbar: HTMLElement; chips: HTMLElement;
  reset(): void; setQuery(query: CatalogQuery): void; destroy(): void;
}
const copy = (query: CatalogQuery): CatalogQuery => ({ ...query, tasks: [...query.tasks], formats: [...query.formats], precisions: [...query.precisions], sort: "name" });

export function createFilters(catalog: Catalog, locale: Locale, initialQuery: CatalogQuery,
  onChange: (query: CatalogQuery) => void): FilterPanel {
  const zh = locale === "zh";
  let current = copy(initialQuery);
  let draft: CatalogQuery | null = null;
  let restoreBackground: (() => void) | undefined;
  const cleanups: Array<() => void> = [];
  const listen = (node: HTMLElement, event: string, handler: EventListener): void => {
    node.addEventListener(event, handler); cleanups.push(() => node.removeEventListener(event, handler));
  };
  const button = (label: string, action: () => void): HTMLButtonElement => {
    const node = document.createElement("button"); node.type = "button"; node.textContent = label;
    listen(node, "click", action); return node;
  };
  const element = document.createElement("aside"); element.className = "catalog-filters";
  element.id = "catalog-filter-panel";
  const heading = document.createElement("h2"); heading.id = "catalog-filter-heading";
  heading.textContent = t(locale, "filter.heading"); element.setAttribute("aria-labelledby", heading.id);
  const hardware = document.createElement("nav"); hardware.className = "hardware-filter";
  hardware.setAttribute("aria-label", t(locale, "filter.platformLabel"));
  const hardwareButtons = ["", ...HARDWARE_IDS].map(id => {
    const node = button(id ? id.toUpperCase() : t(locale, "filter.all"), () => change({ platform: id }));
    node.dataset.platform = id; hardware.append(node); return node;
  });
  const toolbar = document.createElement("div"); toolbar.className = "catalog-toolbar";
  const searchWrap = document.createElement("div"); searchWrap.className = "filter-control filter-search";
  const searchLabel = document.createElement("label"); searchLabel.htmlFor = "catalog-search";
  searchLabel.textContent = t(locale, "filter.searchLabel");
  const search = document.createElement("input"); search.id = "catalog-search"; search.type = "search";
  search.placeholder = t(locale, "filter.searchPlaceholder"); searchWrap.append(searchLabel, search);
  listen(search, "input", () => change({ text: search.value }));
  const open = button(zh ? "筛选" : "Filters", () => {
    restoreBackground?.();
    const background = [...document.querySelectorAll<HTMLElement>(".catalog-results, .hardware-filter, .catalog-summary, .catalog-hero, .directory-heading, .preference-controls, .site-header, .site-footer")];
    const previous = background.map(node => node.inert);
    const overflow = document.body.style.overflow;
    background.forEach(node => { node.inert = true; });
    document.body.style.overflow = "hidden";
    restoreBackground = () => {
      background.forEach((node, index) => { node.inert = previous[index] ?? false; });
      document.body.style.overflow = overflow;
      restoreBackground = undefined;
    };
    draft = copy(current); sync(); element.classList.add("is-open");
    element.setAttribute("role", "dialog"); element.setAttribute("aria-modal", "true");
    open.setAttribute("aria-expanded", "true"); cancel.focus();
  });
  open.className = "mobile-filter-toggle"; open.setAttribute("aria-controls", element.id);
  open.setAttribute("aria-expanded", "false"); toolbar.append(searchWrap, open);
  const chips = document.createElement("div"); chips.className = "active-filters";
  chips.setAttribute("aria-label", zh ? "已选条件" : "Active filters");
  const closeDraft = (): void => {
    restoreBackground?.();
    draft = null; element.classList.remove("is-open"); element.removeAttribute("role");
    element.removeAttribute("aria-modal"); open.setAttribute("aria-expanded", "false"); sync(); open.focus();
  };
  const cancel = button(zh ? "取消" : "Cancel", closeDraft); cancel.className = "mobile-filter-cancel";
  const onResize = (): void => { if (draft && window.innerWidth >= 900) closeDraft(); };
  window.addEventListener("resize", onResize);
  cleanups.push(() => window.removeEventListener("resize", onResize));
  const apply = button(zh ? "应用筛选" : "Apply filters", () => {
    if (draft) current = copy(draft); closeDraft(); emit();
  }); apply.className = "mobile-filter-apply";
  listen(element, "keydown", event => {
    if (!draft) return;
    const key = event as KeyboardEvent;
    if (key.key === "Escape") { key.preventDefault(); closeDraft(); }
    if (key.key === "Tab") {
      const controls = [...element.querySelectorAll<HTMLElement>('button, input, select, summary')]
        .filter(node => node.getClientRects().length > 0);
      const first = controls[0], last = controls.at(-1);
      if (key.shiftKey && document.activeElement === first) { key.preventDefault(); last?.focus(); }
      else if (!key.shiftKey && document.activeElement === last) { key.preventDefault(); first?.focus(); }
    }
  });
  const taskInputs = new Map<string, HTMLInputElement>();
  const groups: Array<{ input: HTMLInputElement; ids: string[] }> = [];
  const tasks = [...new Set(catalog.models.flatMap(model => model.tasks))];
  const taskList = document.createElement("div"); taskList.className = "task-filter-list";
  for (const group of groupTasks(tasks, locale)) {
    const fieldset = document.createElement("fieldset"); const legend = document.createElement("legend");
    const label = document.createElement("label"); const parent = document.createElement("input");
    parent.type = "checkbox"; parent.dataset.taskGroup = group.id; label.append(parent, group.label); legend.append(label);
    groups.push({ input: parent, ids: group.tasks });
    listen(parent, "change", () => {
      const selected = (draft ?? current).tasks.filter(id => !group.tasks.includes(id));
      change({ tasks: parent.checked ? [...selected, ...group.tasks] : selected });
    }); fieldset.append(legend);
    for (const id of group.tasks) {
      const childLabel = document.createElement("label"); const input = document.createElement("input");
      input.type = "checkbox"; input.value = id; input.name = "catalog-task";
      childLabel.append(input, t(locale, taskTranslationKey(id))); taskInputs.set(id, input);
      listen(input, "change", () => {
        const selected = (draft ?? current).tasks.filter(task => task !== id);
        change({ tasks: input.checked ? [...selected, id] : selected });
      }); fieldset.append(childLabel);
    } taskList.append(fieldset);
  }
  const more = document.createElement("details"); more.className = "more-filters";
  const moreLabel = document.createElement("summary"); moreLabel.textContent = zh ? "更多筛选" : "More filters"; more.append(moreLabel);
  const selects: Array<{ key: "formats" | "precisions" | "benchmark"; select: HTMLSelectElement }> = [];
  const addSelect = (key: "formats" | "precisions" | "benchmark", title: string, values: string[]): void => {
    values = [...new Set(values)];
    if (values.length <= 1 && key !== "benchmark") return;
    const label = document.createElement("label"); label.className = "filter-control"; label.textContent = title;
    const select = document.createElement("select"); select.id = `catalog-${key}`;
    for (const value of ["", ...values]) {
      const option = document.createElement("option"); option.value = value;
      option.textContent = !value ? t(locale, "filter.all") : key === "benchmark"
        ? t(locale, value === "performance" ? "filter.performance" : value === "accuracy" ? "filter.accuracy" : "filter.none") : value;
      select.append(option);
    }
    listen(select, "change", () => change(key === "benchmark"
      ? { benchmark: (select.value || "all") as CatalogQuery["benchmark"] }
      : { [key]: select.value ? [select.value] : [] }));
    selects.push({ key, select }); label.append(select); more.append(label);
  };
  const variants = catalog.models.flatMap(getModelVariants);
  addSelect("formats", t(locale, "filter.formatsLabel"), variants.flatMap(v => v.assets.map(a => a.format)));
  addSelect("precisions", t(locale, "filter.precisionsLabel"), variants.flatMap(v => v.benchmarks.flatMap(b => b.precision ? [b.precision] : [])));
  addSelect("benchmark", t(locale, "filter.benchmarkLabel"), ["performance", "accuracy", "none"]);
  const reset = button(t(locale, "filter.reset"), () => {
    if (draft) { draft = copy(DEFAULT_QUERY); sync(); } else { current = copy(DEFAULT_QUERY); emit(); }
  }); reset.dataset.action = "reset-filters";
  const actions = document.createElement("div"); actions.className = "filter-actions"; actions.append(reset, apply);
  element.append(heading, cancel, taskList, more, actions);
  function change(patch: Partial<CatalogQuery>): void {
    if (draft) { draft = { ...draft, ...patch }; sync(); } else { current = { ...current, ...patch }; emit(); }
  }
  function emit(): void { sync(); onChange(copy(current)); }
  function sync(): void {
    const query = draft ?? current; search.value = current.text;
    for (const node of hardwareButtons) node.setAttribute("aria-pressed", String(node.dataset.platform === current.platform));
    for (const [id, input] of taskInputs) input.checked = query.tasks.includes(id);
    for (const { input, ids } of groups) {
      input.checked = ids.every(id => query.tasks.includes(id));
      input.indeterminate = !input.checked && ids.some(id => query.tasks.includes(id));
    }
    for (const { key, select } of selects) select.value = key === "benchmark" ? (query.benchmark === "all" ? "" : query.benchmark) : query[key][0] ?? "";
    chips.replaceChildren();
    const chip = (label: string, patch: Partial<CatalogQuery>): void => {
      const node = document.createElement("button"); node.type = "button"; node.textContent = `${label} ×`;
      node.setAttribute("aria-label", `${zh ? "移除" : "Remove"} ${label}`);
      node.addEventListener("click", () => change(patch)); chips.append(node);
    };
    if (current.text) chip(current.text, { text: "" });
    if (current.platform) chip(current.platform.toUpperCase(), { platform: "" });
    for (const id of current.tasks) chip(t(locale, taskTranslationKey(id)), { tasks: current.tasks.filter(item => item !== id) });
    for (const key of ["formats", "precisions"] as const) for (const value of current[key]) chip(value, { [key]: current[key].filter(item => item !== value) });
    if (current.benchmark !== "all") chip(t(locale, current.benchmark === "performance" ? "filter.performance" : current.benchmark === "accuracy" ? "filter.accuracy" : "filter.none"), { benchmark: "all" });
    chips.hidden = !chips.childElementCount;
  }
  sync();
  return { element, hardware, toolbar, chips, reset: () => { current = copy(DEFAULT_QUERY); emit(); },
    setQuery(query) {
      const wasOpen = draft !== null;
      restoreBackground?.(); current = copy(query); draft = null;
      element.classList.remove("is-open"); element.removeAttribute("role"); element.removeAttribute("aria-modal");
      open.setAttribute("aria-expanded", "false"); sync();
      if (wasOpen) open.focus();
    },
    destroy() {
      if (draft) closeDraft();
      cleanups.forEach(cleanup => cleanup()); chips.replaceChildren();
    }
  };
}
