import type { Catalog, Locale } from "../catalog/types";
import type { CatalogQuery } from "../catalog/query";
import { t, taskTranslationKey } from "../i18n/translations";

export const DEFAULT_QUERY: CatalogQuery = {
  text: "",
  tasks: [],
  formats: [],
  precisions: [],
  benchmark: "all",
  sort: "name"
};

export interface FilterPanel {
  element: HTMLElement;
  reset(): void;
  destroy(): void;
}

type ChangeHandler = (query: CatalogQuery) => void;

function uniqueSorted(values: string[], locale: Locale): string[] {
  const collator = new Intl.Collator(locale === "zh" ? "zh-CN" : "en-US");
  return [...new Set(values)].sort(collator.compare);
}

function labeledSelect(id: string, labelText: string): { wrapper: HTMLDivElement; select: HTMLSelectElement } {
  const wrapper = document.createElement("div");
  wrapper.className = "filter-control";
  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = labelText;
  const select = document.createElement("select");
  select.id = id;
  wrapper.append(label, select);
  return { wrapper, select };
}

function addOption(select: HTMLSelectElement, value: string, label: string): void {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

export function createFilters(
  catalog: Catalog,
  locale: Locale,
  initialQuery: CatalogQuery,
  onChange: ChangeHandler
): FilterPanel {
  const form = document.createElement("section");
  form.className = "catalog-filters";
  form.setAttribute("aria-labelledby", "catalog-filter-heading");
  const heading = document.createElement("h2");
  heading.id = "catalog-filter-heading";
  heading.textContent = t(locale, "filter.heading");

  const searchWrapper = document.createElement("div");
  searchWrapper.className = "filter-control filter-search";
  const searchLabel = document.createElement("label");
  searchLabel.htmlFor = "catalog-search";
  searchLabel.textContent = t(locale, "filter.searchLabel");
  const search = document.createElement("input");
  search.id = "catalog-search";
  search.type = "search";
  search.placeholder = t(locale, "filter.searchPlaceholder");
  search.value = initialQuery.text;
  searchWrapper.append(searchLabel, search);

  const taskControl = labeledSelect("catalog-task", t(locale, "filter.tasksLabel"));
  addOption(taskControl.select, "", t(locale, "filter.all"));
  const tasks = uniqueSorted(catalog.models.flatMap((model) => model.tasks), locale);
  for (const task of tasks) {
    addOption(taskControl.select, task, t(locale, taskTranslationKey(task)));
  }

  const formatControl = labeledSelect("catalog-format", t(locale, "filter.formatsLabel"));
  addOption(formatControl.select, "", t(locale, "filter.all"));
  const formats = uniqueSorted(catalog.models.flatMap((model) => [
    ...model.assets.map((asset) => asset.format),
    ...model.benchmarks.flatMap((benchmark) => benchmark.model_format ? [benchmark.model_format] : [])
  ]), locale);
  for (const format of formats) addOption(formatControl.select, format, format);

  const precisionControl = labeledSelect("catalog-precision", t(locale, "filter.precisionsLabel"));
  addOption(precisionControl.select, "", t(locale, "filter.all"));
  const precisions = uniqueSorted(catalog.models.flatMap((model) =>
    model.benchmarks.flatMap((benchmark) => benchmark.precision ? [benchmark.precision] : [])
  ), locale);
  for (const precision of precisions) addOption(precisionControl.select, precision, precision);

  const benchmarkControl = labeledSelect("catalog-benchmark", t(locale, "filter.benchmarkLabel"));
  addOption(benchmarkControl.select, "all", t(locale, "filter.all"));
  addOption(benchmarkControl.select, "performance", t(locale, "filter.performance"));
  addOption(benchmarkControl.select, "accuracy", t(locale, "filter.accuracy"));
  addOption(benchmarkControl.select, "none", t(locale, "filter.none"));

  const sortControl = labeledSelect("catalog-sort", t(locale, "filter.sortLabel"));
  addOption(sortControl.select, "name", t(locale, "filter.sortName"));
  addOption(sortControl.select, "latency", t(locale, "filter.sortLatency"));
  addOption(sortControl.select, "fps", t(locale, "filter.sortFps"));
  addOption(sortControl.select, "accuracy", t(locale, "filter.sortAccuracy"));

  taskControl.select.value = initialQuery.tasks[0] ?? "";
  formatControl.select.value = initialQuery.formats[0] ?? "";
  precisionControl.select.value = initialQuery.precisions[0] ?? "";
  benchmarkControl.select.value = initialQuery.benchmark;
  sortControl.select.value = initialQuery.sort;

  const reset = document.createElement("button");
  reset.type = "button";
  reset.dataset.action = "reset-filters";
  reset.textContent = t(locale, "filter.reset");

  const controls = document.createElement("div");
  controls.className = "filter-grid";
  controls.append(
    searchWrapper,
    taskControl.wrapper,
    formatControl.wrapper,
    precisionControl.wrapper,
    benchmarkControl.wrapper,
    sortControl.wrapper,
    reset
  );
  form.append(heading, controls);

  const readQuery = (): CatalogQuery => ({
    text: search.value,
    tasks: taskControl.select.value ? [taskControl.select.value] : [],
    formats: formatControl.select.value ? [formatControl.select.value] : [],
    precisions: precisionControl.select.value ? [precisionControl.select.value] : [],
    benchmark: benchmarkControl.select.value as CatalogQuery["benchmark"],
    sort: sortControl.select.value as CatalogQuery["sort"]
  });
  const emit = (): void => onChange(readQuery());
  const resetControls = (): void => {
    search.value = "";
    taskControl.select.value = "";
    formatControl.select.value = "";
    precisionControl.select.value = "";
    benchmarkControl.select.value = "all";
    sortControl.select.value = "name";
    onChange({ ...DEFAULT_QUERY });
  };

  search.addEventListener("input", emit);
  for (const select of [taskControl.select, formatControl.select, precisionControl.select, benchmarkControl.select, sortControl.select]) {
    select.addEventListener("change", emit);
  }
  reset.addEventListener("click", resetControls);

  return {
    element: form,
    reset: resetControls,
    destroy() {
      search.removeEventListener("input", emit);
      for (const select of [taskControl.select, formatControl.select, precisionControl.select, benchmarkControl.select, sortControl.select]) {
        select.removeEventListener("change", emit);
      }
      reset.removeEventListener("click", resetControls);
    }
  };
}
