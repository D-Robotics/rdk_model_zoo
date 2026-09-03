import type { Catalog, Locale } from "../catalog/types";
import { t } from "../i18n/translations";

export interface SummaryStats {
  models: number;
  tasks: number;
  assets: number;
  benchmarks: number;
}

export function catalogSummaryStats(catalog: Catalog): SummaryStats {
  const totalAssets = catalog.models.reduce((total, model) => total + model.assets.length, 0);
  const stats = {
    models: catalog.models.length,
    tasks: new Set(catalog.models.flatMap((model) => model.tasks)).size,
    assets: catalog.models.flatMap((model) => model.assets).filter((asset) => asset.url).length,
    benchmarks: catalog.models.reduce((total, model) => total + model.benchmarks.length, 0)
  };

  const declaredModels = catalog.summary.sample_count;
  const declaredTotalAssets = catalog.summary.asset_count;
  const declaredAssets = catalog.summary.downloadable_asset_count;
  if (declaredModels !== undefined && declaredModels !== stats.models) {
    throw new Error(`Catalog summary declares ${declaredModels} models but contains ${stats.models}.`);
  }
  if (declaredAssets !== undefined && declaredAssets !== stats.assets) {
    throw new Error(`Catalog summary declares ${declaredAssets} downloadable assets but contains ${stats.assets}.`);
  }
  if (declaredTotalAssets !== undefined && declaredTotalAssets !== totalAssets) {
    throw new Error(`Catalog summary declares ${declaredTotalAssets} assets but contains ${totalAssets}.`);
  }
  return stats;
}

function summaryItem(label: string, value: string, testId: string): HTMLDivElement {
  const item = document.createElement("div");
  item.className = "summary-item";
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.dataset.testid = testId;
  description.textContent = value;
  item.append(term, description);
  return item;
}

export function renderSummary(catalog: Catalog, locale: Locale): HTMLElement {
  const stats = catalogSummaryStats(catalog);
  const section = document.createElement("section");
  section.className = "catalog-summary";
  section.setAttribute("aria-labelledby", "catalog-summary-heading");

  const heading = document.createElement("h2");
  heading.id = "catalog-summary-heading";
  heading.textContent = t(locale, "summary.inventory");

  const release = document.createElement("p");
  release.className = "release-summary";
  release.textContent = `${t(locale, "release.current")}: `;
  const tag = document.createElement("strong");
  tag.dataset.testid = "release-tag";
  tag.textContent = catalog.release.tag;
  release.append(tag);

  const list = document.createElement("dl");
  list.className = "summary-grid";
  list.append(
    summaryItem(t(locale, "summary.models"), String(stats.models), "model-count"),
    summaryItem(t(locale, "summary.tasks"), String(stats.tasks), "task-count"),
    summaryItem(t(locale, "summary.assets"), String(stats.assets), "asset-count"),
    summaryItem(t(locale, "summary.benchmarks"), String(stats.benchmarks), "benchmark-count")
  );

  section.append(heading, release, list);
  return section;
}
