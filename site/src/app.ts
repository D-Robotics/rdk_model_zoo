import { queryModels, type CatalogQuery } from "./catalog/query";
import type { Catalog, Locale } from "./catalog/types";
import { t } from "./i18n/translations";
import { createFilters, DEFAULT_QUERY } from "./ui/filters";
import { renderModelCard, type RenderedModelCard } from "./ui/model-card";
import { renderSummary } from "./ui/summary";

export interface AppOptions {
  locale: Locale;
  onSelectModel?: (modelId: string) => void;
}

export interface CatalogApp {
  destroy(): void;
  state(): Readonly<AppState>;
}

interface AppState {
  locale: Locale;
  query: CatalogQuery;
  selectedModelId: string | null;
  theme: "system" | "light" | "dark";
}

export function mountCatalog(root: HTMLElement, catalog: Catalog, options: AppOptions): CatalogApp {
  const state: AppState = {
    locale: options.locale,
    query: { ...DEFAULT_QUERY },
    selectedModelId: null,
    theme: "system"
  };
  document.documentElement.lang = options.locale;

  const content = document.createElement("div");
  content.className = "catalog-app";
  const summary = renderSummary(catalog, options.locale);
  const resultStatus = document.createElement("p");
  resultStatus.className = "result-count";
  resultStatus.setAttribute("aria-live", "polite");
  const sortNotice = document.createElement("p");
  sortNotice.className = "sort-notice";
  sortNotice.setAttribute("role", "status");
  sortNotice.hidden = true;
  const results = document.createElement("section");
  results.className = "catalog-results";
  results.setAttribute("aria-label", t(options.locale, "summary.models"));
  const grid = document.createElement("div");
  grid.className = "model-grid";
  results.append(resultStatus, sortNotice, grid);

  let renderedCards: RenderedModelCard[] = [];
  let destroyed = false;
  const clearCards = (): void => {
    for (const card of renderedCards) card.destroy();
    renderedCards = [];
  };
  const renderResults = (): void => {
    clearCards();
    const queryResult = queryModels(catalog.models, state.query);
    resultStatus.textContent = t(options.locale, "filter.resultCount", { count: queryResult.models.length });
    sortNotice.hidden = queryResult.reason === undefined;
    sortNotice.textContent = queryResult.reason === "missing-benchmarks"
      ? t(options.locale, "filter.sortMissing")
      : queryResult.reason === "incomparable-benchmarks"
        ? t(options.locale, "filter.sortIncomparable")
        : "";
    grid.replaceChildren();
    if (queryResult.models.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-results";
      const heading = document.createElement("h3");
      heading.textContent = t(options.locale, "filter.noResults");
      const hint = document.createElement("p");
      hint.textContent = t(options.locale, "filter.noResultsHint");
      empty.append(heading, hint);
      grid.append(empty);
      return;
    }
    for (const model of queryResult.models) {
      const card = renderModelCard(model, catalog.release.platform, options.locale, (modelId) => {
        state.selectedModelId = modelId;
        options.onSelectModel?.(modelId);
      });
      renderedCards.push(card);
      grid.append(card.element);
    }
  };

  const filters = createFilters(catalog, options.locale, state.query, (query) => {
    if (destroyed) return;
    state.query = query;
    renderResults();
  });
  content.append(summary, filters.element, results);
  root.replaceChildren(content);
  renderResults();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      filters.destroy();
      clearCards();
    },
    state() {
      return { ...state, query: { ...state.query } };
    }
  };
}
