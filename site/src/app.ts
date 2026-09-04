import { queryModels, type CatalogQuery } from "./catalog/query";
import type { Catalog, Locale } from "./catalog/types";
import { createLanguageController, type LanguageController } from "./i18n/language";
import { t } from "./i18n/translations";
import { createFilters, DEFAULT_QUERY } from "./ui/filters";
import { renderModelCard, type RenderedModelCard } from "./ui/model-card";
import { readModelId, renderModelDetails, writeModelId } from "./ui/model-details";
import { renderSummary } from "./ui/summary";

const REPOSITORY_URL = "https://github.com/D-Robotics/rdk_model_zoo";
const THEME_STORAGE_KEY = "rdk-model-zoo-theme";
type ThemePreference = "system" | "light" | "dark";

export interface AppOptions {
  locale: Locale;
  onSelectModel?: (modelId: string) => void;
  languageController?: LanguageController;
  onLocaleChange?: (locale: Locale) => void;
  storage?: Storage;
  matchMedia?: (query: string) => MediaQueryList;
}

export interface CatalogApp {
  destroy(): void;
  state(): Readonly<AppState>;
}

interface AppState {
  locale: Locale;
  query: CatalogQuery;
  selectedModelId: string | null;
  theme: ThemePreference;
}

function storedTheme(storage: Storage): ThemePreference {
  try {
    const value = storage.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" || value === "system" ? value : "system";
  } catch {
    return "system";
  }
}

function persistTheme(storage: Storage, theme: ThemePreference): void {
  try {
    storage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // The visual preference remains usable when storage is blocked.
  }
}

export function mountCatalog(root: HTMLElement, catalog: Catalog, options: AppOptions): CatalogApp {
  const storage = options.storage ?? window.localStorage;
  const languageController = options.languageController
    ?? createLanguageController(storage, options.locale);
  const darkMedia = options.matchMedia
    ? options.matchMedia("(prefers-color-scheme: dark)")
    : typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-color-scheme: dark)")
      : undefined;
  const state: AppState = {
    locale: options.locale,
    query: { ...DEFAULT_QUERY },
    selectedModelId: null,
    theme: storedTheme(storage)
  };
  document.documentElement.lang = options.locale;

  const preferences = document.createElement("div");
  preferences.className = "preference-controls";
  const languageButton = document.createElement("button");
  languageButton.type = "button";
  languageButton.className = "preference-button";
  languageButton.setAttribute("aria-label", t(options.locale, "control.switchLanguage"));
  languageButton.textContent = `${t(options.locale, "control.language")}: ${t(
    options.locale,
    options.locale === "en" ? "control.english" : "control.chinese"
  )}`;
  const themeButton = document.createElement("button");
  themeButton.type = "button";
  themeButton.className = "preference-button";
  themeButton.setAttribute("aria-label", t(options.locale, "control.changeTheme"));
  preferences.append(languageButton, themeButton);

  const themeLabel = (theme: ThemePreference): string => t(
    options.locale,
    theme === "system" ? "control.system" : theme === "light" ? "control.light" : "control.dark"
  );
  const applyTheme = (): void => {
    const resolved = state.theme === "system" ? (darkMedia?.matches ? "dark" : "light") : state.theme;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themePreference = state.theme;
    document.documentElement.style.colorScheme = resolved;
    themeButton.textContent = `${t(options.locale, "control.theme")}: ${themeLabel(state.theme)}`;
  };
  const switchLanguage = (): void => {
    const locale: Locale = state.locale === "en" ? "zh" : "en";
    languageController.set(locale);
    options.onLocaleChange?.(locale);
  };
  const cycleTheme = (): void => {
    state.theme = state.theme === "system" ? "light" : state.theme === "light" ? "dark" : "system";
    persistTheme(storage, state.theme);
    applyTheme();
  };
  const followSystemTheme = (): void => {
    if (state.theme === "system") applyTheme();
  };
  languageButton.addEventListener("click", switchLanguage);
  themeButton.addEventListener("click", cycleTheme);
  darkMedia?.addEventListener("change", followSystemTheme);
  applyTheme();

  const content = document.createElement("div");
  content.className = "catalog-app";
  const summary = renderSummary(catalog, options.locale);
  const resultStatus = document.createElement("p");
  resultStatus.className = "result-count";
  resultStatus.dataset.testid = "result-count";
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
  const detailHost = document.createElement("div");
  detailHost.className = "detail-host";

  let renderedCards: RenderedModelCard[] = [];
  let detailCleanup: (() => void) | undefined;
  let detailOpener: HTMLElement | null = null;
  let destroyed = false;
  const clearCards = (): void => {
    for (const card of renderedCards) card.destroy();
    renderedCards = [];
  };
  const clearDetails = (): void => {
    detailCleanup?.();
    detailCleanup = undefined;
    detailHost.replaceChildren();
  };
  const renderDetails = (modelId: string | null, restoreFocus = false): void => {
    clearDetails();
    state.selectedModelId = modelId;
    if (modelId === null) {
      if (restoreFocus && detailOpener?.isConnected) detailOpener.focus();
      detailOpener = null;
      return;
    }

    const model = catalog.models.find((candidate) => candidate.id === modelId);
    if (!model) {
      const panel = document.createElement("section");
      panel.className = "model-details model-not-found";
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
      const heading = document.createElement("h2");
      heading.id = "model-details-not-found";
      heading.textContent = t(options.locale, "details.notFound");
      panel.setAttribute("aria-labelledby", heading.id);
      const message = document.createElement("p");
      message.textContent = t(options.locale, "error.modelNotFound");
      const clear = document.createElement("button");
      clear.type = "button";
      clear.dataset.action = "clear-model";
      clear.textContent = t(options.locale, "details.clearModel");
      const clearModel = (): void => {
        window.history.pushState({}, "", writeModelId(new URL(window.location.href), null));
        renderDetails(null, true);
      };
      clear.addEventListener("click", clearModel);
      panel.append(heading, message, clear);
      detailHost.append(panel);
      detailCleanup = () => clear.removeEventListener("click", clearModel);
      return;
    }

    const details = renderModelDetails(model, {
      locale: options.locale,
      repositoryUrl: REPOSITORY_URL,
      releaseTag: catalog.release.tag
    });
    const close = details.querySelector<HTMLButtonElement>('[data-action="close-details"]')!;
    const closeDetails = (): void => {
      window.history.pushState({}, "", writeModelId(new URL(window.location.href), null));
      renderDetails(null, true);
    };
    const cancelDetails = (event: Event): void => {
      event.preventDefault();
      closeDetails();
    };
    close.addEventListener("click", closeDetails);
    details.addEventListener("cancel", cancelDetails);
    detailHost.append(details);
    if (details instanceof HTMLDialogElement && typeof details.showModal === "function") {
      details.showModal();
    } else {
      close.focus();
    }
    detailCleanup = () => {
      close.removeEventListener("click", closeDetails);
      details.removeEventListener("cancel", cancelDetails);
    };
  };
  const openDetails = (modelId: string): void => {
    detailOpener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.history.pushState({}, "", writeModelId(new URL(window.location.href), modelId));
    renderDetails(modelId);
    options.onSelectModel?.(modelId);
  };
  const restoreDetails = (): void => renderDetails(readModelId(new URL(window.location.href)));

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
        openDetails(modelId);
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
  content.append(preferences, summary, filters.element, results, detailHost);
  root.replaceChildren(content);
  renderResults();
  window.addEventListener("popstate", restoreDetails);
  restoreDetails();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      window.removeEventListener("popstate", restoreDetails);
      darkMedia?.removeEventListener("change", followSystemTheme);
      languageButton.removeEventListener("click", switchLanguage);
      themeButton.removeEventListener("click", cycleTheme);
      filters.destroy();
      clearCards();
      clearDetails();
    },
    state() {
      return {
        ...state,
        query: {
          ...state.query,
          tasks: [...state.query.tasks],
          formats: [...state.query.formats],
          precisions: [...state.query.precisions]
        }
      };
    }
  };
}
