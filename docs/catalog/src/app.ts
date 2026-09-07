import { queryModels, type CatalogQuery } from "./catalog/query";
import type { Catalog, HardwareId, Locale } from "./catalog/types";
import { createLanguageController, type LanguageController } from "./i18n/language";
import { t } from "./i18n/translations";
import { createFilters } from "./ui/filters";
import { normalizeHardware } from "./catalog/variants";
import { readCatalogQuery, writeCatalogQuery } from "./ui/navigation";
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
    query: readCatalogQuery(new URL(window.location.href)),
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
  const directory = document.createElement("div");
  directory.className = "catalog-directory";
  const titleFor = (name?: string, hardware?: string): void => {
    document.title = [name, hardware?.toUpperCase(), t(options.locale, "app.title")].filter(Boolean).join(" · ");
  };
  const updateSelection = (hardware: HardwareId, task: string, push: boolean): void => {
    const next = new URL(window.location.href);
    next.searchParams.set("hardware", hardware);
    next.searchParams.set("task", task);
    if (next.href !== window.location.href) {
      if (push) window.history.pushState({}, "", next);
      else window.history.replaceState({}, "", next);
    }
    titleFor(catalog.models.find((model) => model.id === state.selectedModelId)?.name, hardware);
  };
  const closeDetails = (): void => {
    const next = writeModelId(new URL(window.location.href), null);
    next.searchParams.delete("hardware");
    next.searchParams.delete("task");
    window.history.pushState({}, "", next);
    renderDetails(null, true);
  };
  const renderDetails = (modelId: string | null, restoreFocus = false): void => {
    clearDetails();
    state.selectedModelId = modelId;
    directory.hidden = modelId !== null;
    content.classList.toggle("showing-detail", modelId !== null);
    document.querySelector(".site-header")?.classList.toggle("detail-page-header", modelId !== null);
    if (modelId === null) {
      titleFor();
      if (restoreFocus && detailOpener?.isConnected) detailOpener.focus();
      detailOpener = null;
      return;
    }
    const model = catalog.models.find((candidate) => candidate.id === modelId);
    if (!model) {
      const panel = document.createElement("section");
      panel.className = "model-details model-not-found";
      const heading = document.createElement("h1");
      heading.id = "model-details-not-found";
      heading.textContent = t(options.locale, "details.notFound");
      panel.setAttribute("aria-labelledby", heading.id);
      const message = document.createElement("p");
      message.textContent = t(options.locale, "error.modelNotFound");
      const clear = document.createElement("button");
      clear.type = "button";
      clear.dataset.action = "clear-model";
      clear.textContent = options.locale === "zh" ? "返回模型目录" : "Back to model catalog";
      clear.addEventListener("click", closeDetails);
      panel.append(heading, message, clear);
      detailHost.append(panel);
      heading.tabIndex = -1;
      heading.focus();
      titleFor(heading.textContent);
      detailCleanup = () => clear.removeEventListener("click", closeDetails);
      return;
    }
    const url = new URL(window.location.href);
    const details = renderModelDetails(model, {
      locale: options.locale,
      repositoryUrl: REPOSITORY_URL,
      releaseTag: catalog.release.tag,
      hardware: normalizeHardware(url.searchParams.get("hardware") ?? state.query.platform),
      task: url.searchParams.get("task") ?? undefined,
      onSelectionChange: (hardware, task) => updateSelection(hardware, task, true)
    });
    const close = details.querySelector<HTMLButtonElement>('[data-action="close-details"]')!;
    close.addEventListener("click", closeDetails);
    detailHost.append(details);
    const hardware = normalizeHardware(details.dataset.hardware ?? "");
    if (hardware && details.dataset.task) updateSelection(hardware, details.dataset.task, false);
    else titleFor(model.name);
    const heading = details.querySelector<HTMLElement>("h1");
    if (heading) { heading.tabIndex = -1; heading.focus(); }
    detailCleanup = () => close.removeEventListener("click", closeDetails);
  };
  const openDetails = (modelId: string, hardware?: HardwareId): void => {
    detailOpener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const next = writeModelId(new URL(window.location.href), modelId);
    const preferred = hardware ?? normalizeHardware(state.query.platform);
    if (preferred) next.searchParams.set("hardware", preferred);
    else next.searchParams.delete("hardware");
    next.searchParams.delete("task");
    window.history.pushState({}, "", next);
    renderDetails(modelId);
    options.onSelectModel?.(modelId);
  };
  const restoreDetails = (): void => {
    const url = new URL(window.location.href);
    const query = readCatalogQuery(url);
    if (JSON.stringify(query) !== JSON.stringify(state.query)) {
      state.query = query;
      filters.setQuery(query);
      renderResults();
    }
    renderDetails(readModelId(url), true);
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
      const card = renderModelCard(model, catalog.release.platform, options.locale, (modelId, hardware) => {
        openDetails(modelId, hardware);
      });
      renderedCards.push(card);
      grid.append(card.element);
    }
  };

  const filters = createFilters(catalog, options.locale, state.query, (query) => {
    if (destroyed) return;
    state.query = query;
    window.history.replaceState({}, "", writeCatalogQuery(new URL(window.location.href), query));
    renderResults();
  });
  directory.append(summary, filters.element, results);
  content.append(preferences, directory, detailHost);
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
