import { mountCatalog, type CatalogApp } from "./app";
import type { Catalog, Locale } from "./catalog/types";
import { createLanguageController } from "./i18n/language";
import { t, type TranslationKey } from "./i18n/translations";
import "./styles.css";

const REPOSITORY_URL = "https://github.com/D-Robotics/rdk_model_zoo";
const MANIFEST_URL = `${REPOSITORY_URL}/blob/rdk_x5/release/models.yaml`;

export interface LoadCatalogOptions {
  locale: Locale;
  fetcher?: typeof fetch;
  catalogUrl?: string;
  languageController?: ReturnType<typeof createLanguageController>;
  onLocaleChange?: (locale: Locale) => void;
}

export function localizeDocumentShell(locale: Locale): void {
  document.documentElement.lang = locale;
  document.title = t(locale, "app.title");
  for (const element of document.querySelectorAll<HTMLElement>("[data-i18n]")) {
    const key = element.dataset.i18n as TranslationKey | undefined;
    if (key) element.textContent = t(locale, key);
  }
}

function renderLoadError(root: HTMLElement, options: LoadCatalogOptions): void {
  root.replaceChildren();
  root.setAttribute("role", "alert");
  const heading = document.createElement("h2");
  heading.textContent = t(options.locale, "error.loadCatalog");
  const actions = document.createElement("div");
  actions.className = "load-error-actions";
  const retry = document.createElement("button");
  retry.type = "button";
  retry.dataset.action = "retry-load";
  retry.textContent = t(options.locale, "error.retry");
  retry.addEventListener("click", () => void loadCatalog(root, options), { once: true });
  const manifest = document.createElement("a");
  manifest.href = MANIFEST_URL;
  manifest.textContent = t(options.locale, "error.openManifest");
  const repository = document.createElement("a");
  repository.href = REPOSITORY_URL;
  repository.textContent = t(options.locale, "error.openRepository");
  actions.append(retry, manifest, repository);
  root.append(heading, actions);
}

export async function loadCatalog(root: HTMLElement, options: LoadCatalogOptions): Promise<CatalogApp | null> {
  const fetcher = options.fetcher ?? fetch;
  const catalogUrl = options.catalogUrl ?? `${import.meta.env.BASE_URL}data/catalog.json`;
  try {
    const response = await fetcher(catalogUrl);
    if (!response.ok) throw new Error(`Catalog request failed with HTTP ${response.status}.`);
    const catalog = await response.json() as Catalog;
    root.removeAttribute("role");
    return mountCatalog(root, catalog, {
      locale: options.locale,
      languageController: options.languageController,
      onLocaleChange: options.onLocaleChange
    });
  } catch {
    renderLoadError(root, options);
    return null;
  }
}

function bootstrap(): void {
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) return;
  const language = createLanguageController(window.localStorage, navigator.language);
  let mounted: CatalogApp | null = null;
  const render = async (locale: Locale): Promise<void> => {
    mounted?.destroy();
    localizeDocumentShell(locale);
    mounted = await loadCatalog(root, {
      locale,
      languageController: language,
      onLocaleChange: (nextLocale) => void render(nextLocale)
    });
  };
  void render(language.current());
}

bootstrap();
