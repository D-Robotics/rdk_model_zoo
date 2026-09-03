import type { Locale } from "../catalog/types";

export const LANGUAGE_STORAGE_KEY = "rdk-model-zoo-locale";

export interface LanguageController {
  current(): Locale;
  set(locale: Locale): void;
  subscribe(listener: (locale: Locale) => void): () => void;
}

function browserLocale(browserLanguage: string): Locale {
  return browserLanguage.trim().toLowerCase().startsWith("zh") ? "zh" : "en";
}

function storedLocale(storage: Storage): Locale | null {
  try {
    const value = storage.getItem(LANGUAGE_STORAGE_KEY);
    return value === "zh" || value === "en" ? value : null;
  } catch {
    return null;
  }
}

function persistLocale(storage: Storage, locale: Locale): void {
  try {
    storage.setItem(LANGUAGE_STORAGE_KEY, locale);
  } catch {
    // A blocked or unavailable storage should not prevent language switching.
  }
}

function setDocumentLanguage(locale: Locale): void {
  if (typeof document !== "undefined") {
    document.documentElement.lang = locale;
  }
}

export function createLanguageController(
  storage: Storage,
  browserLanguage: string
): LanguageController {
  let locale: Locale = storedLocale(storage) ?? browserLocale(browserLanguage);
  const listeners = new Set<(nextLocale: Locale) => void>();
  setDocumentLanguage(locale);

  return {
    current(): Locale {
      return locale;
    },

    set(nextLocale: Locale): void {
      if (nextLocale !== "zh" && nextLocale !== "en") {
        throw new RangeError(`Unsupported locale: ${String(nextLocale)}`);
      }

      persistLocale(storage, nextLocale);
      if (nextLocale === locale) {
        setDocumentLanguage(locale);
        return;
      }

      locale = nextLocale;
      setDocumentLanguage(locale);
      for (const listener of [...listeners]) {
        listener(locale);
      }
    },

    subscribe(listener: (nextLocale: Locale) => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    }
  };
}
