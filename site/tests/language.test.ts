import { beforeEach, describe, expect, it } from "vitest";
import { createLanguageController } from "../src/i18n/language";
import { t, translations, type TranslationKey } from "../src/i18n/translations";

describe("language preference", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.lang = "";
  });

  it("uses Chinese for a zh browser when no preference is stored", () => {
    expect(createLanguageController(localStorage, "zh-CN").current()).toBe("zh");
  });

  it("persists an explicit English selection over browser language", () => {
    const controller = createLanguageController(localStorage, "zh-CN");
    controller.set("en");

    expect(createLanguageController(localStorage, "zh-CN").current()).toBe("en");
  });

  it("falls back to English for an unsupported browser language", () => {
    expect(createLanguageController(localStorage, "fr-FR").current()).toBe("en");
  });

  it("sets the document language for the initial and changed locale", () => {
    const controller = createLanguageController(localStorage, "zh-CN");
    expect(document.documentElement.lang).toBe("zh");

    controller.set("en");
    expect(document.documentElement.lang).toBe("en");
  });

  it("notifies subscribers when the locale changes and lets them unsubscribe", () => {
    const controller = createLanguageController(localStorage, "en-US");
    const locales: string[] = [];
    const unsubscribe = controller.subscribe((locale) => locales.push(locale));

    controller.set("zh");
    unsubscribe();
    controller.set("en");

    expect(locales).toEqual(["zh"]);
  });

  it("substitutes named placeholders in a translated string", () => {
    expect(t("en", "details.modelTitle", { name: "ConvNeXt" })).toBe("ConvNeXt details");
    expect(t("zh", "details.modelTitle", { name: "ConvNeXt" })).toBe("ConvNeXt 详情");
  });

  it("throws when a requested translation key is missing", () => {
    expect(() => t("en", "missing.key" as TranslationKey)).toThrow(/missing translation/i);
  });

  it("keeps Chinese and English dictionaries in exact parity", () => {
    expect(Object.keys(translations.zh).sort()).toEqual(Object.keys(translations.en).sort());
  });
});
