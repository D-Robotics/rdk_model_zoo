// @vitest-environment node
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import {
  manifestTaskTranslationKeys,
  taskTranslationKey,
  translations
} from "../src/i18n/translations";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
describe("manifest task translations", () => {
  it("maps every task in the production manifest to a bilingual translation key", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const productionTaskIds = [...new Set(catalog.models.flatMap((model) => model.tasks))].sort();

    for (const task of productionTaskIds) {
      expect(taskTranslationKey(task), task).not.toBe("task.unknown");
    }

    for (const translationKey of Object.values(manifestTaskTranslationKeys)) {
      expect(translations.en[translationKey]).toBeTruthy();
      expect(translations.zh[translationKey]).toBeTruthy();
    }

    expect(taskTranslationKey("object-detection")).toBe("task.objectDetection");
    expect(taskTranslationKey("future-task")).toBe("task.unknown");
  });
});
