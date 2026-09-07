import { beforeEach, expect, it, vi } from "vitest";
import { createFilters, DEFAULT_QUERY } from "../src/ui/filters";
import type { Catalog } from "../src/catalog/types";
import { createModelFixture } from "./fixtures/catalog";
import { groupTasks } from "../src/catalog/task-groups";

const catalog: Catalog = { schema_version: 1, release: { tag: "test", platform: "x5", version: "1" }, summary: {},
  models: [createModelFixture({ tasks: ["object-detection", "image-classification", "speech-recognition"] })] };
beforeEach(() => { document.body.replaceChildren(); });
function setup() {
  const changed = vi.fn();
  const panel = createFilters(catalog, "en", { ...DEFAULT_QUERY, platform: "x5", text: "model" }, changed);
  document.body.append(panel.element, panel.hardware, panel.toolbar, panel.chips);
  return { panel, changed };
}
it("preserves other filters when toggling task groups and removing one chip", () => {
  const { changed, panel } = setup();
  document.querySelector<HTMLInputElement>('[data-task-group="vision"]')!.click();
  expect(changed.mock.lastCall?.[0]).toMatchObject({ platform: "x5", text: "model", tasks: ["image-classification", "object-detection"] });
  document.querySelector<HTMLButtonElement>('[aria-label="Remove Object detection"]')!.click();
  expect(changed.mock.lastCall?.[0]).toMatchObject({ platform: "x5", text: "model", tasks: ["image-classification"] });
  expect(document.querySelector<HTMLInputElement>('[data-task-group="vision"]')!.indeterminate).toBe(true);
  panel.destroy();
});
it("keeps mobile drafts private until applied and discards cancelled changes", () => {
  const { changed, panel } = setup();
  document.querySelector<HTMLButtonElement>('.mobile-filter-toggle')!.click();
  document.querySelector<HTMLInputElement>('[value="speech-recognition"]')!.click();
  expect(changed).not.toHaveBeenCalled();
  document.querySelector<HTMLButtonElement>('.mobile-filter-cancel')!.click();
  expect(document.querySelector<HTMLInputElement>('[value="speech-recognition"]')!.checked).toBe(false);
  document.querySelector<HTMLButtonElement>('.mobile-filter-toggle')!.click();
  document.querySelector<HTMLInputElement>('[value="speech-recognition"]')!.click();
  document.querySelector<HTMLButtonElement>('.mobile-filter-apply')!.click();
  expect(changed.mock.lastCall?.[0].tasks).toEqual(["speech-recognition"]);
  expect(panel.element.hasAttribute("aria-modal")).toBe(false);
  panel.destroy();
});
it("round trips multiple tasks and preserves unknown tasks in the fallback group", () => {
  const { panel } = setup();
  panel.setQuery({ ...DEFAULT_QUERY, tasks: ["object-detection", "speech-recognition"] });
  expect(document.querySelectorAll('input[name="catalog-task"]:checked')).toHaveLength(2);
  expect(groupTasks(["future-task", "object-detection"], "en").flatMap(g => g.tasks).sort()).toEqual(["future-task", "object-detection"]);
  panel.destroy();
});
