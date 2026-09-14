// @vitest-environment node
import { it, expect } from "vitest";
import { resolve } from "node:path";
import { repositoryCatalog } from "./helpers/repository";

it("keeps Gemma S100P and S600 models while treating shared embeddings as a dependency", async () => {
  const catalog = await repositoryCatalog();
  const model = catalog.models.find(model => model.id === "gemma4-e2b")!;
  expect([...new Set(model.variants!.map(variant => variant.hardware))].sort()).toEqual(["s100p", "s600"]);
  expect(model.variants!.flatMap(variant => variant.assets)).toHaveLength(4);
  for (const variant of model.variants!) {
    expect(variant.name).toMatch(/^Gemma4-E2B (Vision Encoder|Language Model)$/);
    if (variant.name.includes("Language Model")) expect(variant.input?.shape).toBeUndefined();
  }
  expect(model.assets.find(asset => asset.filename === "common/tok_embeddings.bin")?.role).toBe("dependency");
  expect(model.variants!.filter(variant => variant.hardware === "s100p").every(variant => variant.assets[0]?.url?.includes("/rdk_s100/"))).toBe(true);
});
