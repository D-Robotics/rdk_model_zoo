// @vitest-environment node
import { expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";

it("uses canonical 224 classification assets on all three S boards", async () => {
  const catalog = await buildMultiplatformCatalog(fileURLToPath(new URL("../../../", import.meta.url)));
  const variants = catalog.models.flatMap(model => model.variants ?? []).filter(variant =>
    ["s100", "s100p", "s600"].includes(variant.hardware) && variant.assets.some(asset =>
      /(?:yolov8|yolo11)[nsmlx]_cls_/.test(asset.filename)));
  expect(variants).toHaveLength(30);
  for (const variant of variants) {
    expect(variant.task).toBe("image-classification");
    expect(variant.input?.shape?.slice(-2)).toEqual([224, 224]);
    for (const asset of variant.assets) {
      expect(asset.filename).toContain("_224x224_nv12.hbm");
      expect(asset.url?.split("/").at(-1)).toBe(asset.filename.split("/").at(-1));
    }
  }
});
