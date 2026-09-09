import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";

const repositoryRoot = resolve(process.cwd(), "../..");

describe("family rows joined across sample directories", () => {
  it("keeps one row per YOLO26 size on every task, with assets and benchmarks", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolo26 = catalog.models.find((model) => model.id === "yolov26");
    expect(yolo26).toBeDefined();

    for (const task of [
      "object-detection",
      "oriented-bounding-box-detection",
      "instance-segmentation",
      "pose-estimation",
      "image-classification"
    ]) {
      const variants = (yolo26?.variants ?? []).filter(
        (variant) => variant.hardware === "s100" && variant.task === task
      );
      expect(variants, `task ${task}`).toHaveLength(5);
      for (const variant of variants) {
        // The Ultralytics index publishes files for every size, so a row
        // without an asset means the asset-only row was not joined to the
        // measured row.
        expect(variant.assets, `task ${task} variant ${variant.id}`).toHaveLength(1);
      }
    }
  });

  it("does not leave asset-only rows whose assets are already attached to measured rows", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const duplicates: string[] = [];
    for (const model of catalog.models) {
      const measured = (model.variants ?? []).filter((variant) => variant.benchmarks.length > 0);
      for (const variant of model.variants ?? []) {
        if (variant.benchmarks.length > 0) continue;
        // An asset-only row is legitimate ("unmeasured but runnable"); it is a
        // duplicate only when a measured row already carries the same asset.
        const redundant = variant.assets.some((asset) =>
          measured.some((peer) =>
            peer.hardware === variant.hardware
            && peer.task === variant.task
            && peer.assets.some((candidate) => candidate.url === asset.url || candidate.filename === asset.filename)
          )
        );
        if (redundant) duplicates.push(`${model.id}/${variant.id}`);
      }
    }
    expect(duplicates).toEqual([]);
  });
});
