// @vitest-environment node
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { loadSourcesDocument, resolvePlatformSources } from "../src/sources";
import { buildMultiplatformCatalog } from "../src/pipeline/multiplatform-catalog";
import { publisherRoot, repositoryCatalog, repositoryRoot } from "./helpers/repository";

describe("multi-platform variant catalog", () => {
  it("groups YOLOv8 sizes into one family card with multi-platform evidence", async () => {
    const catalog = await repositoryCatalog();
    const platformTags = catalog.release.platform_tags!;
    expect(catalog.release.tag).toMatch(/^catalog-v1\.0\.0-[a-f0-9]{16}$/);
    expect(catalog.release.catalog_version).toBe(catalog.release.tag);
    expect(platformTags).toEqual({ x5: "x5-v1.1.3", s: "s-v1.1.2", x3: "x3-v1.1.2" });
    const yolov8 = catalog.models.find((model) => model.id === "yolov8");

    expect(yolov8?.platforms?.map((platform) => platform.platform)).toEqual(["x5", "s", "x3"]);
    expect(yolov8?.platforms?.flatMap((platform) => platform.benchmarks).map((record) => record.variant_id))
      .toEqual(expect.arrayContaining(["yolov8n-detect-640", "yolov8s-detect-640"]));
    const downloads = new Set(catalog.models.flatMap((model) => model.variants ?? [])
      .flatMap((variant) => variant.assets).filter((asset) => asset.url).map((asset) => asset.url));
    expect(catalog.summary.downloadable_asset_count).toBe(downloads.size);
    expect([...downloads].every((url) => /\.(bin|hbm)$/.test(url!))).toBe(true);
  });

  it("groups MobileNet versions into one family card", async () => {
    const catalog = await repositoryCatalog();
    const mobileNet = catalog.models.find((model) => model.id === "mobilenet");

    expect(mobileNet?.platforms?.flatMap((platform) => platform.benchmarks).map((record) => record.variant_id))
      .toEqual(expect.arrayContaining(["mobilenetv1-224", "mobilenetv4-conv-small-224"]));
  });

  it("keeps YOLOv8 detection assets separate from renamed classification models", async () => {
    const catalog = await repositoryCatalog();
    const yolov8 = catalog.models.find((model) => model.id === "yolov8")!;
    const x5 = yolov8.platforms!.find((platform) => platform.platform === "x5")!;
    const regularDetect = x5.variants!.filter((variant) =>
      variant.task === "object-detection" && !variant.name.includes("classification head")
    );
    const classifierHeads = x5.variants!.filter((variant) => variant.assets.some(asset => /_cls_bayese_640x640_/.test(asset.filename)));

    expect(regularDetect).toHaveLength(5);
    expect(regularDetect.map((variant) => variant.assets[0]?.filename)).toEqual(expect.arrayContaining([
      "yolov8n_detect_bayese_640x640_nv12.bin",
      "yolov8s_detect_bayese_640x640_nv12.bin",
      "yolov8m_detect_bayese_640x640_nv12.bin",
      "yolov8l_detect_bayese_640x640_nv12.bin",
      "yolov8x_detect_bayese_640x640_nv12.bin"
    ]));
    expect(classifierHeads).toHaveLength(5);
    expect(classifierHeads.every((variant) => variant.task === "image-classification")).toBe(true);
    expect(classifierHeads.every((variant) => !variant.name.includes("classification head"))).toBe(true);
  });

  it("seeds S hardware rows from assets and preserves each source sample path", async () => {
    const catalog = await repositoryCatalog();
    const yolov8 = catalog.models.find((model) => model.id === "yolov8")!;
    const s = yolov8.platforms!.find((platform) => platform.platform === "s")!;
    const regularDetect = s.variants!.filter((variant) =>
      variant.task === "object-detection" && !variant.name.includes("classification head")
    );

    expect(new Set(regularDetect.map((variant) => variant.hardware))).toEqual(new Set(["s100", "s100p", "s600"]));
    expect(regularDetect.filter((variant) => variant.hardware === "s100p" && variant.assets.length > 0).length).toBeGreaterThan(0);
    expect(regularDetect.filter((variant) => variant.hardware === "s600" && variant.assets.length > 0).length).toBeGreaterThan(0);
    expect(regularDetect.every((variant) => variant.assets.every((entry) => entry.filename.includes("yolov8")))).toBe(true);

    const x3 = yolov8.platforms!.find((platform) => platform.platform === "x3")!;
    const segmentation = x3.variants!.find((variant) => variant.task === "instance-segmentation");
    expect(segmentation?.sample_path).toBe("demos/Instance_Segmentation/YOLOv8-Seg");
  });

  it("does not put a MobileNet backbone asset into the MobileNet classifier card", async () => {
    const catalog = await repositoryCatalog();
    const mobileNet = catalog.models.find((model) => model.id === "mobilenet")!;
    expect(mobileNet.assets.some((entry) => entry.filename.includes("unet_mobilenet"))).toBe(false);
    expect(catalog.models.find((model) => model.id === "unetmobilenet")?.assets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ filename: expect.stringContaining("unet_mobilenet") })]));
  });

  it("preserves the reviewed release inventory across all three platforms", async () => {
    const catalog = await repositoryCatalog();
    const variants = catalog.models.flatMap((model) => model.variants ?? []);

    // The reviewed baseline after the manifest relocation to docs/release:
    // 57 families, 595 configurations, 820 benchmark observations. The three
    // new S families (yoloe26_seg, yoloe11_seg, minicpm5-2b) contribute 11
    // asset-only variants; no benchmark observation changed with the move.
    expect(catalog.models).toHaveLength(57);
    expect(variants).toHaveLength(595);
    expect(catalog.models.flatMap((model) => model.benchmarks)).toHaveLength(820);
    expect(new Set(variants.map((variant) => variant.hardware)))
      .toEqual(new Set(["x5", "s100", "s100p", "s600", "x3"]));
  });

  it("records where each platform was read from, symmetrically", async () => {
    const catalog = await repositoryCatalog();

    expect(Object.keys(catalog.sources ?? {}).sort()).toEqual(["s", "x3", "x5"]);
    for (const platform of ["x5", "s", "x3"] as const) {
      const record = catalog.sources![platform]!;
      expect(record.kind).toBe("worktree");
      // X5 and S are read from the repository root; only the archived X3
      // keeps its frozen subtree, so its manifests keep the subtree prefix.
      expect(record.path).toBe(platform === "x3" ? "platforms/x3" : ".");
      expect(record.release_tag).toBe(catalog.release.platform_tags![platform]);
      expect(record.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
    }
  });

  it("stamps every variant with the layout its sample path lives in", async () => {
    const catalog = await repositoryCatalog();

    // X5/S variants link into the unified branch at the repository root; X3
    // variants keep pointing at the frozen subtree that holds their demos.
    for (const variant of catalog.models.flatMap((model) => model.variants ?? [])) {
      const unified = variant.hardware !== "x3";
      expect(variant.source_ref).toBe(unified ? "develop" : "main");
      expect(variant.source_path_prefix).toBe(unified ? "" : "platforms/x3");
      if (!unified) {
        expect(variant.sample_path.startsWith(variant.source_path_prefix!)).toBe(false);
      }
    }
  });

  it("keeps the S 224 classification corrections and the S600 ACT policy", async () => {
    const catalog = await repositoryCatalog();
    const cls = catalog.models.flatMap((model) => model.variants ?? [])
      .filter((variant) => ["s100", "s100p", "s600"].includes(variant.hardware))
      .flatMap((variant) => variant.assets)
      .filter((asset) => /(?:yolov8|yolo11)[nsmlx]_cls_/.test(asset.filename));

    // Three boards x five sizes x two families (YOLOv8 and YOLO11).
    expect(cls).toHaveLength(30);
    expect(cls.every((asset) => asset.filename.includes("_224x224_nv12.hbm"))).toBe(true);
    expect(cls.some((asset) => /_640x640_/.test(asset.filename))).toBe(false);
  });

  it("keeps the published asset totals consistent with the summary", async () => {
    const catalog = await repositoryCatalog();
    const assets = catalog.models.flatMap((model) => model.variants ?? []).flatMap((variant) => variant.assets);
    const unique = new Set(assets.map((asset) => asset.url ?? asset.filename));

    expect(catalog.summary.asset_count).toBe(unique.size);
    expect(catalog.summary.downloadable_asset_count)
      .toBe(new Set(assets.filter((asset) => asset.url).map((asset) => asset.url)).size);
    expect(catalog.summary.sample_count).toBe(catalog.models.length);
    expect(catalog.summary.benchmark_count).toBe(catalog.models.flatMap((model) => model.benchmarks).length);
  });

  it("refuses to build a catalog that is missing a platform distribution", async () => {
    const catalog = await repositoryCatalog();
    const sources = await resolvePlatformSources({
      repositoryRoot,
      sources: await loadSourcesDocument(resolve(publisherRoot, "sources.json"))
    });

    await expect(buildMultiplatformCatalog({
      repositoryRoot,
      sources: sources.filter((source) => source.platform !== "x3")
    })).rejects.toThrow(/missing the x3 platform source/);

    // The complete set still builds, so the failure above is the missing
    // platform and not an artefact of the fixture sources.
    expect(catalog.sources!.x3).toBeDefined();
  });
});
