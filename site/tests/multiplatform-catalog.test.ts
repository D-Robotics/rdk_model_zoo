// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

describe("multi-platform variant catalog", () => {
  it("groups YOLOv8 sizes into one family card with multi-platform evidence", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
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
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const mobileNet = catalog.models.find((model) => model.id === "mobilenet");

    expect(mobileNet?.platforms?.flatMap((platform) => platform.benchmarks).map((record) => record.variant_id))
      .toEqual(expect.arrayContaining(["mobilenetv1-224", "mobilenetv4-conv-small-224"]));
  });

  it("keeps YOLOv8 size rows joined to the exact X5 assets and separates classifier heads", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolov8 = catalog.models.find((model) => model.id === "yolov8")!;
    const x5 = yolov8.platforms!.find((platform) => platform.platform === "x5")!;
    const regularDetect = x5.variants!.filter((variant) =>
      variant.task === "object-detection" && !variant.name.includes("classification head")
    );
    const classifierHeads = x5.variants!.filter((variant) => variant.name.includes("classification head"));

    expect(regularDetect).toHaveLength(5);
    expect(regularDetect.map((variant) => variant.assets[0]?.filename)).toEqual(expect.arrayContaining([
      "yolov8n_detect_bayese_640x640_nv12.bin",
      "yolov8s_detect_bayese_640x640_nv12.bin",
      "yolov8m_detect_bayese_640x640_nv12.bin",
      "yolov8l_detect_bayese_640x640_nv12.bin",
      "yolov8x_detect_bayese_640x640_nv12.bin"
    ]));
    expect(classifierHeads).toHaveLength(5);
    expect(classifierHeads.every((variant) => variant.task === "object-detection")).toBe(true);
    expect(classifierHeads.every((variant) => variant.name.includes("classification head"))).toBe(true);
  });

  it("seeds S hardware rows from assets and preserves each source sample path", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
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
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const mobileNet = catalog.models.find((model) => model.id === "mobilenet")!;
    expect(mobileNet.assets.some((entry) => entry.filename.includes("unet_mobilenet"))).toBe(false);
    expect(catalog.models.find((model) => model.id === "unetmobilenet")?.assets)
      .toEqual(expect.arrayContaining([expect.objectContaining({ filename: expect.stringContaining("unet_mobilenet") })]));
  });
});
