// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

describe("multi-platform variant catalog", () => {
  it("keeps exact YOLO variants separate while aggregating their platform evidence", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolov5su = catalog.models.find((model) => model.id === "yolov5su-detect-640");
    const x3Yolov5s = catalog.models.find((model) => model.id.includes("yolov5s-v7.0-640x640"));

    expect(yolov5su?.platforms?.map((platform) => platform.platform)).toEqual(["x5", "s"]);
    expect(yolov5su?.platforms?.flatMap((platform) => platform.benchmarks).some((record) => (record.accuracy?.length ?? 0) > 0)).toBe(true);
    expect(x3Yolov5s?.id).not.toBe(yolov5su?.id);
  });

  it("aggregates only reviewed, exact equivalents", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolov8n = catalog.models.find((model) => model.id === "yolov8n-detect-640");
    const efficientFormer = catalog.models.filter((model) => model.name.includes("EfficientFormer"));

    expect(yolov8n?.platforms?.map((platform) => platform.platform)).toEqual(["x5", "s", "x3"]);
    expect(efficientFormer.every((model) => (model.platforms?.length ?? 0) === 1)).toBe(true);
  });
});
