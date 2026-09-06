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
  });

  it("groups MobileNet versions into one family card", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const mobileNet = catalog.models.find((model) => model.id === "mobilenet");

    expect(mobileNet?.platforms?.flatMap((platform) => platform.benchmarks).map((record) => record.variant_id))
      .toEqual(expect.arrayContaining(["mobilenetv1-224", "mobilenetv4-conv-small-224"]));
  });
});
