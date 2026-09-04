// @vitest-environment node
import { describe, expect, it, vi } from "vitest";
import { fileURLToPath } from "node:url";
import { buildCatalog } from "../scripts/catalog-builder";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

function buildRepositoryCatalog(onWarning?: (message: string) => void) {
  return buildCatalog({
    repositoryRoot,
    modelsPath: "release/models.yaml",
    benchmarksPath: "release/benchmarks.yaml",
    modelsSchemaPath: "release/schemas/models.schema.json",
    benchmarksSchemaPath: "release/schemas/benchmarks.schema.json",
    onWarning
  });
}

describe("buildCatalog", () => {
  it("includes exact published evidence and excludes the prohibited model family", async () => {
    const catalog = await buildRepositoryCatalog();
    const himloco = catalog.models.find((model) => model.id === "himloco");
    const cpp = himloco?.benchmarks.find((record) => record.id === "himloco-cpp-runtime-x5");
    expect(cpp?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 2853.09,
      unit: "fps",
      qualifier: "exact"
    }));
    const text = JSON.stringify(catalog).toLowerCase();
    expect(text).not.toContain(["yolo", "e"].join(""));
  });

  it("keeps plus-suffixed throughput as reported rather than inventing an exact value", async () => {
    const catalog = await buildRepositoryCatalog();
    const atto = catalog.models.find((model) => model.id === "convnext")
      ?.benchmarks.find((record) => record.variant_id === "convnext-atto-224");
    expect(atto?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 732,
      unit: "fps",
      qualifier: "lower-bound"
    }));
  });

  it("includes an exact FasterNet classification latency from the root documentation", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "fasternet")
      ?.benchmarks.find((benchmark) => benchmark.id === "fasternet-s-x5");
    expect(record?.performance).toContainEqual(expect.objectContaining({
      metric: "latency",
      value: 6.73,
      unit: "ms",
      qualifier: "exact"
    }));
  });

  it("includes the distinct FasterNet evaluator multi-thread latency", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "fasternet")
      ?.benchmarks.find((benchmark) => benchmark.id === "fasternet-s-multithread-x5");
    expect(record?.performance).toContainEqual(expect.objectContaining({
      metric: "latency",
      value: 24.34,
      unit: "ms",
      qualifier: "exact",
      scope: "multi-thread"
    }));
  });

  it("includes the published FasterNet float and quantized Top-1 pair", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "fasternet")
      ?.benchmarks.find((benchmark) => benchmark.id === "fasternet-s-x5");
    expect(record?.accuracy).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "top-1", value: 77.04, unit: "percent", model_stage: "float" }),
      expect.objectContaining({ metric: "top-1", value: 76.15, unit: "percent", model_stage: "quantized" })
    ]));
  });

  it("keeps Ultralytics YOLO BPU throughput scoped to its published thread count", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "ultralytics_yolo")
      ?.benchmarks.find((benchmark) => benchmark.id === "ultralytics-yolo11n-detect-x5");
    expect(record?.variant_id).toBe("yolo11n-detect-640");
    expect(record?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 188.9,
      unit: "fps",
      qualifier: "exact",
      concurrency: 2,
      scope: "BPU task"
    }));
  });

  it("keeps Ultralytics YOLO task variants distinct", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.find((model) => model.id === "ultralytics_yolo")?.benchmarks ?? [];
    expect(records.find((benchmark) => benchmark.id === "ultralytics-yolov8n-seg-x5"))
      .toEqual(expect.objectContaining({ variant_id: "yolov8n-seg-640" }));
    expect(records.find((benchmark) => benchmark.id === "ultralytics-yolov8n-pose-x5"))
      .toEqual(expect.objectContaining({ variant_id: "yolov8n-pose-640" }));
  });

  it("includes the explicit YOLO26 FP32 and BPU Python detection accuracy", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "ultralytics_yolo26")
      ?.benchmarks.find((benchmark) => benchmark.id === "ultralytics-yolo26n-detect-x5");
    expect(record?.accuracy).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "bbox-all-map-50-95", value: 0.319, unit: "ratio", model_stage: "float" }),
      expect.objectContaining({ metric: "bbox-all-map-50-95", value: 0.284, unit: "ratio", model_stage: "quantized" })
    ]));
  });

  it("includes FCOS evaluator timing for the distinct two-thread BPU condition", async () => {
    const catalog = await buildRepositoryCatalog();
    const record = catalog.models.find((model) => model.id === "fcos")
      ?.benchmarks.find((benchmark) => benchmark.id === "fcos-efficientnetb0-two-thread-x5");
    expect(record?.performance).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "latency", value: 6.2, unit: "ms", concurrency: 2 }),
      expect.objectContaining({ metric: "throughput", value: 323, unit: "fps", concurrency: 2 })
    ]));
  });

  it("omits UNet accuracy whose source does not publish a usable unit and environment", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.find((model) => model.id === "unet")?.benchmarks ?? [];
    expect(records).toHaveLength(1);
    expect(records[0]?.id).toBe("unet-resnet18-x5");
    expect(records[0]?.accuracy).toBeUndefined();
  });

  it("keeps CLIP empty when its sources publish no numeric benchmark", async () => {
    const catalog = await buildRepositoryCatalog();
    expect(catalog.models.find((model) => model.id === "clip")?.benchmarks).toEqual([]);
  });

  it("warns when published accuracy evidence omits its dataset", async () => {
    const onWarning = vi.fn();

    await buildRepositoryCatalog(onWarning);

    expect(onWarning).toHaveBeenCalledWith(expect.stringContaining("accuracy metrics have no published dataset"));
  });

  it("joins models and benchmarks and preserves the release tag", async () => {
    const catalog = await buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.valid.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    });
    expect(catalog.release.tag).toBe("x5-v1.0.0");
    expect(catalog.models[0]?.benchmarks[0]?.sample_id).toBe("convnext");
  });

  it("rejects source section text that is not an exact Markdown ATX heading", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.non-heading-section.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "SOURCE_SECTION_NOT_FOUND"
    }));
  });

  it("rejects a benchmark whose sample_id is absent from models.yaml", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/invalid-reference/", import.meta.url)),
      modelsPath: "models.yaml",
      benchmarksPath: "benchmarks.yaml",
      modelsSchemaPath: "../../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "UNKNOWN_SAMPLE"
    }));
  });

  it("rejects a benchmark whose source ref is a mutable branch", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.mutable-ref.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "INVALID_SOURCE_REF"
    }));
  });
});
