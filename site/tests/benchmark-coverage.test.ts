// @vitest-environment node
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildCatalog } from "../scripts/catalog-builder";
import type { BenchmarkRecord, MetricRecord } from "../src/catalog/types";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const EXPECTED_MODEL_COUNT = 36;
const EXPECTED_BENCHMARK_RECORD_COUNT = 227;
const EXPECTED_PERFORMANCE_METRIC_COUNT = 623;
const EXPECTED_ACCURACY_METRIC_COUNT = 369;
const EXPECTED_MODEL_PATHS = new Map([
  ["clip", "samples/vision/clip"],
  ["convnext", "samples/vision/convnext"],
  ["edgenext", "samples/vision/edgenext"],
  ["efficient_sam", "samples/vision/efficient_sam"],
  ["efficientformer", "samples/vision/efficientformer"],
  ["efficientformerv2", "samples/vision/efficientformerv2"],
  ["efficientnet", "samples/vision/efficientnet"],
  ["efficientvit", "samples/vision/efficientvit"],
  ["fasternet", "samples/vision/fasternet"],
  ["fastvit", "samples/vision/fastvit"],
  ["fcos", "samples/vision/fcos"],
  ["googlenet", "samples/vision/googlenet"],
  ["hgnetv2", "samples/vision/hgnetv2"],
  ["himloco", "samples/robotics/himloco"],
  ["lprnet", "samples/vision/lprnet"],
  ["mobile_sam", "samples/vision/mobile_sam"],
  ["mobilenetv1", "samples/vision/mobilenetv1"],
  ["mobilenetv2", "samples/vision/mobilenetv2"],
  ["mobilenetv3", "samples/vision/mobilenetv3"],
  ["mobilenetv4", "samples/vision/mobilenetv4"],
  ["mobileone", "samples/vision/mobileone"],
  ["modnet", "samples/vision/modnet"],
  ["paddleocr", "samples/vision/paddleocr"],
  ["pp_liteseg", "samples/vision/pp_liteseg"],
  ["repghost", "samples/vision/repghost"],
  ["repvgg", "samples/vision/repvgg"],
  ["repvit", "samples/vision/repvit"],
  ["resnet", "samples/vision/resnet"],
  ["resnext", "samples/vision/resnext"],
  ["ultralytics_yolo", "samples/vision/ultralytics_yolo"],
  ["ultralytics_yolo26", "samples/vision/ultralytics_yolo26"],
  ["unet", "samples/vision/unet"],
  ["vargconvnet", "samples/vision/vargconvnet"],
  ["yolo26_depth", "samples/vision/yolo26_depth"],
  ["yolov5", "samples/vision/yolov5"],
  ["yoloworld", "samples/vision/yoloworld"]
]);
const EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS = new Set([
  "clip",
  "vargconvnet",
  "yoloworld"
]);

function buildRepositoryCatalog() {
  return buildCatalog({
    repositoryRoot,
    modelsPath: "release/models.yaml",
    benchmarksPath: "release/benchmarks.yaml",
    modelsSchemaPath: "release/schemas/models.schema.json",
    benchmarksSchemaPath: "release/schemas/benchmarks.schema.json"
  });
}

function evidenceContext(record: BenchmarkRecord) {
  return [
    record.sample_id,
    record.variant_id,
    record.asset_filename ?? null,
    record.model_format ?? null,
    record.precision ?? null,
    record.input?.shape ?? null,
    record.input?.layout ?? null,
    record.input?.format ?? null,
    record.environment.hardware,
    record.environment.rdk_os ?? null,
    record.environment.runtime ?? null,
    record.environment.cpu_mode ?? null,
    record.environment.bpu_cores ?? null
  ];
}

function provenanceContext(record: BenchmarkRecord) {
  return [
    record.source.ref,
    record.source.path,
    record.source.section,
    ...evidenceContext(record)
  ];
}

function metricContext(kind: "performance" | "accuracy", metric: MetricRecord) {
  return [
    kind,
    metric.metric,
    metric.unit,
    metric.value,
    metric.qualifier ?? null,
    metric.statistic ?? null,
    metric.model_stage ?? null,
    metric.dataset ?? null,
    metric.concurrency ?? null,
    metric.scope ?? null
  ];
}

function provenanceAwareMetricKeys(record: BenchmarkRecord) {
  const context = provenanceContext(record);
  return (["performance", "accuracy"] as const).flatMap((kind) =>
    (record[kind] ?? []).map((metric) => JSON.stringify([...context, ...metricContext(kind, metric)]))
  );
}

function sourceIndependentEvidenceKeys(record: BenchmarkRecord) {
  const context = evidenceContext(record);
  return (["performance", "accuracy"] as const).flatMap((kind) =>
    (record[kind] ?? []).map((metric) => JSON.stringify([...context, ...metricContext(kind, metric)]))
  );
}

function recordKey(record: BenchmarkRecord) {
  return JSON.stringify([
    ...provenanceContext(record),
    ...provenanceAwareMetricKeys(record).sort()
  ]);
}

function duplicates(values: string[]) {
  const seen = new Set<string>();
  const repeated = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      repeated.add(value);
    }
    seen.add(value);
  }
  return [...repeated];
}

describe("audited benchmark coverage", () => {
  it("accounts for the complete release inventory", async () => {
    const catalog = await buildRepositoryCatalog();
    const emptyModelIds = catalog.models
      .filter((model) => model.benchmarks.length === 0)
      .map((model) => model.id)
      .sort();
    const expectedEmptyModelIds = [...EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS].sort();

    expect(catalog.models).toHaveLength(EXPECTED_MODEL_COUNT);
    expect(new Map(catalog.models.map((model) => [model.id, model.sample_path]))).toEqual(EXPECTED_MODEL_PATHS);
    expect(emptyModelIds).toEqual(expectedEmptyModelIds);
    expect(catalog.models.filter(
      (model) => model.benchmarks.length === 0 && !EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS.has(model.id)
    )).toEqual([]);
  });

  it("keeps benchmark identifiers and semantic evidence unique", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.flatMap((model) => model.benchmarks);

    expect(duplicates(records.map((record) => record.id))).toEqual([]);
    expect(duplicates(records.map(recordKey))).toEqual([]);
    expect(duplicates(records.flatMap(provenanceAwareMetricKeys))).toEqual([]);
    expect(duplicates(records.flatMap(sourceIndependentEvidenceKeys))).toEqual([]);
  });

  it("locks the audited totals and requires evidence in every record", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.flatMap((model) => model.benchmarks);
    const performanceMetrics = records.flatMap((record) => record.performance ?? []);
    const accuracyMetrics = records.flatMap((record) => record.accuracy ?? []);

    expect(records).toHaveLength(EXPECTED_BENCHMARK_RECORD_COUNT);
    expect(performanceMetrics).toHaveLength(EXPECTED_PERFORMANCE_METRIC_COUNT);
    expect(accuracyMetrics).toHaveLength(EXPECTED_ACCURACY_METRIC_COUNT);
    expect(records.every((record) => (record.performance?.length ?? 0) + (record.accuracy?.length ?? 0) > 0)).toBe(true);
  });

  it("keeps the audited canonical source corrections", async () => {
    const catalog = await buildRepositoryCatalog();
    const mobileNetV4 = catalog.models.find((model) => model.id === "mobilenetv4")?.benchmarks ?? [];
    const fcosRootRecords = (catalog.models.find((model) => model.id === "fcos")?.benchmarks ?? [])
      .filter((record) => record.source.path === "samples/vision/fcos/README.md");
    const unet = catalog.models.find((model) => model.id === "unet")?.benchmarks[0];

    expect(mobileNetV4).toHaveLength(2);
    expect(mobileNetV4.every(
      (record) => record.source.path === "samples/vision/mobilenetv4/evaluator/README.md"
    )).toBe(true);
    expect(mobileNetV4.find((record) => record.variant_id === "mobilenetv4-conv-medium-224")?.performance)
      .toEqual(expect.arrayContaining([
        expect.objectContaining({ metric: "latency", value: 2.42, unit: "ms", concurrency: 1 }),
        expect.objectContaining({ metric: "throughput", value: 572.36, unit: "fps", qualifier: "exact" })
      ]));

    expect(fcosRootRecords).toHaveLength(3);
    expect(fcosRootRecords.every(
      (record) => record.performance?.length === 1 && record.performance[0]?.metric === "post_process_latency"
    )).toBe(true);

    expect(unet?.source.section).toBe("## Reference Results");
    expect(unet?.performance?.every(
      (metric) => metric.scope?.includes("historical earlier ResNet18 checkpoint")
        && metric.scope.includes("current board revalidation pending")
    )).toBe(true);
  });

  it("keeps immutable sources resolvable and the prohibited family absent", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.flatMap((model) => model.benchmarks);

    expect(records.every((record) => record.source.ref === catalog.release.tag)).toBe(true);
    expect(records.every((record) => record.source.path.length > 0 && record.source.section.length > 0)).toBe(true);

    const prohibitedFamily = ["yolo", "e"].join("");
    expect(JSON.stringify(catalog).toLowerCase()).not.toContain(prohibitedFamily);
  });
});
