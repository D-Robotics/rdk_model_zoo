// @vitest-environment node
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { platformBenchmarks, platformSlice, repositoryCatalog, repositoryRoot } from "./helpers/repository";
import type { BenchmarkRecord, MetricRecord } from "../src/catalog/types";

const EXPECTED_MODEL_COUNT = 36;
const EXPECTED_BENCHMARK_RECORD_COUNT = 239;
const EXPECTED_PERFORMANCE_METRIC_COUNT = 636;
const EXPECTED_ACCURACY_METRIC_COUNT = 419;
// YOLO26 rows resolve under the unified ultralytics_yolo sample (the separate
// yolo26 directory no longer exists on develop), and the OCR pilot lives at
// samples/vision/paddle_ocr; the audited directory count shrank accordingly.
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
  ["paddleocr", "samples/vision/paddle_ocr"],
  ["pp_liteseg", "samples/vision/pp_liteseg"],
  ["repghost", "samples/vision/repghost"],
  ["repvgg", "samples/vision/repvgg"],
  ["repvit", "samples/vision/repvit"],
  ["resnet", "samples/vision/resnet"],
  ["resnext", "samples/vision/resnext"],
  ["ultralytics_yolo", "samples/vision/ultralytics_yolo"],
  ["unet", "samples/vision/unet"],
  ["vargconvnet", "samples/vision/vargconvnet"],
  ["yolo26_depth", "samples/vision/yolo26_depth"],
  ["yoloe", "samples/vision/yoloe"],
  ["yolov5", "samples/vision/yolov5"],
  ["yoloworld", "samples/vision/yoloworld"]
]);
const EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS = new Set([
  "clip",
  "vargconvnet",
  "yoloworld"
]);
/**
 * Two RDK X3 paddleocr rows were historically carried inside the X5 manifest.
 * They belong to the X3 distribution, which publishes them from its own sample
 * tree, so an X5-only count is short by exactly these two records.
 */
const CROSS_PUBLISHED_ON_X3 = ["paddleocr-det-x3", "paddleocr-rec-x3"];

/**
 * The audited inventory is the X5 release this suite was written against. The
 * migration keeps it as the X5 slice of the multi-platform catalog, so every
 * assertion below reads the `platforms/x5` distribution through the same
 * pipeline the published artifact uses.
 */
function buildRepositoryCatalog(): Promise<import("../src/catalog/types").Catalog> {
  return repositoryCatalog();
}

function x5Records(catalog: Awaited<ReturnType<typeof repositoryCatalog>>) {
  return platformBenchmarks(catalog, "x5");
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
    // A family card merges the sample directories that share a family (the four
    // MobileNet revisions, YOLOv5, YOLO26-Depth), so the card's own sample_path
    // is only a representative. The audited unit is the sample directory a
    // resolved variant belongs to, which every merge preserves.
    const x5Variants = catalog.models.flatMap((model) => model.variants ?? [])
      .filter((variant) => variant.hardware === "x5");
    const samplePaths = new Set(x5Variants.map((variant) => variant.sample_path));
    const measuredPaths = new Set(
      x5Variants.filter((variant) => variant.benchmarks.length > 0).map((variant) => variant.sample_path)
    );
    const expectedPaths = [...EXPECTED_MODEL_PATHS.values()];

    expect(samplePaths.size).toBe(EXPECTED_MODEL_COUNT);
    expect([...samplePaths].sort()).toEqual([...expectedPaths].sort());
    expect(expectedPaths.filter((path) => !measuredPaths.has(path)).sort()).toEqual(
      [...EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS].map((id) => EXPECTED_MODEL_PATHS.get(id)!).sort()
    );
  });

  it("keeps benchmark identifiers and semantic evidence unique", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = x5Records(catalog);

    expect(duplicates(records.map((record) => record.id))).toEqual([]);
    expect(duplicates(records.map(recordKey))).toEqual([]);
    expect(duplicates(records.flatMap(provenanceAwareMetricKeys))).toEqual([]);
    expect(duplicates(records.flatMap(sourceIndependentEvidenceKeys))).toEqual([]);
  });

  it("locks the audited totals and requires evidence in every record", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = x5Records(catalog);
    const performanceMetrics = records.flatMap((record) => record.performance ?? []);
    const accuracyMetrics = records.flatMap((record) => record.accuracy ?? []);

    // The X5 manifest historically also carried two RDK X3 paddleocr rows. They
    // stay in the catalog, published by the X3 distribution that owns them, so
    // the audited totals survive the split without being counted twice.
    expect(records).toHaveLength(EXPECTED_BENCHMARK_RECORD_COUNT - CROSS_PUBLISHED_ON_X3.length);
    expect(performanceMetrics).toHaveLength(EXPECTED_PERFORMANCE_METRIC_COUNT - CROSS_PUBLISHED_ON_X3.length);
    expect(accuracyMetrics).toHaveLength(EXPECTED_ACCURACY_METRIC_COUNT);
    expect(records.every((record) => (record.performance?.length ?? 0) + (record.accuracy?.length ?? 0) > 0)).toBe(true);

    const crossPublished = platformBenchmarks(catalog, "x3").filter(
      (record) => CROSS_PUBLISHED_ON_X3.includes(record.id)
    );
    expect(crossPublished.map((record) => record.id).sort()).toEqual([...CROSS_PUBLISHED_ON_X3].sort());
    expect(records.length + crossPublished.length).toBe(EXPECTED_BENCHMARK_RECORD_COUNT);
  });

  it("keeps the audited canonical source corrections", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = x5Records(catalog);
    const mobileNetV4 = records.filter(
      (record) => record.source.path === "samples/vision/mobilenetv4/evaluator/README.md"
    );
    const fcosRootRecords = records.filter((record) => record.source.path === "samples/vision/fcos/README.md");
    const unet = records.find((record) => record.source.path.startsWith("samples/vision/unet/"));

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

  it("keeps immutable sources resolvable and includes the completed YOLOE sample", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = x5Records(catalog);
    const x5Tag = catalog.release.platform_tags!.x5;

    expect(records.every((record) => record.source.ref === x5Tag || /^[a-f0-9]{40}$/i.test(record.source.ref))).toBe(true);
    for (const ref of new Set(records.map(record => record.source.ref))) {
      if (ref === x5Tag) continue;
      expect(execFileSync("git", ["-C", repositoryRoot, "cat-file", "-t", ref], { encoding: "utf8" }).trim()).toBe("commit");
    }
    expect(records.every((record) => record.source.path.length > 0 && record.source.section.length > 0)).toBe(true);

    const yoloe = platformSlice(catalog, "x5", "yoloe");
    expect(yoloe.entry?.assets).toHaveLength(3);
    expect(yoloe.entry?.assets.every(asset => /^[a-f0-9]{64}$/.test(asset.sha256 ?? ""))).toBe(true);
    expect(yoloe.benchmarks.flatMap(record => record.performance ?? [])).toHaveLength(12);
    expect(yoloe.benchmarks.every(record => record.performance?.every(metric => metric.concurrency === 1))).toBe(true);
  });
});
