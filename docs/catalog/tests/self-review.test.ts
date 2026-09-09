// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { groupPerformanceMetrics, pairAccuracyMetrics } from "../src/catalog/metric-display";
import { variantRecords } from "../src/ui/variant-benchmark-table";
import { stripHardwareSuffix } from "../src/catalog/model-naming";
import type { BenchmarkRecord, HardwareId, MetricRecord } from "../src/catalog/types";
import { benchmarkFixture } from "./fixtures/catalog";

const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));

function perf(overrides: Partial<MetricRecord>): MetricRecord {
  return { metric: "latency", value: 1, unit: "ms", ...overrides };
}

function record(overrides: Partial<BenchmarkRecord>): BenchmarkRecord {
  return benchmarkFixture({
    variant_id: "yolov8n-detect-640",
    display_name: "YOLOv8n Detect",
    asset_filename: "yolov8n_detect_640_nv12.bin",
    ...overrides
  });
}

describe("performance row grouping", () => {
  it("folds the thread-wording synonyms into shared rows and keeps every value visible", () => {
    const groups = groupPerformanceMetrics([record({ performance: [
      perf({ metric: "latency", value: 8.8, scope: undefined }),
      perf({ metric: "throughput", value: 113.35, scope: undefined, unit: "fps" }),
      perf({ metric: "latency", value: 32.31, scope: "multi-thread" })
    ] })]);
    // The scope-less latency and FPS share one row; the multi-thread latency
    // cannot share the same unknown-thread bucket, so it keeps its own row
    // instead of being hidden.
    const paired = groups.filter((group) => group.threads.some((thread) => thread.throughput));
    expect(paired).toHaveLength(1);
    const thread = paired[0]!.threads.find((candidate) => candidate.concurrency === undefined);
    expect(thread?.latency?.metric.value).toBe(8.8);
    expect(thread?.throughput?.metric.value).toBe(113.35);
    const latencies = groups.flatMap((group) => group.threads.map((candidate) => candidate.latency?.metric.value));
    expect(latencies).toContain(32.31);
  });

  it("never hides a second latency that cannot share a thread bucket", () => {
    const groups = groupPerformanceMetrics([record({ performance: [
      perf({ metric: "latency", value: 1.96, scope: "single-thread", concurrency: 1 }),
      perf({ metric: "latency", value: 3.29, scope: "multi-thread" }),
      perf({ metric: "throughput", value: 902.09, scope: "frame rate", unit: "fps" })
    ] })]);
    const latencies = groups.flatMap((group) => group.threads.map((thread) => thread.latency?.metric.value));
    expect(latencies).toContain(1.96);
    expect(latencies).toContain(3.29);
    expect(groups.some((group) => group.threads.some((thread) => thread.throughput?.metric.value === 902.09)))
      .toBe(true);
  });

  it("merges an unstated statistic with the mean statistic of the same scope", () => {
    const groups = groupPerformanceMetrics([record({ performance: [
      perf({ metric: "latency", value: 34.426, scope: "100 frames", statistic: "mean" }),
      perf({ metric: "throughput", value: 29.008, scope: "100 frames", unit: "fps", statistic: undefined })
    ] })]);
    expect(groups).toHaveLength(1);
    expect(groups[0]!.threads[0]!.latency?.metric.value).toBe(34.426);
    expect(groups[0]!.threads[0]!.throughput?.metric.value).toBe(29.008);
  });

  it("keeps separate rows for genuinely different contexts such as encoder and decoder", () => {
    const groups = groupPerformanceMetrics([record({ performance: [
      perf({ metric: "latency", value: 11.78, scope: "encoder", concurrency: 1 }),
      perf({ metric: "latency", value: 3.25, scope: "decoder", concurrency: 1 })
    ] })]);
    expect(groups).toHaveLength(2);
  });

  it("keeps post-processing out of the BPU timing rows", () => {
    const groups = groupPerformanceMetrics([record({ performance: [
      perf({ metric: "latency", value: 6.3, scope: "BPU task", concurrency: 1 }),
      perf({ metric: "post_process_latency", value: 5, scope: "single-core CPU" })
    ] })]);
    const timing = groups.filter((group) => group.threads.some((thread) => thread.latency || thread.throughput));
    expect(timing).toHaveLength(1);
    expect(timing[0]!.threads[0]!.latency?.metric.value).toBe(6.3);
  });
});

describe("self-review of the published catalog", () => {
  it("records the per-thread X3 measurements for the three summary-only YOLO records", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolov8 = catalog.models.find((model) => model.id === "yolov8")!;
    const variant = yolov8.variants?.find((candidate) => candidate.hardware === "x3"
      && candidate.task === "object-detection")!;
    const groups = groupPerformanceMetrics(variantRecords(variant, "x3" as HardwareId))
      .filter((group) => group.threads.some((thread) => thread.latency || thread.throughput));
    expect(groups).toHaveLength(1);
    const byConcurrency = new Map(groups[0]!.threads.map((thread) => [thread.concurrency, thread]));
    expect(byConcurrency.get(1)?.latency?.metric.value).toBe(99.8);
    expect(byConcurrency.get(1)?.throughput?.metric.value).toBe(10);
    expect(byConcurrency.get(2)?.latency?.metric.value).toBe(102);
    expect(byConcurrency.get(4)?.throughput?.metric.value).toBe(30.2);
    expect(byConcurrency.get(8)?.latency?.metric.value).toBe(231);
    expect(byConcurrency.get(8)?.throughput?.metric.value).toBe(34.1);
  });

  it("does not duplicate X3 rows through the X5 manifest's cross-hardware records", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const paddleocr = catalog.models.find((model) => model.id === "paddleocr")!;
    const x3Variants = (paddleocr.variants ?? []).filter((variant) => variant.hardware === "x3");
    expect(x3Variants).toHaveLength(2);
    const perfEntries = x3Variants.flatMap((variant) =>
      variantRecords(variant, "x3" as HardwareId).flatMap((record) => record.performance ?? []));
    expect(perfEntries).toHaveLength(2);
    expect(new Set(perfEntries.map((entry) => entry.value))).toEqual(new Set([41.96, 78.92]));
  });

  it("publishes the SigLIP float baseline and MSE beside the BPU values", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const siglip = catalog.models.find((model) => model.id === "siglip")!;
    const variant = siglip.variants?.find((candidate) => candidate.hardware === "s100"
      && candidate.id.includes("base-patch16-224"))!;
    const pairs = pairAccuracyMetrics(variantRecords(variant, "s100" as HardwareId));
    const byCanonical = new Map(pairs.map((pair) => [pair.canonicalMetric, pair]));
    expect(byCanonical.get("top-1")?.float?.metric.value).toBe(0.7123);
    expect(byCanonical.get("top-1")?.other?.metric.value).toBe(0.7118);
    expect(byCanonical.get("top-5")?.float?.metric.value).toBe(0.9143);
    expect(byCanonical.get("mse")?.other?.metric.value).toBe(0.087);
    expect(byCanonical.get("cosine-similarity")?.other?.metric.value).toBe(0.991);
  });

  it("spells row names consistently across platforms", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const rowNames = new Set<string>();
    for (const model of catalog.models) {
      for (const variant of model.variants ?? []) {
        rowNames.add(stripHardwareSuffix(variant.name || variant.id));
      }
    }
    expect([...rowNames].some((name) => /Resnet\d/.test(name))).toBe(false);
    expect([...rowNames].some((name) => /YOLOv?\d+\w*-(CLS|Seg|Pose|Obb)/.test(name))).toBe(false);
    expect([...rowNames].some((name) => /\bCls\b/.test(name))).toBe(false);
    expect([...rowNames].some((name) => name.includes("PP-OCRv3_det"))).toBe(false);

    const siglip = catalog.models.find((model) => model.id === "siglip")!;
    const names = (siglip.variants ?? []).filter((variant) => variant.id.includes("base-patch16-224"))
      .map((variant) => stripHardwareSuffix(variant.name || variant.id));
    expect(new Set(names).size).toBe(1);
  });

  it("carries the S post-processing column as evidence, not as a BPU latency row", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const yolov8 = catalog.models.find((model) => model.id === "yolov8")!;
    const variant = yolov8.variants?.find((candidate) => candidate.hardware === "s600"
      && candidate.task === "object-detection" && candidate.id.includes("v8n"))!;
    const records = variantRecords(variant, "s600" as HardwareId);
    const post = records.flatMap((record) => record.performance ?? [])
      .filter((entry) => entry.metric === "post_process_latency");
    expect(post).toHaveLength(1);
    expect(post[0]!.value).toBe(2);
    const groups = groupPerformanceMetrics(records)
      .filter((group) => group.threads.some((thread) => thread.latency || thread.throughput));
    // BPU task timing only: the post-processing entry must not add a row.
    expect(groups).toHaveLength(1);
  });
});
