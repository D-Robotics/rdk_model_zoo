import { describe, expect, it } from "vitest";
import {
  deriveRetention,
  formatMetricValue,
  groupPerformanceMetrics,
  isHigherBetterMetric,
  pairAccuracyMetrics
} from "../src/catalog/metric-display";
import type { BenchmarkRecord, MetricRecord } from "../src/catalog/types";
import { benchmarkFixture } from "./fixtures/catalog";

function metric(overrides: Partial<MetricRecord>): MetricRecord {
  return {
    metric: "bbox-all-map-50-95",
    value: 0.306,
    unit: "ratio",
    dataset: "COCO2017",
    model_stage: "float",
    ...overrides
  };
}

function record(overrides: Partial<BenchmarkRecord>): BenchmarkRecord {
  return benchmarkFixture({
    variant_id: "yolov8n-detect-640",
    display_name: "YOLOv8n Detect",
    asset_filename: "yolov8n_detect_640_nv12.bin",
    ...overrides
  });
}

describe("metric display", () => {
  it("derives accuracy retention as a percentage without capping values", () => {
    expect(deriveRetention(metric({ value: 0.306 }), metric({ value: 0.292, model_stage: "quantized" })))
      .toBeCloseTo(95.424836, 5);
    expect(formatMetricValue({ metric: "retention", value: 95.42, unit: "percent" }, "en"))
      .toBe("95.42%");
    expect(deriveRetention(metric({ value: 0.1 }), metric({ value: 0.2, model_stage: "quantized" })))
      .toBe(200);
  });

  it("prefers an explicit retention metric and keeps wrong datasets separate", () => {
    const pairs = pairAccuracyMetrics([
      record({ accuracy: [
        metric({ model_stage: "float" }),
        metric({ value: 0.292, model_stage: "quantized", dataset: "COCO2017" }),
        metric({ metric: "bbox-all-map-50-95-retention", value: 94, unit: "percent", model_stage: undefined })
      ] }),
      record({
        id: "wrong-dataset",
        accuracy: [metric({ value: 0.291, model_stage: "quantized", dataset: "ImageNet" })]
      })
    ]);

    const coco = pairs.find((pair) => pair.dataset === "COCO2017");
    expect(coco?.retention?.metric.metric).toBe("bbox-all-map-50-95-retention");
    expect(coco?.retentionSource).toBe("explicit");
    expect(pairs.some((pair) => pair.dataset === "ImageNet" && pair.retentionValue === undefined)).toBe(true);
  });

  it("attaches a generic explicit retention record when exactly one accuracy pair matches", () => {
    const pairs = pairAccuracyMetrics([record({ accuracy: [
      metric({ model_stage: "float" }),
      metric({ value: 0.292, model_stage: "quantized" }),
      { metric: "retention", value: 94, unit: "percent", dataset: "COCO2017" }
    ] })]);

    expect(pairs).toHaveLength(1);
    expect(pairs[0]?.retentionSource).toBe("explicit");
    expect(pairs[0]?.retentionValue).toBe(94);
  });

  it("does not derive retention for errors or a zero float denominator", () => {
    expect(deriveRetention(
      metric({ metric: "mae", unit: "mae", value: 1 }),
      metric({ metric: "mae", unit: "mae", value: 0.5, model_stage: "quantized" })
    )).toBeUndefined();
    expect(deriveRetention(metric({ value: 0 }), metric({ value: 0.5, model_stage: "quantized" }))).toBeUndefined();
    expect(isHigherBetterMetric(metric({ metric: "wer", value: 0.1 }))).toBe(false);
    expect(isHigherBetterMetric(metric({ metric: "loss", value: 0.1 }))).toBe(false);
  });

  it("marks mismatched accuracy scopes as not comparable", () => {
    const pairs = pairAccuracyMetrics([record({ accuracy: [
      metric({ model_stage: "float", scope: "BPU" }),
      metric({ value: 0.292, model_stage: "quantized", scope: "CPU" })
    ] })]);

    expect(pairs).toHaveLength(2);
    expect(pairs.every((pair) => pair.retentionStatus === "not-comparable")).toBe(true);
  });

  it("groups unknown concurrency and separates X3 post-processing from BPU latency", () => {
    const groups = groupPerformanceMetrics([record({
      environment: { hardware: "RDK X3" },
      performance: [
        { metric: "throughput", value: 34.1, unit: "fps", scope: "BPU throughput" },
        { metric: "post_process_latency", value: 6, unit: "ms", scope: "Python post-process" }
      ]
    })]);

    expect(groups).toHaveLength(2);
    const bpu = groups.find((group) => group.scope === "BPU throughput");
    expect(bpu?.threads[0]?.concurrency).toBeUndefined();
    expect(bpu?.threads[0]?.throughput?.metric.metric).toBe("throughput");
    const postProcess = groups.find((group) => group.scope === "Python post-process");
    expect(postProcess?.threads[0]?.latency).toBeUndefined();
    expect(postProcess?.metrics[0]?.metric.metric).toBe("post_process_latency");
  });
});
