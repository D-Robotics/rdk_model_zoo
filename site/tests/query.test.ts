import { describe, expect, it } from "vitest";
import {
  comparisonKey,
  queryModels,
  type CatalogQuery
} from "../src/catalog/query";
import {
  benchmarkFixture,
  catalogFilterFixtures,
  comparableAccuracyModels,
  comparablePerformanceModels,
  incompleteAccuracyRecord,
  incomparableModels,
  missingTargetMetricModel,
  createModelFixture,
  modelFixture
} from "./fixtures/catalog";

const baseQuery: CatalogQuery = {
  text: "",
  tasks: [],
  formats: [],
  precisions: [],
  benchmark: "all",
  sort: "name"
};

describe("queryModels", () => {
  it("matches names, ids, tasks, variants, and asset filenames case-insensitively", () => {
    const result = queryModels([modelFixture], {
      ...baseQuery,
      text: "ATTO"
    });
    expect(result.models.map((model) => model.id)).toEqual(["convnext"]);

    expect(queryModels([modelFixture], { ...baseQuery, text: "IMAGE-CLASSIFICATION" }).models)
      .toHaveLength(1);
    expect(queryModels([modelFixture], { ...baseQuery, text: "CONVNEXT" }).models)
      .toHaveLength(1);
    expect(queryModels([modelFixture], { ...baseQuery, text: "224" }).models)
      .toHaveLength(1);
    expect(queryModels([modelFixture], { ...baseQuery, text: "NV12.BIN" }).models)
      .toHaveLength(1);
  });

  it("normalizes compatibility forms before searching", () => {
    const result = queryModels([modelFixture], {
      ...baseQuery,
      text: "ＣｏｎｖＮｅｘｔ"
    });
    expect(result.models.map((model) => model.id)).toEqual(["convnext"]);
  });

  it("filters tasks, formats, precisions, and benchmark availability", () => {
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, tasks: ["OBJECT-DETECTION"] }).models.map((model) => model.id))
      .toEqual(["detector"]);
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, formats: ["ONNX"] }).models.map((model) => model.id))
      .toEqual(["detector"]);
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, precisions: ["FLOAT32"] }).models.map((model) => model.id))
      .toEqual(["detector"]);
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, benchmark: "performance" }).models.map((model) => model.id))
      .toEqual(["convnext"]);
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, benchmark: "accuracy" }).models.map((model) => model.id))
      .toEqual(["convnext", "detector"]);
    expect(queryModels(catalogFilterFixtures, { ...baseQuery, benchmark: "none" }).models.map((model) => model.id))
      .toEqual(["manual-model"]);
  });

  it("uses OR within a facet and AND across facets", () => {
    const result = queryModels(catalogFilterFixtures, {
      ...baseQuery,
      tasks: ["image-classification", "object-detection"],
      formats: ["onnx"],
      precisions: ["float32"]
    });
    expect(result.models.map((model) => model.id)).toEqual(["detector"]);
  });

  it("filters formats published only by a benchmark", () => {
    const result = queryModels([
      createModelFixture({
        id: "benchmark-format",
        name: "Benchmark format",
        assets: [{
          filename: "model.bin",
          format: "bin",
          url: "https://archive.example.test/model.bin",
          sha256: null
        }],
        benchmarks: [benchmarkFixture({ model_format: "onnx" })]
      })
    ], { ...baseQuery, formats: ["onnx"] });

    expect(result.models.map((model) => model.id)).toEqual(["benchmark-format"]);
  });

  it("sorts names deterministically and does not mutate the input array", () => {
    const models = [createModelFixture({ id: "zeta", name: "Zeta" }), createModelFixture({ id: "alpha", name: "Alpha" })];
    const result = queryModels(models, baseQuery);
    expect(result.models.map((model) => model.id)).toEqual(["alpha", "zeta"]);
    expect(models.map((model) => model.id)).toEqual(["zeta", "alpha"]);
    expect(result.sortApplied).toBe(true);
  });

  it("sorts comparable FPS metrics in descending order", () => {
    const result = queryModels(comparablePerformanceModels, {
      ...baseQuery,
      sort: "fps"
    });
    expect(result.models.map((model) => model.id)).toEqual(["fast", "slow"]);
    expect(result.sortApplied).toBe(true);
    expect(result.reason).toBeUndefined();
  });

  it("sorts comparable latency metrics in ascending order", () => {
    const result = queryModels([
      createModelFixture({
        id: "slower",
        name: "Slower",
        benchmarks: [benchmarkFixture({
          performance: [{
            metric: "latency",
            value: 5,
            unit: "ms",
            statistic: "mean",
            scope: "single-frame, single-thread, single-BPU-core",
            concurrency: 1
          }]
        })]
      }),
      createModelFixture({
        id: "faster",
        name: "Faster",
        benchmarks: [benchmarkFixture({
          performance: [{
            metric: "latency",
            value: 2,
            unit: "ms",
            statistic: "mean",
            scope: "single-frame, single-thread, single-BPU-core",
            concurrency: 1
          }]
        })]
      })
    ], { ...baseQuery, sort: "latency" });

    expect(result.models.map((model) => model.id)).toEqual(["faster", "slower"]);
    expect(result.sortApplied).toBe(true);
  });

  it.each(["lower-bound", "upper-bound", "approximate"] as const)(
    "does not numerically sort %s FPS metrics",
    (qualifier) => {
      const result = queryModels([
        createModelFixture({
          id: "alpha",
          name: "Alpha",
          benchmarks: [benchmarkFixture({
            performance: [{
              metric: "throughput",
              value: 100,
              unit: "fps",
              qualifier,
              scope: "four-thread concurrent",
              concurrency: 4
            }]
          })]
        }),
        createModelFixture({
          id: "beta",
          name: "Beta",
          benchmarks: [benchmarkFixture({
            performance: [{
              metric: "throughput",
              value: 300,
              unit: "fps",
              scope: "four-thread concurrent",
              concurrency: 4
            }]
          })]
        })
      ], { ...baseQuery, sort: "fps" });

      expect(result.models.map((model) => model.id)).toEqual(["alpha", "beta"]);
      expect(result.sortApplied).toBe(false);
      expect(result.reason).toBe("incomparable-benchmarks");
    }
  );

  it("sorts comparable accuracy metrics in descending order", () => {
    const result = queryModels(comparableAccuracyModels, {
      ...baseQuery,
      sort: "accuracy"
    });
    expect(result.models.map((model) => model.id)).toEqual(["higher-accuracy", "lower-accuracy"]);
    expect(result.sortApplied).toBe(true);
  });

  it("reports missing benchmarks when a requested metric is absent", () => {
    const result = queryModels([modelFixture, missingTargetMetricModel], {
      ...baseQuery,
      sort: "fps"
    });
    expect(result.sortApplied).toBe(false);
    expect(result.reason).toBe("missing-benchmarks");
    expect(result.models.map((model) => model.id)).toEqual(["convnext", "latency-only"]);
  });

  it("returns no comparison key when timing scope or concurrency is missing", () => {
    const record = benchmarkFixture({ environment: { hardware: "RDK X5" } });
    expect(comparisonKey(record, { metric: "throughput", value: 100, unit: "fps" })).toBeNull();
  });

  it("includes all comparable performance conditions in the key", () => {
    const record = benchmarkFixture();
    const metric = record.performance?.[1];
    expect(metric).toBeDefined();
    const key = comparisonKey(record, metric!);
    expect(key).toContain("rdk x5");
    expect(key).toContain("throughput");
    expect(key).toContain("fps");
    expect(key).toContain("four-thread concurrent");
    expect(key).toContain("224");
    expect(key).toContain("nchw");
    expect(key).toContain("nv12");
    expect(key).toContain("1");
  });

  it("requires dataset and model stage for accuracy comparison keys", () => {
    const record = benchmarkFixture();
    const metric = incompleteAccuracyRecord.accuracy?.[0];
    expect(metric).toBeDefined();
    expect(comparisonKey(record, metric!)).toBeNull();
  });

  it("does not sort incomparable performance records by their numeric values", () => {
    const result = queryModels(incomparableModels, {
      ...baseQuery,
      benchmark: "performance",
      sort: "fps"
    });
    expect(result.sortApplied).toBe(false);
    expect(result.reason).toBe("incomparable-benchmarks");
    expect(result.models.map((model) => model.id)).toEqual(["alpha", "beta"]);
  });
});
