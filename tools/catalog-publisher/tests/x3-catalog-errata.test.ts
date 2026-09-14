// @vitest-environment node
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { applyX3CatalogErrata } from "../src/pipeline/x3-catalog-errata";
import type { BenchmarkRecord, MetricRecord } from "../src/catalog/types";

const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));
const X3_TAG = "x3-v1.1.2";
// Historical layout: the tag predates the docs/release move.
const PIN_BENCHMARKS_PATH = "release/benchmarks.yaml";
/** Literal CN heading over the detailed performance tables in the cited READMEs. */
const DETAILED_PERFORMANCE_SECTION = "## 性能数据";
/** X3 subsection heading that holds the detailed rows in all five cited files. */
const X3_TABLE_HEADING = "### RDK X3 & RDK X3 Module";

const CLASSIFICATION_IDS = [
  "googlenet-x3",
  "mobilenetv1-x3",
  "mobilenetv2-x3",
  "mobilenetv4-x3",
  "mobileone-x3",
  "repghost-x3",
  "repvgg-x3",
  "repvit-x3",
  "resnet-x3"
] as const;
type ClassificationId = (typeof CLASSIFICATION_IDS)[number];

const DETECTION_IDS = [
  "fcos-x3",
  "yolov5s-v2-x3",
  "yolov5x-v2-x3",
  "yolov5n-v7-x3",
  "yolov5s-v7-x3",
  "yolov5x-v7-x3",
  "yolov8n-x3",
  "yolov10n-x3",
  "yolov8n-seg-x3"
] as const;
type DetectionId = (typeof DETECTION_IDS)[number];

/** Section anchors recorded in the raw pin that resolve nowhere in the cited files. */
const STALE_SECTIONS: Record<DetectionId, string> = {
  "fcos-x3": "## Performance data",
  "yolov5s-v2-x3": "## Performance data",
  "yolov5x-v2-x3": "## Performance data",
  "yolov5n-v7-x3": "## Performance data",
  "yolov5s-v7-x3": "## Performance data",
  "yolov5x-v7-x3": "## Performance data",
  "yolov8n-x3": "### Object Detection",
  "yolov10n-x3": "### Object Detection",
  "yolov8n-seg-x3": "### Instance Segmentation"
};

/** Cited CN README paths; the errata must repoint sections without touching paths. */
const SOURCE_PATHS: Record<DetectionId, string> = {
  "fcos-x3": "demos/detect/FCOS/README_cn.md",
  "yolov5s-v2-x3": "demos/detect/YOLOv5/README_cn.md",
  "yolov5x-v2-x3": "demos/detect/YOLOv5/README_cn.md",
  "yolov5n-v7-x3": "demos/detect/YOLOv5/README_cn.md",
  "yolov5s-v7-x3": "demos/detect/YOLOv5/README_cn.md",
  "yolov5x-v7-x3": "demos/detect/YOLOv5/README_cn.md",
  "yolov8n-x3": "demos/detect/YOLOv8/README_cn.md",
  "yolov10n-x3": "demos/detect/YOLOv10/README_cn.md",
  "yolov8n-seg-x3": "demos/Instance_Segmentation/YOLOv8-Seg/README_cn.md"
};

/** Frame-rate FPS per classification record, straight from the pinned README tables. */
const EXPECTED_FRAME_RATE: Record<ClassificationId, number> = {
  "googlenet-x3": 243.51,
  "mobilenetv1-x3": 647.83,
  "mobilenetv2-x3": 890.99,
  "mobilenetv4-x3": 1309.17,
  "mobileone-x3": 455.87,
  "repghost-x3": 855.18,
  "repvgg-x3": 174.94,
  "repvit-x3": 96.47,
  "resnet-x3": 232.74
};

/** Top-1 accuracy per stage, straight from the pinned README tables. */
const EXPECTED_ACCURACY: Record<ClassificationId, { float: number; quantized: number }> = {
  "googlenet-x3": { float: 68.72, quantized: 67.71 },
  "mobilenetv1-x3": { float: 71.74, quantized: 65.36 },
  "mobilenetv2-x3": { float: 72.0, quantized: 68.17 },
  "mobilenetv4-x3": { float: 70.5, quantized: 70.26 },
  "mobileone-x3": { float: 72.0, quantized: 71.0 },
  "repghost-x3": { float: 72.5, quantized: 72.25 },
  "repvgg-x3": { float: 74.46, quantized: 62.78 },
  "repvit-x3": { float: 75.25, quantized: 75.75 },
  "resnet-x3": { float: 71.49, quantized: 70.5 }
};

interface X3BenchmarkDocument {
  release: { platform: string; tag: string };
  benchmarks: BenchmarkRecord[];
}

function loadRawX3Pin(): X3BenchmarkDocument {
  const text = execFileSync("git", ["-C", repositoryRoot, "show", `${X3_TAG}:${PIN_BENCHMARKS_PATH}`], { encoding: "utf8" });
  return parse(text) as X3BenchmarkDocument;
}

function recordById(benchmarks: BenchmarkRecord[], id: string): BenchmarkRecord {
  const record = benchmarks.find(candidate => candidate.id === id);
  if (!record) throw new Error(`record ${id} is missing from the x3 pin`);
  return record;
}

function frameRateMetric(record: BenchmarkRecord): MetricRecord | undefined {
  return (record.performance ?? []).find(metric =>
    metric.metric === "throughput" && metric.unit === "fps" && metric.scope === "frame rate");
}

function numericFingerprint(
  benchmarks: BenchmarkRecord[]
): Record<string, Array<Pick<MetricRecord, "metric" | "value" | "unit" | "qualifier" | "statistic" | "dataset">>> {
  const fingerprint: Record<string, Array<Pick<MetricRecord, "metric" | "value" | "unit" | "qualifier" | "statistic" | "dataset">>> = {};
  for (const record of benchmarks) {
    fingerprint[record.id] = [...(record.performance ?? []), ...(record.accuracy ?? [])].map(metric => ({
      metric: metric.metric,
      value: metric.value,
      unit: metric.unit,
      qualifier: metric.qualifier,
      statistic: metric.statistic,
      dataset: metric.dataset
    }));
  }
  return fingerprint;
}

interface Mutation {
  path: string;
  before?: unknown;
  after?: unknown;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function collectMutations(left: unknown, right: unknown, path: string, mutations: Mutation[]): void {
  if (left === right) return;
  if (isPlainObject(left) && isPlainObject(right)) {
    for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
      collectMutations(left[key], right[key], `${path}.${key}`, mutations);
    }
    return;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    if (left.length !== right.length) {
      mutations.push({ path, before: left, after: right });
      return;
    }
    for (let index = 0; index < left.length; index += 1) {
      collectMutations(left[index], right[index], `${path}[${index}]`, mutations);
    }
    return;
  }
  mutations.push({ path, before: left, after: right });
}

function mutationsByPath(mutations: Mutation[]): Record<string, { before?: unknown; after?: unknown }> {
  return Object.fromEntries(mutations.map(mutation => [mutation.path, { before: mutation.before, after: mutation.after }]));
}

function expectedMutations(snapshot: BenchmarkRecord[]): Record<string, { before?: unknown; after?: unknown }> {
  const expected: Record<string, { before?: unknown; after?: unknown }> = {};
  const indexOf = (id: string): number => {
    const index = snapshot.findIndex(record => record.id === id);
    if (index < 0) throw new Error(`record ${id} is missing from the x3 pin`);
    return index;
  };
  for (const id of CLASSIFICATION_IDS) {
    const index = indexOf(id);
    const record = recordById(snapshot, id);
    const frameRateIndex = (record.performance ?? []).findIndex(metric =>
      metric.metric === "throughput" && metric.unit === "fps" && metric.scope === "frame rate");
    const accuracy = record.accuracy ?? [];
    const floatIndex = accuracy.findIndex(metric => metric.scope === "floating-point");
    const quantizedIndex = accuracy.findIndex(metric => metric.scope === "quantized");
    expected[`benchmarks[${index}].performance[${frameRateIndex}].concurrency`] = { after: 4 };
    expected[`benchmarks[${index}].accuracy[${floatIndex}].scope`] = { before: "floating-point" };
    expected[`benchmarks[${index}].accuracy[${quantizedIndex}].scope`] = { before: "quantized" };
  }
  for (const id of DETECTION_IDS) {
    expected[`benchmarks[${indexOf(id)}].source.section`] = {
      before: STALE_SECTIONS[id],
      after: DETAILED_PERFORMANCE_SECTION
    };
  }
  return expected;
}

describe("applyX3CatalogErrata on the raw x3-v1.1.2 pin", () => {
  it("loads the immutable annotated tag with the audited raw state", () => {
    expect(execFileSync("git", ["-C", repositoryRoot, "cat-file", "-t", X3_TAG], { encoding: "utf8" }).trim()).toBe("tag");
    const pin = loadRawX3Pin();
    expect(pin.release.platform).toBe("x3");
    expect(pin.release.tag).toBe(X3_TAG);
    expect(pin.benchmarks).toHaveLength(20);
    // The raw pin ships exactly the defects the errata exists for: frame-rate
    // FPS without the documented thread count, stage-only accuracy scopes,
    // and section anchors that resolve nowhere in the cited files.
    for (const id of CLASSIFICATION_IDS) {
      const record = recordById(pin.benchmarks, id);
      const frameRate = frameRateMetric(record);
      if (!frameRate) throw new Error(`frame-rate metric missing for ${id}`);
      expect(frameRate.concurrency).toBeUndefined();
      expect((record.accuracy ?? []).map(metric => metric.scope)).toEqual(["floating-point", "quantized"]);
    }
    for (const id of DETECTION_IDS) {
      expect(recordById(pin.benchmarks, id).source.section).toBe(STALE_SECTIONS[id]);
    }
  });

  it("changes only the 36 source-backed labels; every value, unit, and qualifier is untouched", () => {
    const pin = loadRawX3Pin();
    const snapshot = JSON.parse(JSON.stringify(pin.benchmarks)) as BenchmarkRecord[];
    applyX3CatalogErrata(X3_TAG, pin.benchmarks);
    const mutations: Mutation[] = [];
    collectMutations(snapshot, pin.benchmarks, "benchmarks", mutations);
    expect(mutationsByPath(mutations)).toEqual(expectedMutations(snapshot));
  });

  it("leaves every numeric value, unit, qualifier, statistic, and dataset exactly as published", () => {
    const pin = loadRawX3Pin();
    const before = numericFingerprint(pin.benchmarks);
    applyX3CatalogErrata(X3_TAG, pin.benchmarks);
    expect(numericFingerprint(pin.benchmarks)).toEqual(before);
  });

  it("marks all nine classification frame-rate FPS as the documented 4-thread rate", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const actual = Object.fromEntries(CLASSIFICATION_IDS.map(id => {
      const frameRate = frameRateMetric(recordById(benchmarks, id));
      return [id, { concurrency: frameRate?.concurrency, fps: frameRate?.value, unit: frameRate?.unit }];
    }));
    const expected = Object.fromEntries(CLASSIFICATION_IDS.map(id => [
      id, { concurrency: 4, fps: EXPECTED_FRAME_RATE[id], unit: "fps" }
    ]));
    expect(actual).toEqual(expected);
  });

  it("keeps the multi-threaded latency thread count unknown and the single-threaded at one thread", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const actual = Object.fromEntries(CLASSIFICATION_IDS.map(id => {
      const performance = recordById(benchmarks, id).performance ?? [];
      const single = performance.find(metric => metric.metric === "latency" && metric.scope === "single-threaded");
      const multi = performance.find(metric => metric.metric === "latency" && metric.scope === "multi-threaded");
      return [id, {
        singleConcurrency: single?.concurrency,
        singleScope: single?.scope,
        multiConcurrency: multi?.concurrency,
        multiScope: multi?.scope
      }];
    }));
    const expected = Object.fromEntries(CLASSIFICATION_IDS.map(id => [id, {
      singleConcurrency: 1,
      singleScope: "single-threaded",
      multiConcurrency: undefined,
      multiScope: "multi-threaded"
    }]));
    expect(actual).toEqual(expected);
  });

  it("normalizes the stage-only accuracy scopes onto model_stage with values intact", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const actual = Object.fromEntries(CLASSIFICATION_IDS.map(id => {
      const accuracy = recordById(benchmarks, id).accuracy ?? [];
      return [id, accuracy.map(metric => ({
        metric: metric.metric,
        unit: metric.unit,
        value: metric.value,
        model_stage: metric.model_stage,
        scope: metric.scope
      }))];
    }));
    const expected = Object.fromEntries(CLASSIFICATION_IDS.map(id => [id, [
      { metric: "top-1", unit: "percent", value: EXPECTED_ACCURACY[id].float, model_stage: "float", scope: undefined },
      { metric: "top-1", unit: "percent", value: EXPECTED_ACCURACY[id].quantized, model_stage: "quantized", scope: undefined }
    ]]));
    expect(actual).toEqual(expected);
  });

  it("preserves non-stage scopes and stages on records outside the classification fix", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    // YOLOv5 v7.0 float mAP keeps its mAP:50-95 scope (not a stage-only scope).
    expect(recordById(benchmarks, "yolov5n-v7-x3").accuracy?.map(metric =>
      [metric.scope, metric.model_stage, metric.value, metric.dataset])).toEqual([["mAP:50-95", "float", 28.0, "COCO"]]);
    // The compound seg accuracy keeps both map kinds with their scopes.
    expect(recordById(benchmarks, "yolov8n-seg-x3").accuracy?.map(metric =>
      [metric.metric, metric.scope, metric.model_stage, metric.value])).toEqual([
      ["bbox-map", "mAP:50-95", "float", 36.7],
      ["mask-map", "mAP:50-95", "float", 30.5]
    ]);
    // PaddleOCR's dataset-table scope strings are untouched.
    expect(recordById(benchmarks, "paddleocr-det-x3").performance?.map(metric => metric.scope))
      .toEqual(["PP-OCRv3_det; dataset table: ICDAR2019-ArT"]);
  });

  it("repoints the nine detection records at the real detailed CN performance heading", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const actual = Object.fromEntries(DETECTION_IDS.map(id => {
      const record = recordById(benchmarks, id);
      return [id, { path: record.source.path, section: record.source.section }];
    }));
    const expected = Object.fromEntries(DETECTION_IDS.map(id => [
      id, { path: SOURCE_PATHS[id], section: DETAILED_PERFORMANCE_SECTION }
    ]));
    expect(actual).toEqual(expected);
    // The resolvable anchors of the remaining records are untouched.
    expect(recordById(benchmarks, "googlenet-x3").source.section).toBe("## 2. Model performance data");
    expect(recordById(benchmarks, "paddleocr-det-x3").source.section).toBe("## 2. Performance data");
    expect(recordById(benchmarks, "paddleocr-rec-x3").source.section).toBe("## 2. Performance data");
  });

  it("cites headings that literally exist in the pinned CN READMEs", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const headingsCache = new Map<string, string[]>();
    for (const id of DETECTION_IDS) {
      const record = recordById(benchmarks, id);
      let headings = headingsCache.get(record.source.path);
      if (headings === undefined) {
        const text = execFileSync("git", ["-C", repositoryRoot, "show", `${X3_TAG}:${record.source.path}`], { encoding: "utf8" });
        headings = text.split(/\r?\n/).filter(line => /^#{1,6} /.test(line));
        headingsCache.set(record.source.path, headings);
      }
      expect(headings, id).toContain(record.source.section);
      expect(headings, id).toContain(X3_TABLE_HEADING);
    }
  });

  it("is a no-op for any other release tag", () => {
    const rawPin = JSON.stringify(loadRawX3Pin().benchmarks);
    for (const tag of ["s-v1.1.2", "x5-v1.1.2", "x3-v1.1.1", "x3-v1.1.3"]) {
      const benchmarks = loadRawX3Pin().benchmarks;
      applyX3CatalogErrata(tag, benchmarks);
      expect(JSON.stringify(benchmarks), tag).toBe(rawPin);
    }
  });

  it("is idempotent when applied twice", () => {
    const benchmarks = loadRawX3Pin().benchmarks;
    applyX3CatalogErrata(X3_TAG, benchmarks);
    const appliedOnce = JSON.stringify(benchmarks);
    applyX3CatalogErrata(X3_TAG, benchmarks);
    expect(JSON.stringify(benchmarks)).toBe(appliedOnce);
  });
});
