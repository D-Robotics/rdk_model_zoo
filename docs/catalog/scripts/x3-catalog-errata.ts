import type { BenchmarkRecord } from "../src/catalog/types";

/** Release tag this errata is locked to; every other tag passes through unchanged. */
const X3_RELEASE_TAG = "x3-v1.1.2";

/**
 * Classification records whose shared README note 3 documents the thread
 * condition of the frame rate. The note is identical in all nine samples
 * (e.g. demos/classification/GoogLeNet/README.md line 48 and README_cn.md
 * line 49 at the pinned ref): the frame rate is a 4-thread project rate —
 * 4 threads simultaneously send tasks to the dual-core BPU. The
 * multi-threaded latency column states no thread count anywhere.
 */
const CLASSIFICATION_RECORD_IDS = new Set([
  "googlenet-x3",
  "mobilenetv1-x3",
  "mobilenetv2-x3",
  "mobilenetv4-x3",
  "mobileone-x3",
  "repghost-x3",
  "repvgg-x3",
  "repvit-x3",
  "resnet-x3"
]);

/**
 * Detection records whose source.section cites a heading that resolves in
 * neither the cited CN README nor its EN sibling. All five cited files head
 * their detailed performance tables with the same CN heading at the pinned
 * ref — FCOS README_cn.md line 87, YOLOv5 line 604, YOLOv8 line 736,
 * YOLOv10 line 527, YOLOv8-Seg line 821 — each followed by an
 * `### RDK X3 & RDK X3 Module` subsection holding the X3 rows.
 */
const SECTION_FIX_RECORD_IDS = new Set([
  "fcos-x3",
  "yolov5s-v2-x3",
  "yolov5x-v2-x3",
  "yolov5n-v7-x3",
  "yolov5s-v7-x3",
  "yolov5x-v7-x3",
  "yolov8n-x3",
  "yolov10n-x3",
  "yolov8n-seg-x3"
]);

/** Literal CN heading over the detailed performance table in the five cited files. */
const DETAILED_PERFORMANCE_SECTION = "## 性能数据";

/**
 * Source-backed label repairs for the historical X3 release. The published
 * tag stays immutable, so the fixes ride on the loaded benchmark records
 * instead of the YAML. Only measurement labels and provenance anchors move:
 * every numeric value, unit, and qualifier is passed through untouched.
 *
 * - The nine classification frame-rate FPS values gain the thread count their
 *   README note documents (4 threads). The multi-threaded latency keeps its
 *   thread count unknown — the source never states one, so none is invented.
 * - The classification accuracy columns 浮点精度/量化精度 (floating-point /
 *   quantized) are measurement stages. The manifest duplicated the stage into
 *   scope, which blocks float→quantized pairing; the stage moves onto
 *   model_stage and the stage-only scope is dropped. All other scopes
 *   (mAP:50-95, dataset tables, …) are preserved.
 * - The nine detection records listed above are repointed at the detailed
 *   performance heading that literally exists in their cited CN READMEs.
 */
export function applyX3CatalogErrata(tag: string, benchmarks: BenchmarkRecord[]): void {
  if (tag !== X3_RELEASE_TAG) return;
  for (const record of benchmarks) {
    if (CLASSIFICATION_RECORD_IDS.has(record.id)) applyClassificationConditions(record);
    else if (SECTION_FIX_RECORD_IDS.has(record.id)) applyCnPerformanceSection(record);
  }
}

function applyClassificationConditions(record: BenchmarkRecord): void {
  for (const metric of record.performance ?? []) {
    // Note 3's 4-thread condition belongs to the frame-rate throughput only;
    // the single-threaded latency already carries concurrency 1.
    if (metric.metric === "throughput" && metric.unit === "fps" && metric.scope === "frame rate") {
      metric.concurrency = 4;
    }
  }
  for (const metric of record.accuracy ?? []) {
    // Only a scope that merely restates the stage is stage-only; anything
    // richer (mAP:50-95, dataset tables, …) stays as recorded.
    if (metric.scope === "floating-point") {
      metric.model_stage = "float";
      delete metric.scope;
    } else if (metric.scope === "quantized") {
      metric.model_stage = "quantized";
      delete metric.scope;
    }
  }
}

function applyCnPerformanceSection(record: BenchmarkRecord): void {
  if (record.source.path.endsWith("README_cn.md")) {
    record.source.section = DETAILED_PERFORMANCE_SECTION;
  }
}
