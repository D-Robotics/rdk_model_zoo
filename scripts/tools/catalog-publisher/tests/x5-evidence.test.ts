// @vitest-environment node
import { describe, expect, it } from "vitest";
import { platformSlice, repositoryCatalog } from "./helpers/repository";

/**
 * The X5 release evidence that the audited catalog was built against, read back
 * through the multi-platform pipeline. Each value is a transcription of a
 * number the X5 documentation publishes; none of them may be normalised,
 * rounded, or promoted to a stronger qualifier than the source states.
 */
describe("X5 published evidence", () => {
  it("keeps the exact C++ runtime throughput and the completed YOLOE family", async () => {
    const catalog = await repositoryCatalog();
    const himloco = platformSlice(catalog, "x5", "himloco");
    const cpp = himloco.benchmarks.find((record) => record.id === "himloco-cpp-runtime-x5");

    expect(cpp?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 2853.09,
      unit: "fps",
      qualifier: "exact"
    }));
    expect(platformSlice(catalog, "x5", "yoloe").benchmarks).toHaveLength(3);
  });

  it("keeps plus-suffixed throughput as reported rather than inventing an exact value", async () => {
    const catalog = await repositoryCatalog();
    const atto = platformSlice(catalog, "x5", "convnext").benchmarks
      .find((record) => record.variant_id === "convnext-atto-224");

    expect(atto?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 732,
      unit: "fps",
      qualifier: "lower-bound"
    }));
  });

  it("keeps the FasterNet single- and multi-thread latencies distinct", async () => {
    const records = platformSlice(await repositoryCatalog(), "x5", "fasternet").benchmarks;

    expect(records.find((record) => record.id === "fasternet-s-x5")?.performance)
      .toContainEqual(expect.objectContaining({ metric: "latency", value: 6.73, unit: "ms", qualifier: "exact" }));
    expect(records.find((record) => record.id === "fasternet-s-multithread-x5")?.performance)
      .toContainEqual(expect.objectContaining({
        metric: "latency",
        value: 24.34,
        unit: "ms",
        qualifier: "exact",
        scope: "multi-thread"
      }));
  });

  it("keeps the published FasterNet float and quantized Top-1 pair", async () => {
    const record = platformSlice(await repositoryCatalog(), "x5", "fasternet").benchmarks
      .find((benchmark) => benchmark.id === "fasternet-s-x5");

    expect(record?.accuracy).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "top-1", value: 77.04, unit: "percent", model_stage: "float" }),
      expect.objectContaining({ metric: "top-1", value: 76.15, unit: "percent", model_stage: "quantized" })
    ]));
  });

  it("scopes Ultralytics YOLO throughput to its published thread count", async () => {
    // The Ultralytics sample directory publishes several YOLO generations; the
    // catalog joins each to its own family, so the record is reached through
    // `yolov11` rather than through the shared sample directory.
    const records = platformSlice(await repositoryCatalog(), "x5", "yolov11").benchmarks;
    const record = records.find((benchmark) => benchmark.id === "ultralytics-yolo11n-detect-x5");

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
    const records = platformSlice(await repositoryCatalog(), "x5", "yolov8").benchmarks;

    expect(records.find((benchmark) => benchmark.id === "ultralytics-yolov8n-seg-x5"))
      .toEqual(expect.objectContaining({ variant_id: "yolov8n-seg-640" }));
    expect(records.find((benchmark) => benchmark.id === "ultralytics-yolov8n-pose-x5"))
      .toEqual(expect.objectContaining({ variant_id: "yolov8n-pose-640" }));
  });

  it("keeps the YOLO26 FP32 and BPU Python detection accuracy pair", async () => {
    const record = platformSlice(await repositoryCatalog(), "x5", "yolov26").benchmarks
      .find((benchmark) => benchmark.id === "ultralytics-yolo26n-detect-x5");

    expect(record?.accuracy).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "bbox-all-map-50-95", value: 0.319, unit: "ratio", model_stage: "float" }),
      expect.objectContaining({ metric: "bbox-all-map-50-95", value: 0.284, unit: "ratio", model_stage: "quantized" })
    ]));
  });

  it("keeps FCOS evaluator timing for the distinct two-thread BPU condition", async () => {
    const record = platformSlice(await repositoryCatalog(), "x5", "fcos").benchmarks
      .find((benchmark) => benchmark.id === "fcos-efficientnetb0-two-thread-x5");

    expect(record?.performance).toEqual(expect.arrayContaining([
      expect.objectContaining({ metric: "latency", value: 6.2, unit: "ms", concurrency: 2 }),
      expect.objectContaining({ metric: "throughput", value: 323, unit: "fps", concurrency: 2 })
    ]));
  });

  it("preserves UNet historical accuracy and host release results with their limitations", async () => {
    const records = platformSlice(await repositoryCatalog(), "x5", "unet").benchmarks;

    expect(records).toHaveLength(8);
    expect(records.flatMap((record) => record.accuracy ?? [])).toHaveLength(18);
    expect(records.filter((record) => record.id.includes("history"))
      .every((record) => record.accuracy?.every((metric) => metric.scope?.includes("not current download revalidation"))))
      .toBe(true);
    expect(records.filter((record) => record.id.includes("release-reference"))
      .every((record) => record.accuracy?.every((metric) => metric.scope?.includes("board runtime pending"))))
      .toBe(true);
  });

  it("keeps CLIP empty when its sources publish no numeric benchmark", async () => {
    expect(platformSlice(await repositoryCatalog(), "x5", "clip").benchmarks).toEqual([]);
  });
});
