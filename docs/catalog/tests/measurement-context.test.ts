import { describe, expect, it } from "vitest";
import { formatMetricValue, groupPerformanceMetrics } from "../src/catalog/metric-display";
import { renderModelDetails } from "../src/ui/model-details";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

const context = { locale: "en" as const, repositoryUrl: "https://example.test", releaseTag: "x5-v1.1.2" };
describe("measurement contexts", () => {
  it("does not label a mixed millisecond/microsecond column as milliseconds", () => {
    const record = benchmarkFixture({ performance: [
      { metric: "latency", value: 0.885, unit: "ms", scope: "Python runtime" },
      { metric: "latency", value: 63, unit: "us", scope: "Compiler estimate" }
    ], accuracy: [] });
    const page = renderModelDetails(createModelFixture({ benchmarks: [record] }), context);
    expect(page.querySelector('thead th[data-metric="latency"]')?.textContent).toBe("Latency");
    expect(page.textContent).toContain("63 us");
    expect(page.textContent).toContain("0.885 ms");
  });
  it("preserves raw ratio precision when percentage conversion is disabled", () => {
    expect(formatMetricValue({ metric: "cosine", value: 0.999606, unit: "ratio" }, "en", { asPercentage: false })).toBe("0.999606ratio");
  });
  it("shows separately named stage latencies instead of claiming performance is unmeasured", () => {
    const record = benchmarkFixture({ performance: [
      { metric: "encoder-latency", value: 33.63, unit: "ms", scope: "hbm_runtime" },
      { metric: "decoder-latency", value: 7.12, unit: "ms", scope: "hbm_runtime" },
      { metric: "post_process_latency", value: 9, unit: "ms", scope: "Python post-process" }
    ], accuracy: [] });
    const groups = groupPerformanceMetrics([record]);
    expect(groups.flatMap(g => g.threads.map(t => t.latency?.metric.value))).toEqual([33.63, 7.12, 9]);
    const page = renderModelDetails(createModelFixture({ benchmarks: [record] }), context);
    expect(page.querySelectorAll('.model-detail-spec-row')).toHaveLength(1);
    expect(page.querySelector('thead')?.textContent).toContain("Encoder Latency");
    expect(page.querySelector('thead')?.textContent).toContain("Decoder Latency");
    expect(page.querySelector('.model-detail-stage-value[data-metric="encoder-latency"]')?.textContent).toContain("33.63");
    expect(page.querySelector('.model-detail-stage-value[data-metric="decoder-latency"]')?.textContent).toContain("7.12");
    expect(page.querySelector('.model-detail-postprocess')?.textContent).toContain("9 ms");
  });
  it("keeps conflicting measurements with the same scope visible", () => {
    const records = [1129.37, 2853.09].map((value, index) => benchmarkFixture({ id: `runtime-${index}`, performance: [{ metric: "throughput", value, unit: "fps", scope: "sequential; 100 inputs; 10 warm-up runs" }], accuracy: [] }));
    const visible = groupPerformanceMetrics(records).flatMap(g => g.threads.map(t => t.throughput?.metric.value));
    expect(visible).toEqual([1129.37, 2853.09]);
  });
  it("does not merge measurements from different runtime environments", () => {
    const records = ["Python", "C++"].map((runtime, index) => benchmarkFixture({ id: `runtime-${index}`, environment: { runtime }, performance: [{ metric: "latency", value: 1, unit: "ms", concurrency: 1 }], accuracy: [] }));
    expect(groupPerformanceMetrics(records)).toHaveLength(2);
  });
  it("keeps runtime observations and compilation accuracy in one configuration with their scopes", () => {
    const python = benchmarkFixture({ id: "python", display_name: "HIMLoco Go2 Python Runtime", performance: [{ metric: "latency", value: 0.885, unit: "ms", scope: "Python synchronous call" }], accuracy: [{ metric: "cosine_similarity", value: 0.9999, unit: "ratio", scope: "MIX PTQ compilation output" }] });
    const cpp = benchmarkFixture({ id: "cpp", display_name: "HIMLoco Go2 C++ Runtime", performance: [{ metric: "latency", value: 0.35, unit: "ms", scope: "C++ inference and wait" }], accuracy: [] });
    const page = renderModelDetails(createModelFixture({ benchmarks: [python, cpp], variants: [{ id: "himloco-x5", name: "HIMLoco Go2", hardware: "x5", task: "image-classification", benchmarks: [python, cpp], assets: [], sample_path: "samples/robotics/himloco", release_tag: context.releaseTag }] }), context);
    expect(page.querySelectorAll('.model-detail-spec-row')).toHaveLength(1);
    const row = page.querySelector('.model-detail-spec-row')!;
    expect(row.querySelector('th')?.textContent).toContain("HIMLoco Go2");
    const observations = [...row.querySelectorAll('.model-detail-observation')].map(item => item.textContent);
    expect(observations.some(text => text?.includes("0.885 ms") && text.includes("Python synchronous call"))).toBe(true);
    expect(observations.some(text => text?.includes("0.35 ms") && text.includes("C++ inference and wait"))).toBe(true);
    const accuracy = page.querySelector('.model-detail-spec-row .model-detail-accuracy-value');
    expect(accuracy?.textContent).toContain("0.9999");
    expect(accuracy?.closest('tr')?.textContent).toContain("MIX PTQ compilation output");
  });
});
