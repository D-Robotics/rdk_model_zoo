import { describe, expect, it, vi } from "vitest";
import { readModelId, renderModelDetails, writeModelId } from "../src/ui/model-details";
import type { ModelVariant } from "../src/catalog/types";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

const detailContext = {
  locale: "en" as const,
  repositoryUrl: "https://github.com/D-Robotics/rdk_model_zoo",
  releaseTag: "x5-v1.0.0"
};

function variant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  const benchmark = benchmarkFixture();
  return {
    id: "convnext-atto-224",
    name: "ConvNeXt Atto",
    hardware: "x5",
    task: "image-classification",
    input: benchmark.input,
    assets: [{ filename: "ConvNeXt_atto_224x224_nv12.bin", format: "bin", url: "https://archive.example.test/convnext.bin" }],
    benchmarks: [benchmark],
    sample_path: "samples/vision/convnext",
    release_tag: "x5-v1.0.0",
    ...overrides
  };
}

describe("model details", () => {
  it("reads and writes a shareable model query parameter without losing other URL state", () => {
    expect(readModelId(new URL("https://example.test/?model=convnext"))).toBe("convnext");
    expect(readModelId(new URL("https://example.test/?model="))).toBeNull();

    const opened = writeModelId(new URL("https://example.test/catalog?view=cards#models"), "himloco");
    expect(opened.search).toBe("?view=cards&model=himloco");
    expect(opened.hash).toBe("#models");

    const closed = writeModelId(opened, null);
    expect(closed.search).toBe("?view=cards");
    expect(closed.hash).toBe("#models");
  });

  it("renders a section with only supported hardware tabs and exposes canonical selection state", () => {
    const model = createModelFixture({ variants: [
      variant({ hardware: "x5" }),
      variant({ hardware: "s100", id: "convnext-s100", release_tag: "s-v1.0.0" })
    ] });

    const element = renderModelDetails(model, { ...detailContext, hardware: "s100" });

    expect(element.tagName).toBe("SECTION");
    expect(element.querySelector("h1")?.textContent).toContain("ConvNeXt");
    expect(element.dataset.hardware).toBe("s100");
    expect(element.dataset.task).toBe("image-classification");
    expect([...element.querySelectorAll<HTMLButtonElement>('[role="tab"]')].map((tab) => tab.dataset.hardware))
      .toEqual(["x5", "s100"]);
    expect(element.querySelector('[data-action="close-details"]')).toBeTruthy();
  });

  it("rerenders files and benchmark contents locally when a hardware tab is switched", () => {
    const x5Benchmark = benchmarkFixture({ environment: { hardware: "RDK X5" }, asset_filename: "x5.bin" });
    const s100Benchmark = benchmarkFixture({ environment: { hardware: "RDK S100" }, asset_filename: "s100.bin" });
    const model = createModelFixture({ variants: [
      variant({ hardware: "x5", assets: [{ filename: "x5.bin", format: "bin", url: "https://archive.example.test/x5.bin" }], benchmarks: [x5Benchmark] }),
      variant({ hardware: "s100", id: "convnext-s100", release_tag: "s-v1.0.0", assets: [{ filename: "s100.bin", format: "bin", url: "https://archive.example.test/s100.bin" }], benchmarks: [s100Benchmark] })
    ] });
    const onSelectionChange = vi.fn();
    const element = renderModelDetails(model, { ...detailContext, onSelectionChange });
    const s100Tab = element.querySelector<HTMLButtonElement>('[role="tab"][data-hardware="s100"]')!;

    expect(element.textContent).toContain("x5.bin");
    expect(element.textContent).not.toContain("s100.bin");
    s100Tab.click();
    expect(element.dataset.hardware).toBe("s100");
    expect(element.textContent).toContain("s100.bin");
    expect(element.textContent).not.toContain("x5.bin");
    expect(onSelectionChange).toHaveBeenCalledWith("s100", "image-classification");
  });

  it("keeps latency and FPS for each thread on the same row and includes unknown threads", () => {
    const benchmark = benchmarkFixture({
      performance: [
        { metric: "latency", value: 2, unit: "ms", scope: "BPU", concurrency: 1 },
        { metric: "throughput", value: 500, unit: "fps", scope: "BPU", concurrency: 1 },
        { metric: "latency", value: 3, unit: "ms", scope: "BPU", concurrency: 2 },
        { metric: "throughput", value: 900, unit: "fps", scope: "BPU", concurrency: 2 },
        { metric: "throughput", value: 34.1, unit: "fps", scope: "BPU throughput" }
      ],
      accuracy: undefined
    });
    const model = createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] });
    const element = renderModelDetails(model, detailContext);
    const row = element.querySelector<HTMLTableRowElement>("tbody tr")!;

    expect(row.textContent).toContain("2 ms");
    expect(row.textContent).toContain("500 FPS");
    expect(row.textContent).toContain("3 ms");
    expect(row.textContent).toContain("900 FPS");
    expect(element.textContent).toContain("Concurrency not recorded");
    expect(element.textContent).toContain("34.1 FPS");
  });

  it("shows explicit retention before derived retention and does not pair a wrong dataset", () => {
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "bbox-all-map-50-95", value: 0.306, unit: "ratio", dataset: "COCO", model_stage: "float" },
        { metric: "bbox-all-map-50-95", value: 0.292, unit: "ratio", dataset: "COCO", model_stage: "quantized" },
        { metric: "bbox-all-map-50-95-retention", value: 94, unit: "percent", dataset: "COCO" },
        { metric: "bbox-all-map-50-95", value: 0.8, unit: "ratio", dataset: "ImageNet", model_stage: "quantized" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);

    expect(element.textContent).toContain("94%");
    expect(element.textContent).not.toContain("95.42%");
    expect(element.textContent).toContain("ImageNet");
    expect(element.textContent).toContain("Not comparable");
  });

  it("labels one shared accuracy metric in the group header and derives its retention", () => {
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "bbox-all-map-50-95", value: 0.306, unit: "ratio", dataset: "COCO2017", model_stage: "float" },
        { metric: "bbox-all-map-50-95", value: 0.292, unit: "ratio", dataset: "COCO2017", model_stage: "quantized" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;

    expect(table.textContent).toContain("COCO2017 · bbox mAP@0.5:0.95 (%)");
    expect(table.textContent).toContain("95.42%");
    expect(table.querySelector("tbody td.model-detail-accuracy-cell")?.textContent).not.toContain("bbox-all-map-50-95");
  });

  it("does not treat X3 post-processing as BPU latency and keeps all metrics in row details", () => {
    const benchmark = benchmarkFixture({
      environment: { hardware: "RDK X3" },
      performance: [
        { metric: "throughput", value: 34.1, unit: "fps", scope: "BPU throughput" },
        { metric: "post_process_latency", value: 6, unit: "ms", scope: "Python post-process" }
      ]
    });
    const model = createModelFixture({ variants: [variant({ hardware: "x3", benchmarks: [benchmark] })] });
    const element = renderModelDetails(model, { ...detailContext, hardware: "x3" });
    const latencyCell = element.querySelector<HTMLElement>('tbody td[data-metric="latency"][data-concurrency="unknown"]');

    expect(latencyCell).toBeNull();
    expect(element.querySelector("details")?.textContent).toContain("post_process_latency");
    expect(element.querySelector("details")?.textContent).toContain("6 ms");
  });

  it("keeps a recorded non-BPU latency scope visible beside the specification", () => {
    const benchmark = benchmarkFixture({
      environment: { hardware: "RDK X3" },
      performance: [{ metric: "latency", value: 6, unit: "ms", scope: "Python post-process" }]
    });
    const model = createModelFixture({ variants: [variant({ hardware: "x3", benchmarks: [benchmark] })] });
    const element = renderModelDetails(model, { ...detailContext, hardware: "x3" });

    expect(element.querySelector(".model-detail-measurement-scope")?.textContent)
      .toContain("Python post-process");
  });

  it("renders manual and runnable downloads from the selected variant only", () => {
    const model = createModelFixture({ variants: [variant({
      assets: [
        { filename: "model.bin", format: "bin", url: "https://archive.example.test/model.bin" },
        { filename: "model.onnx", format: "onnx", url: "https://archive.example.test/model.onnx" },
        { filename: "manual.bin", format: "bin" }
      ]
    })] });
    const element = renderModelDetails(model, detailContext);

    expect(element.querySelector('a[download][href$="model.bin"]')).toBeTruthy();
    expect(element.querySelector('a[href$="model.onnx"]')).toBeNull();
    expect(element.textContent).toContain("Download address not recorded");
  });

  it("keeps download-only supported rows visible with explicit missing labels and compact links", () => {
    const element = renderModelDetails(createModelFixture({ variants: [variant({
      benchmarks: [],
      assets: [{ filename: "download_only.hbm", format: "hbm", url: "https://archive.example.test/download_only.hbm" }]
    })] }), detailContext);

    expect(element.textContent).toContain("Performance not yet measured");
    expect(element.querySelector('[data-action="download-model"]')?.textContent).toBe("Download .hbm");
    expect(element.querySelector(".model-detail-row-details")?.textContent).toContain("download_only.hbm");
    expect(element.textContent).not.toContain("—");
  });
});
