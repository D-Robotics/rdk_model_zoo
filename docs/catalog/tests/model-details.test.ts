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

  it("labels one shared accuracy metric in its own column group and derives its retention", () => {
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "bbox-all-map-50-95", value: 0.306, unit: "ratio", dataset: "COCO2017", model_stage: "float" },
        { metric: "bbox-all-map-50-95", value: 0.292, unit: "ratio", dataset: "COCO2017", model_stage: "quantized" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;

    // The metric owns a column group; its header carries dataset and scale.
    expect(table.textContent).toContain("bbox mAP@0.5:0.95");
    expect(table.textContent).toContain("COCO2017 · 0–1");
    // Values are shown exactly as published: no percent sign, and retention —
    // which is defined as a percentage — keeps it.
    expect(table.textContent).toContain("0.306");
    expect(table.textContent).toContain("0.292");
    expect(table.textContent).toContain("95.42%");
    expect(table.textContent).not.toContain("30.6 %");
    expect(table.querySelector("tbody td.model-detail-accuracy-cell")?.textContent).not.toContain("bbox-all-map-50-95");
  });

  it("keeps a published accuracy value visible when the source labels no model stage", () => {
    // SigLIP publishes BPU top-1/top-5 with no float/quantized split. An
    // unlabelled stage is not the same statement as “not yet measured”.
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "top-1", value: 0.7482, unit: "ratio", dataset: "ImageNet-1k" },
        { metric: "top-5", value: 0.934, unit: "ratio", dataset: "ImageNet-1k" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;
    const cells = [...table.querySelectorAll("tbody td.model-detail-accuracy-cell")].map((cell) => cell.textContent).join("|");

    expect(table.textContent).toContain("Top-1");
    expect(table.textContent).toContain("Top-5");
    expect(cells).toContain("0.7482");
    expect(cells).toContain("0.934");
    expect(cells).not.toContain("Accuracy not yet measured");
  });

  it("merges one metric published under different source spellings into one column", () => {
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "TOP1", value: 0.783, unit: "ratio", dataset: "ImageNet-1k", model_stage: "float" },
        { metric: "top-1", value: 0.718, unit: "ratio", dataset: "ImageNet-1k", model_stage: "quantized" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;
    const groups = [...table.querySelectorAll("thead th.model-detail-accuracy-group")];

    expect(groups).toHaveLength(1);
    expect(groups[0]!.textContent).toContain("Top-1");
    expect(table.textContent).toContain("0.783");
    expect(table.textContent).toContain("0.718");
    expect(table.textContent).toContain("91.7%");
  });

  it("gives each accuracy metric its own column instead of piling them into one cell", () => {
    const benchmark = benchmarkFixture({
      accuracy: [
        { metric: "bbox-all-map-50-95", value: 0.391, unit: "ratio", dataset: "COCO2017", model_stage: "float" },
        { metric: "bbox-small-map-50-95", value: 0.195, unit: "ratio", dataset: "COCO2017", model_stage: "float" },
        { metric: "bbox-medium-map-50-95", value: 0.437, unit: "ratio", dataset: "COCO2017", model_stage: "float" },
        { metric: "bbox-large-map-50-95", value: 0.566, unit: "ratio", dataset: "COCO2017", model_stage: "float" }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;
    const groupLabels = [...table.querySelectorAll("thead th.model-detail-accuracy-group")].map((group) => group.textContent);
    const firstAccuracyCell = table.querySelector("tbody td.model-detail-accuracy-cell")!;

    expect(groupLabels).toHaveLength(4);
    expect(groupLabels.join("|")).toContain("(small)");
    expect(groupLabels.join("|")).toContain("(medium)");
    expect(groupLabels.join("|")).toContain("(large)");
    // One measurement per cell: four metrics no longer share a single cell.
    expect(firstAccuracyCell.textContent!.trim()).toBe("0.391");
  });

  it("drops the hardware qualifier from a row name because the tab already states it", () => {
    const benchmark = benchmarkFixture({ environment: { hardware: "RDK S100" } });
    const element = renderModelDetails(createModelFixture({
      variants: [variant({ hardware: "s100", name: "YOLOv8n Detect on RDK S100", benchmarks: [benchmark] })]
    }), { ...detailContext, hardware: "s100" });
    const specification = element.querySelector("th.model-detail-specification")!;

    expect(specification.textContent).toContain("YOLOv8n Detect");
    expect(specification.textContent).not.toContain("RDK S100");
  });

  it("separates a missing FPS from a model that published no performance table", () => {
    const benchmark = benchmarkFixture({
      performance: [{ metric: "latency", value: 26.8, unit: "ms", scope: "pooler output" }]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;
    const throughput = table.querySelector("tbody td.model-detail-throughput")!;

    expect(table.textContent).toContain("26.8 ms");
    expect(throughput.textContent).toContain("Not recorded");
    expect(throughput.textContent).not.toContain("Performance not yet measured");
  });

  it("keeps an accuracy measurement out of a row whose timing scope does not cover it", () => {
    // SigLIP reports top-1/top-5 for the pooler output and cosine similarity for
    // the last hidden state; each row shows only the scope it measured.
    const benchmark = benchmarkFixture({
      environment: { hardware: "RDK S100" },
      performance: [
        { metric: "latency", value: 26.8, unit: "ms", scope: "pooler output" },
        { metric: "latency", value: 26, unit: "ms", scope: "last hidden state" }
      ],
      accuracy: [
        { metric: "top-1", value: 0.7118, unit: "ratio", dataset: "ImageNet-1k", scope: "pooler output zero-shot classification on BPU" },
        { metric: "cosine-similarity", value: 0.991, unit: "ratio", dataset: "COCO2014", scope: "last hidden state mean" }
      ]
    });
    const element = renderModelDetails(createModelFixture({
      variants: [variant({ hardware: "s100", task: "vision-embedding", benchmarks: [benchmark] })]
    }), { ...detailContext, hardware: "s100", task: "vision-embedding" });
    const rows = [...element.querySelectorAll<HTMLTableRowElement>("tbody tr.model-detail-spec-row")];

    expect(rows).toHaveLength(2);
    expect(rows[0]!.textContent).toContain("0.7118");
    expect(rows[0]!.textContent).not.toContain("0.991");
    expect(rows[1]!.textContent).toContain("0.991");
    expect(rows[1]!.textContent).not.toContain("0.7118");
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

  it("renders expanded evidence as a full-width sibling row", () => {
    const element = renderModelDetails(createModelFixture({ variants: [variant()] }), detailContext);
    const table = element.querySelector<HTMLTableElement>(".model-detail-specifications-table")!;
    const mainRow = table.querySelector<HTMLTableRowElement>("tbody tr.model-detail-spec-row")!;
    const expandedRow = mainRow.nextElementSibling as HTMLTableRowElement;
    const headerColumns = [...table.tHead!.rows[0]!.cells]
      .reduce((total, header) => total + header.colSpan, 0);

    expect(expandedRow.classList.contains("model-detail-expanded-row")).toBe(true);
    expect(expandedRow.hidden).toBe(true);
    expect(expandedRow.querySelector("td")?.colSpan).toBe(headerColumns);
    expect(expandedRow.querySelector(".model-detail-row-details")).toBeTruthy();
    mainRow.querySelector<HTMLButtonElement>('[data-action="toggle-row-details"]')!.click();
    expect(expandedRow.hidden).toBe(false);
  });

  it("keeps extra thread columns behind a URL-addressable table control", () => {
    window.history.replaceState({}, "", "/?model=convnext");
    const benchmark = benchmarkFixture({
      performance: [
        { metric: "latency", value: 2, unit: "ms", concurrency: 1 },
        { metric: "throughput", value: 500, unit: "fps", concurrency: 1 },
        { metric: "latency", value: 3, unit: "ms", concurrency: 4 },
        { metric: "throughput", value: 900, unit: "fps", concurrency: 4 }
      ]
    });
    const element = renderModelDetails(createModelFixture({ variants: [variant({ benchmarks: [benchmark] })] }), detailContext);
    const tableHost = element.querySelector<HTMLElement>(".model-detail-benchmark-table")!;
    const toggle = element.querySelector<HTMLButtonElement>('[data-action="toggle-threads"]')!;

    expect(tableHost.querySelector('thead th[data-concurrency="4"]')).toBeNull();
    toggle.click();
    expect(tableHost.querySelector('thead th[data-concurrency="4"]')).not.toBeNull();
    expect(new URL(window.location.href).searchParams.get("threads")).toBe("all");
    toggle.click();
    expect(new URL(window.location.href).searchParams.has("threads")).toBe(false);
  });
});
