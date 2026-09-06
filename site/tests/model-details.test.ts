import { describe, expect, it } from "vitest";
import { readModelId, renderModelDetails, writeModelId } from "../src/ui/model-details";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

const detailContext = {
  locale: "en" as const,
  repositoryUrl: "https://github.com/D-Robotics/rdk_model_zoo",
  releaseTag: "x5-v1.0.0"
};

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

  it("renders immutable source links in the platform benchmark table", () => {
    const element = renderModelDetails(createModelFixture(), detailContext);

    expect(element.textContent).toContain("Platform benchmarks");
    expect(element.textContent).toContain("1 thread latency");
    expect(element.textContent).toContain("Quantized/BPU accuracy");
    const source = element.querySelector<HTMLAnchorElement>('a[data-testid="benchmark-source"]')!;
    expect(source.href).toContain("/blob/x5-v1.0.0/samples/vision/convnext/README.md");
  });

  it("combines each hardware and model size into one thread-oriented benchmark row", () => {
    const model = createModelFixture();
    const benchmark = model.benchmarks[0]!;
    benchmark.performance = [
      { metric: "latency", value: 2, unit: "ms", concurrency: 1 },
      { metric: "throughput", value: 500, unit: "fps", concurrency: 1 },
      { metric: "latency", value: 3, unit: "ms", concurrency: 2 },
      { metric: "throughput", value: 900, unit: "fps", concurrency: 2 }
    ];
    benchmark.accuracy = [
      { metric: "top-1", value: 75, unit: "percent", model_stage: "float" },
      { metric: "top-1", value: 73, unit: "percent", model_stage: "quantized" }
    ];
    const element = renderModelDetails(model, detailContext);
    const matrix = [...element.querySelectorAll("table")].find((table) => table.textContent?.includes("1 thread latency"));

    expect(matrix?.textContent).toContain("2 ms");
    expect(matrix?.textContent).toContain("500 FPS");
    expect(matrix?.textContent).toContain("3 ms");
    expect(matrix?.textContent).toContain("900 FPS");
    expect(matrix?.textContent).toContain("75 %");
    expect(matrix?.textContent).toContain("73 %");
  });

  it("limits the platform table to the selected hardware", () => {
    const model = createModelFixture({
      platforms: [
        { ...createModelFixture(), platform: "x5", release_tag: "x5-v1.0.0" },
        { ...createModelFixture(), platform: "s", release_tag: "s-v1.0.0", benchmarks: [benchmarkFixture({ environment: { hardware: "RDK S100" } })] }
      ]
    });
    const element = renderModelDetails(model, { ...detailContext, platform: "RDK S100" });

    expect(element.textContent).toContain("RDK S100");
    expect(element.textContent).not.toContain("RDK X5");
  });

  it("renders a captioned platform benchmark table and asset table", () => {
    const element = renderModelDetails(createModelFixture(), detailContext);
    const captions = [...element.querySelectorAll("caption")].map((caption) => caption.textContent);

    expect(captions).toEqual(expect.arrayContaining([
      "Platform benchmarks",
      "Assets"
    ]));
    expect(element.querySelectorAll('a[data-testid="benchmark-source"]')).toHaveLength(1);
    const headers = [...element.querySelectorAll("th")].map((header) => header.textContent);
    expect(headers).toEqual(expect.arrayContaining(["1 thread FPS", "2 thread FPS", "FP32 accuracy"]));
  });

  it("renders assets without a URL as manual downloads", () => {
    const manualAssetFixture = createModelFixture({
      availability: "manual",
      assets: [{ filename: "modnet_photographic_portrait_matting_512x512_nv12.bin", format: "bin", sha256: null }]
    });

    const element = renderModelDetails(manualAssetFixture, detailContext);

    expect(element.textContent).toContain("Manual model required");
    expect(element.querySelector("a[download]")).toBeNull();
  });

  it("describes an empty platform benchmark table as not yet measured", () => {
    const model = createModelFixture({ benchmarks: [] });

    const element = renderModelDetails(model, detailContext);
    const variantsTable = element.querySelector("table")!;

    expect(variantsTable.textContent).toContain("Performance not yet measured");
  });

  it("keeps float and quantized accuracy in separate platform columns", () => {
    const model = createModelFixture();
    const benchmark = model.benchmarks[0]!;
    benchmark.accuracy = [{
      metric: "top-1",
      value: 73,
      unit: "percent",
      model_stage: "quantized"
    }];

    const element = renderModelDetails(model, detailContext);
    const headers = [...element.querySelectorAll("th")].map((header) => header.textContent);
    expect(headers).toEqual(expect.arrayContaining(["FP32 accuracy", "Quantized/BPU accuracy"]));
    expect(element.textContent).toContain("73 %");
  });
});
