import { describe, expect, it } from "vitest";
import { readModelId, renderModelDetails, writeModelId } from "../src/ui/model-details";
import { createModelFixture } from "./fixtures/catalog";

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

  it("renders metric conditions and immutable source links", () => {
    const element = renderModelDetails(createModelFixture(), detailContext);

    expect(element.textContent).toContain("single-frame, single-thread, single-BPU-core");
    expect(element.textContent).toContain("ImageNet-1K");
    expect(element.textContent).toContain("quantized");
    const source = element.querySelector<HTMLAnchorElement>('a[data-testid="benchmark-source"]')!;
    expect(source.href).toContain("/blob/x5-v1.0.0/samples/vision/convnext/README.md");
  });

  it("renders captioned variants, performance, accuracy, and asset tables", () => {
    const element = renderModelDetails(createModelFixture(), detailContext);
    const captions = [...element.querySelectorAll("caption")].map((caption) => caption.textContent);

    expect(captions).toEqual(expect.arrayContaining([
      "Model variants",
      "Performance",
      "Accuracy",
      "Assets"
    ]));
    expect(element.querySelectorAll('a[data-testid="benchmark-source"]')).toHaveLength(3);
    const headers = [...element.querySelectorAll("th")].map((header) => header.textContent);
    expect(headers).toEqual(expect.arrayContaining(["Value", "Unit", "Qualifier"]));
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

  it("describes an empty variant table without calling it missing benchmark data", () => {
    const model = createModelFixture({ benchmarks: [] });

    const element = renderModelDetails(model, detailContext);
    const variantsTable = element.querySelector("table")!;

    expect(variantsTable.textContent).toContain("No published model variant data");
    expect(variantsTable.textContent).not.toContain("benchmark data");
  });
});
