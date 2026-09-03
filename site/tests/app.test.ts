import { beforeEach, describe, expect, it, vi } from "vitest";
import { mountCatalog } from "../src/app";
import { loadCatalog, localizeDocumentShell } from "../src/main";
import type { Catalog } from "../src/catalog/types";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

const catalog: Catalog = {
  schema_version: 1,
  release: { tag: "x5-v1.0.0", platform: "x5", version: "1.0.0" },
  summary: {
    sample_count: 2,
    asset_count: 2,
    downloadable_asset_count: 2
  },
  models: [
    createModelFixture(),
    createModelFixture({
      id: "himloco",
      name: "HiMLoco",
      tasks: ["legged-locomotion-control"],
      sample_path: "samples/robotics/himloco",
      assets: [{
        filename: "himloco.onnx",
        format: "onnx",
        url: "https://archive.example.test/himloco.onnx",
        sha256: "a".repeat(64)
      }],
      benchmarks: [benchmarkFixture({
        id: "himloco-runtime-x5",
        sample_id: "himloco",
        variant_id: "himloco-policy",
        display_name: "HiMLoco policy",
        model_format: "onnx",
        precision: "float32",
        performance: [{
          metric: "throughput",
          value: 2800,
          unit: "fps",
          qualifier: "lower-bound",
          scope: "single-thread runtime",
          concurrency: 1
        }],
        accuracy: undefined
      })]
    })
  ]
};

const catalogWithoutBenchmarks: Catalog = {
  ...catalog,
  summary: { sample_count: 1, asset_count: 1, downloadable_asset_count: 1 },
  models: [createModelFixture({ id: "clip", name: "CLIP", benchmarks: [] })]
};

function root(): HTMLElement {
  return document.querySelector<HTMLElement>("#app")!;
}

describe("catalog application", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    document.documentElement.lang = "";
    document.body.innerHTML = '<main id="app"></main>';
  });

  it("renders release statistics and filters model cards", () => {
    const app = mountCatalog(root(), catalog, { locale: "en" });

    expect(document.querySelector('[data-testid="release-tag"]')?.textContent).toContain("x5-v1.0.0");
    expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(2);

    const search = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    search.value = "HiMLoco";
    search.dispatchEvent(new Event("input", { bubbles: true }));

    expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(1);
    expect(document.querySelector("article")?.textContent).toContain("HiMLoco");
    app.destroy();
  });

  it("rejects a catalog whose declared asset total disagrees with its models", () => {
    const inconsistentCatalog: Catalog = {
      ...catalog,
      summary: { ...catalog.summary, asset_count: 3 }
    };

    expect(() => mountCatalog(root(), inconsistentCatalog, { locale: "en" }))
      .toThrow("Catalog summary declares 3 assets but contains 2.");
  });

  it("shows explicit performance and accuracy empty states", () => {
    mountCatalog(root(), catalogWithoutBenchmarks, { locale: "en" });

    expect(document.body.textContent).toContain("No published performance data");
    expect(document.body.textContent).toContain("No published accuracy data");
  });

  it("resets search and filters to the full catalog", () => {
    mountCatalog(root(), catalog, { locale: "en" });
    const search = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    search.value = "HiMLoco";
    search.dispatchEvent(new Event("input", { bubbles: true }));
    expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(1);

    document.querySelector<HTMLButtonElement>('[data-action="reset-filters"]')!.click();

    expect(search.value).toBe("");
    expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(2);
  });

  it("explains when published conditions prevent the requested numeric sort", () => {
    mountCatalog(root(), catalog, { locale: "en" });
    const sort = document.querySelector<HTMLSelectElement>("#catalog-sort")!;
    sort.value = "fps";
    sort.dispatchEvent(new Event("change", { bubbles: true }));

    expect(document.querySelector(".sort-notice")?.textContent)
      .toContain("published benchmark conditions are not comparable");
  });

  it("uses bilingual task labels from the manifest mapping", () => {
    mountCatalog(root(), catalog, { locale: "zh" });

    expect(document.body.textContent).toContain("图像分类");
    expect(document.body.textContent).toContain("足式运动控制");
    expect(document.documentElement.lang).toBe("zh");
  });

  it("uses the localized fallback for an unknown future task", () => {
    const futureCatalog: Catalog = {
      ...catalogWithoutBenchmarks,
      models: [createModelFixture({
        id: "future-model",
        name: "Future Model",
        tasks: ["future-task"],
        benchmarks: []
      })]
    };

    mountCatalog(root(), futureCatalog, { locale: "en" });

    expect(document.querySelector(".task-badges")?.textContent).toBe("Task");
    expect(document.body.textContent).not.toContain("future-task");
  });

  it("localizes semantic page chrome outside the mounted catalog", () => {
    document.body.innerHTML = `
      <a data-i18n="app.skipToContent"></a>
      <header><p data-i18n="app.title"></p><p data-i18n="app.subtitle"></p></header>
      <main id="app"></main>
      <footer data-i18n="app.footer"></footer>
    `;

    localizeDocumentShell("zh");

    expect(document.querySelector("header")?.textContent).toContain("RDK 模型库");
    expect(document.querySelector("footer")?.textContent).toBe("RDK 模型库目录");
    expect(document.documentElement.lang).toBe("zh");
  });

  it("renders metric qualifiers and concise test conditions", () => {
    mountCatalog(root(), catalog, { locale: "en" });
    const card = document.querySelector<HTMLElement>('[data-model-id="himloco"]')!;

    expect(card.textContent).toContain("2,800 FPS");
    expect(card.textContent).toContain("lower bound");
    expect(card.textContent).toContain("RDK X5");
    expect(card.textContent).toContain("single-thread runtime");
  });

  it("keeps representative latency and throughput visible when several timings exist", () => {
    const timingCatalog: Catalog = {
      ...catalogWithoutBenchmarks,
      models: [createModelFixture({
        benchmarks: [benchmarkFixture({
          performance: [
            { metric: "latency", value: 2, unit: "ms", scope: "single frame", concurrency: 1 },
            { metric: "latency", value: 3, unit: "ms", scope: "end to end", concurrency: 1 },
            { metric: "throughput", value: 500, unit: "fps", scope: "four threads", concurrency: 4 }
          ]
        })]
      })]
    };

    mountCatalog(root(), timingCatalog, { locale: "en" });

    const card = document.querySelector<HTMLElement>(".model-card")!;
    expect(card.textContent).toContain("Latency: 2 ms");
    expect(card.textContent).toContain("Throughput: 500 FPS");
  });

  it("removes registered event listeners when destroyed", () => {
    const app = mountCatalog(root(), catalog, { locale: "en" });
    const search = document.querySelector<HTMLInputElement>('input[type="search"]')!;
    app.destroy();

    search.value = "HiMLoco";
    search.dispatchEvent(new Event("input", { bubbles: true }));

    expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(2);
  });

  it("opens and closes shareable details while preserving unrelated URL parameters and focus", () => {
    window.history.replaceState({}, "", "/?view=cards");
    mountCatalog(root(), catalog, { locale: "en" });
    const opener = document.querySelector<HTMLButtonElement>(
      '[data-model-id="himloco"] [data-action="open-details"]'
    )!;
    opener.focus();
    opener.click();

    expect(new URL(window.location.href).searchParams.get("model")).toBe("himloco");
    expect(document.querySelector('.model-details[data-model-id="himloco"]')).not.toBeNull();

    document.querySelector<HTMLButtonElement>('[data-action="close-details"]')!.click();

    expect(new URL(window.location.href).searchParams.get("model")).toBeNull();
    expect(new URL(window.location.href).searchParams.get("view")).toBe("cards");
    expect(document.querySelector(".model-details")).toBeNull();
    expect(document.activeElement).toBe(opener);
  });

  it("restores details on popstate and recovers from an unknown model id", () => {
    mountCatalog(root(), catalog, { locale: "en" });
    window.history.pushState({}, "", "/?model=convnext");
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(document.querySelector('.model-details[data-model-id="convnext"]')).not.toBeNull();

    window.history.pushState({}, "", "/?model=missing-model");
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(document.body.textContent).toContain("Model not found");
    document.querySelector<HTMLButtonElement>('[data-action="clear-model"]')!.click();
    expect(new URL(window.location.href).searchParams.get("model")).toBeNull();
    expect(document.querySelector(".model-details")).toBeNull();
  });

  it("opens an initial deep link from the model query parameter", () => {
    window.history.replaceState({}, "", "/?model=himloco");
    mountCatalog(root(), catalog, { locale: "en" });

    expect(document.querySelector('.model-details[data-model-id="himloco"]')).not.toBeNull();
  });

  it("renders a recoverable localized error when catalog loading fails", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("unavailable", { status: 503 }));

    const app = await loadCatalog(root(), { locale: "en", fetcher });

    expect(app).toBeNull();
    expect(root().getAttribute("role")).toBe("alert");
    expect(root().textContent).toContain("The catalog could not be loaded.");
    expect(root().querySelector('a[href*="release/models.yaml"]')).not.toBeNull();
    expect(root().querySelector('a[href="https://github.com/D-Robotics/rdk_model_zoo"]')).not.toBeNull();
    expect(root().querySelector('[data-action="retry-load"]')).not.toBeNull();
  });
});
