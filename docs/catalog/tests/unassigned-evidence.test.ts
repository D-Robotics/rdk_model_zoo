import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { renderModelDetails } from "../src/ui/model-details";
import { renderEvidenceDetails } from "../src/ui/evidence-details";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

const context = { locale: "en" as const, repositoryUrl: "https://github.com/D-Robotics/rdk_model_zoo", releaseTag: "s-v1.1.2" };

describe("unassigned benchmark evidence", () => {
  it("puts units only in the dedicated evidence unit column", () => {
    const element = renderEvidenceDetails([benchmarkFixture({ performance: [{ metric: "latency", value: 3.75, unit: "ms" }], accuracy: [] })], context);
    const row = element.querySelector('tbody tr')!;
    expect(row.children[1]?.textContent).toBe("3.75");
    expect(row.children[2]?.textContent).toBe("ms");
  });
  it("retains S toolchain and unspecified-board source records without inventing hardware variants", async () => {
    const catalog = await buildMultiplatformCatalog(resolve(process.cwd(), "../.."));
    for (const [sample, recordId] of [["dinov2", "dinov2-ptq-nash-e"], ["pointnet", "pointnet-s100"]]) {
      const model = catalog.models.find(m => m.platforms?.some(p => p.benchmarks.some(r => r.sample_id === sample)))!;
      expect(model.benchmarks.map(r => r.id)).toContain(recordId);
      expect(model.variants?.flatMap(v => v.benchmarks).map(r => r.id)).not.toContain(recordId);
      const page = renderModelDetails(model, context);
      expect(page.querySelector(`[data-unassigned-record="${recordId}"]`)).not.toBeNull();
    }
  });

  it("includes the ASR cosine values published in the pinned evaluator image", async () => {
    const catalog = await buildMultiplatformCatalog(resolve(process.cwd(), "../.."));
    const model = catalog.models.find(m => m.id === "asr")!;
    const evidence = model.benchmarks.find(r => r.id === "asr-toolchain-cosine-image");
    expect(evidence?.accuracy?.map(metric => metric.value)).toEqual([0.999105, 0.999181]);
    expect(evidence?.source.path).toBe("samples/speech/asr/test_data/readme_img/acc.jpg");
    expect(model.variants?.flatMap(v => v.benchmarks).some(r => r.id === evidence?.id)).toBe(false);
  });

  it("shows unassigned values and original hardware independently of board performance rows", () => {
    const record = benchmarkFixture({ id: "compiler-result", environment: { hardware: "OpenExplore nash-e toolchain" }, performance: [], accuracy: [{ metric: "cosine_similarity", value: 0.9999, unit: "ratio", model_stage: "compiled" }] });
    const model = createModelFixture({ benchmarks: [record], assets: [], variants: [] });
    const page = renderModelDetails(model, context);
    const evidence = page.querySelector('[data-unassigned-record="compiler-result"]');
    expect(evidence?.textContent).toContain("OpenExplore nash-e toolchain");
    expect(evidence?.textContent).toContain("0.9999");
    expect(page.querySelectorAll('[role="tab"]')).toHaveLength(0);
    expect(page.querySelectorAll('.model-detail-spec-row')).toHaveLength(0);
  });
});
