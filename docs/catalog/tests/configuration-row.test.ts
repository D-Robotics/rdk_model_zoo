import { beforeAll, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { renderModelDetails } from "../src/ui/model-details";
import type { Catalog } from "../src/catalog/types";
let catalog: Catalog;
beforeAll(async () => { catalog = await buildMultiplatformCatalog(resolve("../..")); });
it("renders every production configuration once, with all source metrics in that row or its details", () => {
  window.history.replaceState({}, "", "/?threads=all");
  for (const model of catalog.models) {
    const variants = model.variants ?? [];
    for (const key of new Set(variants.map(v => `${v.hardware}|${v.task}`))) {
      const selected = variants.filter(v => `${v.hardware}|${v.task}` === key);
      const page = renderModelDetails(model, {hardware:selected[0]!.hardware,task:selected[0]!.task,locale:"en",repositoryUrl:"https://github.com/D-Robotics/rdk_model_zoo",releaseTag:catalog.release.tag});
      const rows = [...page.querySelectorAll<HTMLElement>('.model-detail-spec-row')];
      expect(rows.length, `${model.id}/${key}`).toBe(selected.length);
      for (const variant of selected) {
        const own = rows.filter(row => row.dataset.variantId === variant.id);
        expect(own.length, variant.id).toBe(1);
        const text = own[0]!.textContent! + own[0]!.nextElementSibling!.textContent!;
        for (const record of variant.benchmarks) for (const metric of [...record.performance ?? [], ...record.accuracy ?? []]) {
          expect(text, `${variant.id}/${record.id}/${metric.metric}`).toContain(metric.value.toLocaleString("en-US", {maximumFractionDigits:12}));
        }
      }
    }
  }
}, 30000);
it("keeps YOLOE statistics and SAM accuracy visible without extra configuration rows", () => {
  for (const id of ["yoloe", "efficient_sam", "mobile_sam", "unet", "himloco"]) {
    const model = catalog.models.find(m => m.id === id)!;
    const page = renderModelDetails(model,{hardware:"x5",locale:"en",repositoryUrl:"https://github.com/D-Robotics/rdk_model_zoo",releaseTag:catalog.release.tag});
    const rows = page.querySelectorAll('.model-detail-spec-row');
    expect(rows.length,id).toBe(model.variants!.filter(v=>v.hardware==="x5").length);
    if(id==="yoloe") for(const stat of ["mean","p50","p95"]) expect(rows[0]!.textContent).toContain(stat);
    if(id==="efficient_sam") expect([...rows].map(row=>row.textContent).join(' ')).toContain('0.968013');
  }
});
