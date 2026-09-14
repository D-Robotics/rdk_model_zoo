import { beforeAll, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { renderModelDetails } from "../src/ui/model-details";
import type { Catalog } from "../src/catalog/types";
let catalog: Catalog;
beforeAll(async () => { catalog = await buildMultiplatformCatalog(resolve("../..")); });
for (const hardware of ["x5", "s600"] as const) for (const locale of ["zh", "en"] as const) {
  it(`keeps YOLOv8n ${hardware}/${locale} BPU and CPU measurements in one model row`, () => {
    const model = catalog.models.find(model => model.id === "yolov8")!;
    const variant = model.variants!.find(v => v.hardware === hardware && v.task === "object-detection" && /yolov8n-detect/.test(v.id))!;
    const page = renderModelDetails(model, {hardware, locale, task:"object-detection", repositoryUrl:"https://github.com/D-Robotics/rdk_model_zoo",releaseTag:catalog.release.tag});
    const rows = [...page.querySelectorAll<HTMLElement>('.model-detail-spec-row')].filter(row => row.dataset.variantId === variant.id);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.querySelector('.model-detail-postprocess')?.textContent).toContain(hardware === "x5" ? "5 ms" : "2 ms");
    expect(rows[0]!.querySelector('.model-detail-latency')?.textContent).toContain(hardware === "x5" ? "7 ms" : "0.776 ms");
    expect(page.querySelectorAll('thead [data-concurrency="unknown"]')).toHaveLength(0);
    expect(rows[0]!.querySelector('.model-detail-specification')?.textContent).not.toContain('single-core CPU');
  });
}
