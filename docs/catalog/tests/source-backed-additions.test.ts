import { beforeAll, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { renderModelDetails } from "../src/ui/model-details";
import type { Catalog, HardwareId } from "../src/catalog/types";

let catalog: Catalog;
beforeAll(async () => { catalog = await buildMultiplatformCatalog(resolve("../..")); });
function page(id: string, hardware: HardwareId, task?: string) {
  return renderModelDetails(catalog.models.find(model => model.id === id)!, {
    hardware, task, locale: "en", repositoryUrl: "https://github.com/D-Robotics/rdk_model_zoo", releaseTag: catalog.release.tag
  });
}
it("retains conflicting YOLO26 Depth README timings without replacing evaluator values", () => {
  const records = catalog.models.find(model => model.id === "yolov26")!.benchmarks;
  expect(records.filter(record => record.id.endsWith("-readme-reference")).flatMap(record => record.performance ?? []).map(metric => metric.value).sort((a,b) => a-b)).toEqual([8.1, 10.8, 11, 13.7, 20.6]);
  expect(records.find(record => record.id === "yolo26-depth-l-s100")?.performance?.[0]?.value).toBe(9.79);
  expect(page("yolov26", "s100", "monocular-depth-estimation").textContent).toContain("differs from evaluator latency table");
});
it("shows CLIP text encoder as a related file, separate from the BPU download", () => {
  const result = page("clip", "x5");
  expect(result.querySelector('.model-detail-related-files')?.textContent).toContain("text_encoder.onnx");
  expect(result.querySelector('.model-detail-spec-row')?.textContent).toContain("Image Encoder");
});
it("keeps a tracking pipeline and its detector assets together across all supported boards", () => {
  const tracker = catalog.models.find(model => model.id === "bytetrack")!;
  expect(tracker.variants?.map(variant => variant.hardware).sort()).toEqual(["s100", "s100p", "s600"]);
  expect(catalog.models.find(model => model.id === "yolov5")?.variants?.some(variant => variant.task === "multi-object-tracking")).toBe(false);
  expect(page("bytetrack", "s100").querySelector('.model-detail-spec-row')?.textContent).toContain("≈2.37 ms");
});
it("keeps legacy YOLO11 S600 files out of S100 even when the basename says nashe", () => {
  const model = catalog.models.find(model => model.id === "yolov11")!;
  for (const sample of ["yolo11", "yolo11_pose", "yolo11_seg"]) {
    const component = sample === "yolo11" ? "detect" : sample.replace("yolo11_", "");
    const variants = model.variants!.filter(variant => variant.assets.some(asset => asset.filename.includes(`yolo11n_${component}_nashe_`)));
    expect(variants.map(variant => variant.hardware).sort(), sample).toEqual(["s100", "s600"]);
    for (const variant of variants) expect(variant.assets.filter(asset => /^(s100|s600)\//.test(asset.filename) && asset.filename.includes(`yolo11n_${component}_nashe_`)).every(asset => asset.filename.startsWith(`${variant.hardware}/`))).toBe(true);
  }
});
it("retains all PP-OCRv6 board downloads and separates detection from recognition", () => {
  const platform = catalog.models.find(model => model.id === "paddleocr")!.platforms!.find(platform => platform.platform === "s")!;
  expect(platform.variants).toHaveLength(6);
  for (const hardware of ["s100", "s100p", "s600"]) {
    const variants = platform.variants!.filter(variant => variant.hardware === hardware);
    expect(variants.map(variant => variant.task).sort()).toEqual(["ocr-text-detection", "ocr-text-recognition"]);
    expect(variants.every(variant => variant.assets.length === 1)).toBe(true);
    expect(variants.every(variant => variant.assets[0]!.url!.includes(hardware === "s600" ? "/rdk_s600/" : "/rdk_s100/"))).toBe(true);
  }
});
it("preserves Pi0 fixed-input comparison precision and immutable submodule provenance", () => {
  const result = page("pi0", "s600");
  expect(result.textContent).toContain("0.999981858");
  expect(result.textContent).toContain("not task success rate");
  expect(result.querySelector<HTMLAnchorElement>('.model-detail-sample-link')?.href)
    .toBe("https://github.com/D-Robotics/rdk_LeRobot_tools/blob/a32de276bc1681a2b1531012de111eaa1c16acb6/models/pi0/README.md");
  expect(result.querySelectorAll('[data-action="download-model"]')).toHaveLength(0);
});
it("shows Gemma demo token rate and utilization without presenting token rate as FPS", () => {
  const result = page("gemma4-e2b", "s100p");
  const extra = result.querySelector('.model-detail-additional-metrics');
  expect(extra?.textContent).toContain("≈6.9tokens/s");
  expect(extra?.textContent).toContain("86%");
  expect(extra?.textContent).toContain("Published text-chat demo");
  expect(result.querySelector('.model-detail-related-files')?.textContent).toContain("tok_embeddings.bin");
});
it("keeps ACT explicitly S100 and S600/manual without inventing a numerical benchmark", () => {
  const model = catalog.models.find(model => model.id === "act")!;
  expect(model.variants?.map(variant => variant.hardware)).toEqual(["s100", "s600"]);
  expect(model.benchmarks.flatMap(record => [...record.performance ?? [], ...record.accuracy ?? []])).toEqual([]);
  expect(page("act", "s100").querySelector<HTMLAnchorElement>('.model-detail-sample-link')?.href)
    .toContain("326ea043be204de25223d95c7d918efe8672dc66");
});
it("binds all three Paraformer components to its measured pipeline", () => {
  const model = catalog.models.find(model => model.id === "paraformer")!;
  expect(model.variants).toHaveLength(1);
  expect(model.variants![0]!.assets.map(asset => asset.filename)).toEqual(expect.arrayContaining([
    "s100/paraformer_large_encoder_400x560_s100.hbm",
    "s100/paraformer_large_predictor_400x512_s100.hbm",
    "s100/paraformer_large_decoder_400x512_s100.hbm"
  ]));
  const rendered = page("paraformer", "s100").textContent!;
  for (const expected of ["33.63", "33.15", "40.81", "45.61", "0.007", "0.008", "WAV frontend excluded"]) expect(rendered).toContain(expected);
  expect(rendered).not.toContain("No performance recorded for this configuration");
});
it("attaches the published YOLOv5 v7 nano file without creating an unmeasured duplicate", () => {
  const variants = catalog.models.find(model => model.id === "yolov5")!.variants!
    .filter(variant => variant.hardware === "x5" && variant.sample_path === "samples/vision/yolov5");
  expect(variants).toHaveLength(9);
  expect(variants.find(variant => variant.benchmarks.some(record => record.id === "yolov5-v7-n-x5"))?.assets.map(asset => asset.filename))
    .toEqual(["yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"]);
  expect(variants.some(variant => variant.benchmarks.length === 0)).toBe(false);
});
it("preserves both ends of DINOv2 observed cosine ranges on their actual boards", () => {
  const model = catalog.models.find(model => model.id === "dinov2")!;
  for (const hardware of ["s100", "s100p", "s600"] as const) {
    const range = model.benchmarks.find(record => record.id === `dinov2-${hardware}-board-cosine`)!;
    expect(range.accuracy).toHaveLength(4);
    expect(range.accuracy?.map(metric => metric.statistic)).toEqual(["min", "max", "min", "max"]);
    expect(page("dinov2", hardware).textContent).toContain(hardware === "s600" ? "0.9975" : "0.9977");
  }
  expect(model.benchmarks.find(record => record.id === "dinov2-ptq-nash-e")?.accuracy).toHaveLength(4);
});
it("keeps SigLIP published distribution details shared instead of assigning another board test", () => {
  const model = catalog.models.find(model => model.id === "siglip")!;
  const distributions = model.benchmarks.filter(record => record.id.endsWith("-distribution"));
  expect(distributions).toHaveLength(8);
  expect(distributions.flatMap(record => record.accuracy ?? [])).toHaveLength(48);
  expect(model.variants?.flatMap(variant => variant.benchmarks).some(record => record.id.endsWith("-distribution"))).toBe(false);
  expect(page("siglip", "s100").querySelector('.model-detail-unassigned-evidence')?.textContent).toContain("1% low");
});
it("restores ViT float baselines without dropping the quantized observations", () => {
  const accuracy = catalog.models.find(model => model.id === "vit")!.benchmarks[0]!.accuracy!;
  expect(accuracy.map(metric => [metric.model_stage, metric.value])).toEqual([["quantized",72.62],["quantized",98.03],["float",74.54],["float",98.36]]);
});
it("uses source-script DiffusionDrive URLs and preserves five-case means separately", () => {
  const model = catalog.models.find(model => model.id === "diffusiondrive")!;
  expect(model.assets.every(asset => !asset.url?.includes("/diffusiondrive/"))).toBe(true);
  const aggregate = model.benchmarks.find(record => record.id === "diffusiondrive-s100p-five-case-means")!;
  expect(aggregate.accuracy?.map(metric => metric.value)).toEqual([0.999785,0.997986,0.998799,0.955664,0.819837]);
  expect(aggregate.accuracy?.every(metric => metric.scope?.includes("five packaged cases"))).toBe(true);
});
