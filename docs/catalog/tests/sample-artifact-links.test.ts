// @vitest-environment node
import { beforeAll, describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import type { Catalog } from "../src/catalog/types";
import { groupPerformanceMetrics } from "../src/catalog/metric-display";

let catalog: Catalog;
beforeAll(async () => { catalog = await buildMultiplatformCatalog(resolve(process.cwd(), "../..")); });

describe("source-documented X5 artifact associations", () => {
  it.each([
    ["efficient_sam", "efficient-sam-encoder", "efficient_sam_vitt_encoder_512x512_default_none.bin"],
    ["efficient_sam", "efficient-sam-decoder", "efficient_sam_vitt_decoder_fixedprompt_512_default.bin"],
    ["mobile_sam", "mobile-sam-encoder", "mobile_sam_image_encoder_norm_512x512_allint16.bin"],
    ["mobile_sam", "mobile-sam-decoder", "mobile_sam_decoder_512_box_default.bin"],
    ["lprnet", "lprnet", "lpr.bin"],
    ["modnet", "modnet", "modnet_512x512_rgb.bin"],
    ["himloco", "himloco", "himloco_go2_bayese_1x270.bin"]
  ])("%s / %s joins measured rows to %s without an unmeasured duplicate", (sample, prefix, filename) => {
    const variants = catalog.models.find(m => m.id === sample)!.variants!.filter(v => v.hardware === "x5");
    const measured = variants.filter(v => v.benchmarks.some(r => r.variant_id.startsWith(prefix) && (r.performance?.length ?? 0) > 0));
    expect(measured).toHaveLength(1);
    expect(measured[0]!.assets.map(a => a.filename)).toEqual([filename]);
    expect(variants.filter(v => v.assets.some(a => a.filename === filename))).toHaveLength(1);
    expect(measured[0]!.name).not.toMatch(/(?:One|Eight|Single|Two) Threads?/);
    if (sample === "modnet") expect(measured[0]!.assets[0]!.url).toBeFalsy();
  });
  it("uses a family name without repeating its filename acronym", () => {
    const asr = catalog.models.find(model => model.id === "asr")!;
    const variant = asr.variants!.find(value => value.hardware === "s600")!;
    expect(variant.name).toBe(asr.name);
  });
  it.each(["convnext", "edgenext", "efficientformer", "efficientformerv2", "efficientnet", "efficientvit", "fasternet", "fastvit", "googlenet", "hgnetv2", "mobileone", "repghost", "repvgg", "repvit", "resnext"])("%s preserves single-thread latency and multi-thread throughput conditions", sample => {
    const variants = catalog.models.find(model => model.id === sample)!.variants!.filter(variant => variant.hardware === "x5");
    for (const variant of variants) {
      expect(variant.input?.format).toBe("NV12");
      const metrics = variant.benchmarks.flatMap(record => record.performance ?? []);
      expect(metrics.filter(metric => metric.metric === "latency" && metric.scope?.includes("single-thread")).every(metric => metric.concurrency === 1)).toBe(true);
      expect(metrics.filter(metric => metric.metric === "throughput").every(metric => metric.scope?.includes(sample === "convnext" ? "four-thread" : "multi-thread"))).toBe(true);
      expect(groupPerformanceMetrics(variant.benchmarks)).toHaveLength(1);
    }
  });
  it.each(["efficient_sam", "mobile_sam"])("%s attaches validation to the actual encoder and decoder configurations", sample => {
    const variants = catalog.models.find(model => model.id === sample)!.variants!.filter(variant => variant.hardware === "x5");
    expect(variants).toHaveLength(2);
    expect(variants.every(variant => variant.assets.length === 1 && variant.benchmarks.some(record => record.accuracy?.length))).toBe(true);
    expect(variants.flatMap(variant => variant.benchmarks.flatMap(record => record.accuracy ?? []))).toHaveLength(3);
  });
});
