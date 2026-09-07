// @vitest-environment node
import { describe, expect, it } from "vitest";
import { correctSArtifacts } from "../scripts/source-corrections";
import { createModelFixture, benchmarkFixture } from "./fixtures/catalog";

describe("tagged S artifact compatibility corrections", () => {
  it("uses original download-script spellings and SoC directories without changing benchmark identity", () => {
    const filename = "nash-p/yolo8n_detect_nashp_640x640_nv12.hbm";
    const models = { models: [createModelFixture({ id: "ultralytics_yolo", assets: [{ filename, format: "hbm",
      url: `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/${filename}` }] })] };
    const record = benchmarkFixture({ sample_id: "ultralytics_yolo", asset_filename: filename });
    correctSArtifacts(models, { benchmarks: [record] });
    expect(models.models[0]!.assets[0]!.filename).toBe("nash-p/yolov8n_detect_nashp_640x640_nv12.hbm");
    expect(models.models[0]!.assets[0]!.url).toContain("/rdk_s600/Ultralytics_YOLO_OE_3.7.0/nash-p/yolov8n_");
    expect(record.asset_filename).toBe(models.models[0]!.assets[0]!.filename);
    expect(record.variant_id).toBe("convnext-atto-224");
  });
  it("does not advertise the undocumented YOLOv9n or rewrite unrelated samples", () => {
    const asset = { filename: "nash-e/yolo9n_detect_nashe_640x640_nv12.hbm", format: "hbm" };
    const models = { models: [createModelFixture({ id: "ultralytics_yolo", assets: [asset] }), createModelFixture({ id: "unrelated", assets: [asset] })] };
    correctSArtifacts(models, { benchmarks: [] });
    expect(models.models[0]!.assets).toHaveLength(0);
    expect(models.models[1]!.assets[0]!.filename).toBe(asset.filename);
  });
});
