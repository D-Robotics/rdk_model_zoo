// @vitest-environment node
import { describe, expect, it } from "vitest";
import { buildModelVariants, getHardwareIds, normalizeHardware } from "../src/catalog/variants";
import type { BenchmarkRecord, ModelRecord } from "../src/catalog/types";

function asset(filename: string, format = filename.split(".").pop() ?? "bin") {
  return {
    filename,
    format,
    url: `https://example.test/${filename}`
  };
}

function benchmark(
  variantId: string,
  hardware: string,
  inputShape: number[] | undefined,
  displayName = variantId,
  assetFilename?: string,
  inputFormat?: string
): BenchmarkRecord {
  return {
    id: `${variantId}-${hardware.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`,
    sample_id: "fixture",
    variant_id: variantId,
    display_name: displayName,
    asset_filename: assetFilename,
    model_format: "hbm",
    precision: "int8",
    input: inputShape ? { shape: inputShape, format: inputFormat } : undefined,
    environment: { hardware, runtime: "fixture" },
    performance: [{ metric: "latency", value: 1, unit: "ms" }],
    source: {
      ref: "fixture-v1.0.0",
      path: "samples/fixture/README.md",
      section: "## Benchmark",
      provenance: "existing-repository-documentation"
    }
  };
}

function model(overrides: Partial<ModelRecord> = {}): ModelRecord {
  return {
    id: "yolov8",
    name: "YOLOv8",
    tasks: ["object-detection", "image-classification"],
    sample_path: "samples/fixture",
    availability: "download",
    download_scripts: [],
    assets: [],
    benchmarks: [],
    ...overrides
  };
}

describe("hardware model variants", () => {
  it("normalizes verified X3, X5, and S-series identities", () => {
    expect(normalizeHardware("RDK X3 & RDK X3 Module (Bernoulli2)")).toBe("x3");
    expect(normalizeHardware("RDK X5")).toBe("x5");
    expect(normalizeHardware("nash-e/yolov8n_detect_nashe.hbm")).toBe("s100");
    expect(normalizeHardware("nash-m/yolov8n_detect_nashm.hbm")).toBe("s100p");
    expect(normalizeHardware("nash-p/yolov8n_detect_nashp.hbm")).toBe("s600");
    expect(normalizeHardware("RDK S100/S100P/S600")).toBeUndefined();
    expect(normalizeHardware("X3/X5")).toBeUndefined();
    expect(normalizeHardware("RDK X3 & RDK X5")).toBeUndefined();
    expect(normalizeHardware("S100 / S100P")).toBeUndefined();
    expect(normalizeHardware("rdk_s100/nash-m/yolov8n_nashm.hbm")).toBe("s100p");
    expect(normalizeHardware("OpenExplorer quantization report")).toBeUndefined();
  });

  it("keeps runnable S artifacts visible when their hardware has no benchmark", () => {
    const variants = buildModelVariants(model({
      tasks: ["object-detection"],
      assets: [
        asset("nash-m/yolov8s_detect_nashm_640x640_nv12.hbm", "hbm"),
        asset("nash-p/yolov8s_detect_nashp_640x640_nv12.hbm", "hbm"),
        asset("nash-p/yolov8s_detect_nashp_640x640_nv12.onnx", "onnx")
      ]
    }));

    expect(variants.map((variant) => variant.hardware)).toEqual(["s100p", "s600"]);
    expect(variants.every((variant) => variant.benchmarks.length === 0)).toBe(true);
    expect(variants.flatMap((variant) => variant.assets).map((entry) => entry.filename))
      .toEqual(expect.arrayContaining([
        "nash-m/yolov8s_detect_nashm_640x640_nv12.hbm",
        "nash-p/yolov8s_detect_nashp_640x640_nv12.hbm"
      ]));
    expect(variants.flatMap((variant) => variant.assets).some((entry) => entry.filename.endsWith(".onnx"))).toBe(false);
  });

  it("joins only the exact task, size, and input tuple", () => {
    const variants = buildModelVariants(model({
      assets: [
        asset("yolov8n_detect_bayese_640x640_nv12.bin"),
        asset("yolov8n_detect_bayese_320x320_rgb.bin"),
        asset("yolov8n_cls_detect_bayese_640x640_nv12.bin"),
        asset("yolov8n_cls_bayese_224x224_nv12.bin")
      ],
      benchmarks: [
        benchmark("yolov8n-detect-640", "RDK X5", [640, 640], "yolov8n-detect-640", undefined, "NV12"),
        benchmark("yolov8n-detect-320", "RDK X5", [320, 320], "yolov8n-detect-320", undefined, "RGB"),
        benchmark("yolov8n-cls-224", "RDK X5", [224, 224])
      ]
    }), { defaultHardware: "x5" });

    const detection = variants.find((variant) => variant.task === "object-detection"
      && variant.benchmarks.some((record) => record.input?.shape?.[0] === 640));
    const lowResolutionDetection = variants.find((variant) => variant.task === "object-detection"
      && variant.benchmarks.some((record) => record.input?.shape?.[0] === 320));
    const classification = variants.find((variant) => variant.task === "image-classification");
    const classifierHead = variants.find((variant) => variant.name.includes("classification head"));

    expect(detection?.assets.map((entry) => entry.filename)).toEqual([
      "yolov8n_detect_bayese_640x640_nv12.bin"
    ]);
    expect(lowResolutionDetection?.assets.map((entry) => entry.filename)).toEqual([
      "yolov8n_detect_bayese_320x320_rgb.bin"
    ]);
    expect(classification?.assets.map((entry) => entry.filename)).toEqual([
      "yolov8n_cls_bayese_224x224_nv12.bin"
    ]);
    expect(classifierHead?.task).toBe("object-detection");
    expect(classifierHead?.assets.map((entry) => entry.filename)).toEqual([
      "yolov8n_cls_detect_bayese_640x640_nv12.bin"
    ]);
    expect(classifierHead?.name).toContain("classification head");
  });

  it("returns the public hardware order from resolved variants", () => {
    const variants = buildModelVariants(model({
      tasks: ["object-detection"],
      assets: [
        asset("nash-p/yolov8n_detect_nashp_640x640_nv12.hbm", "hbm"),
        asset("nash-e/yolov8n_detect_nashe_640x640_nv12.hbm", "hbm"),
        asset("yolov8n_detect_bayese_640x640_nv12.bin", "bin")
      ]
    }), { defaultHardware: "x5" });
    expect(getHardwareIds({ ...model(), variants })).toEqual(["x5", "s100", "s600"]);
  });

  it("gives explicit semantic and OCR detection names precedence over generic tokens", () => {
    const semantic = buildModelVariants(model({
      id: "unet",
      name: "UNet",
      tasks: ["semantic-segmentation", "instance-segmentation"],
      assets: [asset("unet_semantic_seg_1024x2048_nv12.bin")]
    }), { defaultHardware: "x5" });
    const ocr = buildModelVariants(model({
      id: "paddleocr",
      name: "PaddleOCR",
      tasks: ["ocr-text-detection", "object-detection"],
      assets: [asset("paddleocr_ocr_det_640x640_nv12.bin")]
    }), { defaultHardware: "x5" });

    expect(semantic[0]?.task).toBe("semantic-segmentation");
    expect(ocr[0]?.task).toBe("ocr-text-detection");
  });

  it("does not attach two differently sized assets to a benchmark with unknown shape", () => {
    const variants = buildModelVariants(model({
      id: "customdetector",
      name: "Custom detector",
      tasks: ["object-detection"],
      assets: [
        asset("customdetector_detect_640x640_nv12.bin"),
        asset("customdetector_detect_320x320_nv12.bin")
      ],
      benchmarks: [benchmark("customdetector-detect", "RDK X5", undefined)]
    }), { defaultHardware: "x5" });

    const measured = variants.find((variant) => variant.benchmarks.length > 0);
    expect(measured?.assets).toEqual([]);
    expect(variants.filter((variant) => variant.benchmarks.length === 0)).toHaveLength(2);
    expect(variants.flatMap((variant) => variant.assets).map((entry) => entry.filename))
      .toEqual(expect.arrayContaining([
        "customdetector_detect_640x640_nv12.bin",
        "customdetector_detect_320x320_nv12.bin"
      ]));
  });

  it("uses explicit NHWC/NCHW layout for four dimensional inputs and stays conservative without it", () => {
    const assetRecord = asset("customdetector_detect_640x640_nv12.bin");
    const withLayoutBenchmark = benchmark("customdetector-detect", "RDK X5", [1, 640, 640, 3]);
    withLayoutBenchmark.input = { shape: [1, 640, 640, 3], layout: "NHWC", format: "NV12" };
    const withLayout = buildModelVariants(model({
      id: "customdetector",
      name: "Custom detector",
      tasks: ["object-detection"],
      assets: [assetRecord],
      benchmarks: [withLayoutBenchmark]
    }), { defaultHardware: "x5" });
    const withoutLayout = buildModelVariants(model({
      id: "customdetector",
      name: "Custom detector",
      tasks: ["object-detection"],
      assets: [assetRecord],
      benchmarks: [benchmark("customdetector-detect", "RDK X5", [1, 640, 640, 3])]
    }), { defaultHardware: "x5" });

    expect(withLayout.find((variant) => variant.benchmarks.length > 0)?.assets).toHaveLength(1);
    expect(withoutLayout.find((variant) => variant.benchmarks.length > 0)?.assets).toEqual([]);
  });
});
