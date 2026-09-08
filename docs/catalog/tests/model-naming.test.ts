// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { officialFamilyName, officialYoloName, stripHardwareSuffix } from "../src/catalog/model-naming";

const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));

describe("official model naming", () => {
  it("keeps the v only for the YOLO majors that officially use it", () => {
    // samples/vision/ultralytics_yolo/README.md publishes:
    //   YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13
    // and ultralytics_yolo26/README.md publishes "YOLO26".
    expect([5, 8, 9, 10].map((version) => officialYoloName(String(version))))
      .toEqual(["YOLOv5", "YOLOv8", "YOLOv9", "YOLOv10"]);
    expect([11, 12, 13, 26].map((version) => officialYoloName(String(version))))
      .toEqual(["YOLO11", "YOLO12", "YOLO13", "YOLO26"]);
  });

  it("corrects family names that disagree with the sample README title", () => {
    expect(officialFamilyName("resnet18", "Resnet18")).toBe("ResNet18");
    expect(officialFamilyName("paddle_ocr", "PaddleOCR v6")).toBe("PaddleOCR");
    // Unknown families keep the name published by their manifest.
    expect(officialFamilyName("convnext", "ConvNeXt")).toBe("ConvNeXt");
  });

  it("removes trailing hardware and input-size qualifiers without eating the model name", () => {
    expect(stripHardwareSuffix("YOLOv8n Detect 640x640 on RDK S100")).toBe("YOLOv8n Detect");
    expect(stripHardwareSuffix("SigLIP base patch16 224 on RDK S100P")).toBe("SigLIP base patch16 224");
    expect(stripHardwareSuffix("YOLOv10n Detect 640x640")).toBe("YOLOv10n Detect");
    // A name that is only a qualifier must not collapse to nothing.
    expect(stripHardwareSuffix("on RDK S100")).toBe("on RDK S100");
    // A model version such as YOLO26 is not an input size.
    expect(stripHardwareSuffix("YOLO26n Detect 640x640")).toBe("YOLO26n Detect");
  });
});

describe("production catalog naming", () => {
  it("publishes official names and one card per model family", async () => {
    const catalog = await buildMultiplatformCatalog(repositoryRoot);
    const nameById = new Map(catalog.models.map((model) => [model.id, model.name]));

    expect(nameById.get("yolov26")).toBe("YOLO26");
    expect(nameById.get("yolov11")).toBe("YOLO11");
    expect(nameById.get("yolov12")).toBe("YOLO12");
    expect(nameById.get("yolov13")).toBe("YOLO13");
    expect(nameById.get("yolov8")).toBe("YOLOv8");
    expect(nameById.get("yolov5")).toBe("YOLOv5");
    expect(nameById.get("resnet18")).toBe("ResNet18");
    expect(resnetName(catalog.models, "resnet50")).toBe("ResNet50");
    expect(resnetName(catalog.models, "resnet152")).toBe("ResNet152");

    // Two sample slugs publishing one model must not become two identical cards.
    const names = catalog.models.map((model) => model.name);
    const duplicates = names.filter((name, index) => names.indexOf(name) !== index);
    expect(duplicates).toEqual([]);

    const paddleocr = catalog.models.find((model) => model.name === "PaddleOCR");
    const hardware = new Set((paddleocr?.variants ?? []).map((variant) => variant.hardware));
    expect(paddleocr?.id).toBe("paddleocr");

    // A task id must be one the catalog knows. The S manifest once labelled
    // YOLO26 OBB "oriented-object-detection", which taskFor() could not match,
    // so OBB rows fell back to the detection task and appeared inside the
    // detection table.
    const yolo26 = catalog.models.find((model) => model.id === "yolov26");
    const tasks = new Set((yolo26?.variants ?? []).map((variant) => variant.task));
    for (const task of tasks) {
      expect(["object-detection", "instance-segmentation", "pose-estimation",
        "image-classification", "monocular-depth-estimation",
        "oriented-bounding-box-detection"]).toContain(task);
    }
    expect(tasks.has("oriented-bounding-box-detection")).toBe(true);
    const obbVariants = (yolo26?.variants ?? []).filter((variant) => variant.task === "oriented-bounding-box-detection");
    expect(obbVariants.length).toBeGreaterThan(0);
    for (const variant of obbVariants) {
      expect(variant.task).not.toBe("object-detection");
    }
    expect([...hardware].sort()).toEqual(["s100", "x3", "x5"]);
  });
});

function resnetName(models: Array<{ id: string; name: string }>, id: string): string | undefined {
  return models.find((model) => model.id === id)?.name;
}
