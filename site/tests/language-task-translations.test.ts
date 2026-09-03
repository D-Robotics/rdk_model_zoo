// @vitest-environment node
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildCatalog } from "../scripts/catalog-builder";
import {
  manifestTaskTranslationKeys,
  taskTranslationKey,
  translations,
  type ManifestTaskId,
  type TranslationKey
} from "../src/i18n/translations";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const EXPECTED_MANIFEST_TASK_TRANSLATION_KEYS = {
  "image-classification": "task.imageClassification",
  "image-text-similarity": "task.imageTextSimilarity",
  "instance-segmentation": "task.instanceSegmentation",
  "legged-locomotion-control": "task.leggedLocomotionControl",
  "license-plate-recognition": "task.licensePlateRecognition",
  "monocular-depth-estimation": "task.monocularDepthEstimation",
  "object-detection": "task.objectDetection",
  "ocr-text-detection": "task.ocrTextDetection",
  "ocr-text-recognition": "task.ocrTextRecognition",
  "open-vocabulary-object-detection": "task.openVocabularyObjectDetection",
  "oriented-bounding-box-detection": "task.orientedBoundingBoxDetection",
  "portrait-matting": "task.portraitMatting",
  "pose-estimation": "task.poseEstimation",
  "promptable-image-segmentation": "task.promptableImageSegmentation",
  "semantic-segmentation": "task.semanticSegmentation"
} as const satisfies Record<ManifestTaskId, TranslationKey>;

describe("manifest task translations", () => {
  it("maps every task in the production manifest to a bilingual translation key", async () => {
    const catalog = await buildCatalog({
      repositoryRoot,
      modelsPath: "release/models.yaml",
      benchmarksPath: "release/benchmarks.yaml",
      modelsSchemaPath: "release/schemas/models.schema.json",
      benchmarksSchemaPath: "release/schemas/benchmarks.schema.json"
    });
    const productionTaskIds = [...new Set(catalog.models.flatMap((model) => model.tasks))].sort();

    expect(productionTaskIds).toEqual(Object.keys(EXPECTED_MANIFEST_TASK_TRANSLATION_KEYS).sort());
    expect(Object.keys(manifestTaskTranslationKeys).sort()).toEqual(productionTaskIds);
    expect(manifestTaskTranslationKeys).toEqual(EXPECTED_MANIFEST_TASK_TRANSLATION_KEYS);

    for (const translationKey of Object.values(manifestTaskTranslationKeys)) {
      expect(translations.en[translationKey]).toBeTruthy();
      expect(translations.zh[translationKey]).toBeTruthy();
    }

    expect(taskTranslationKey("object-detection")).toBe("task.objectDetection");
    expect(taskTranslationKey("future-task")).toBe("task.unknown");
  });
});
