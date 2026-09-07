import type { HardwareId, ModelRecord, ModelVariant } from "./types";
import { getHardwareIds, getModelVariants, normalizeHardware } from "./variants";

/** Decorative categories used by the card artwork. They carry no model data. */
export type CardVisualKind =
  | "detection"
  | "segmentation"
  | "classification"
  | "pose"
  | "text"
  | "audio"
  | "robotics"
  | "language"
  | "general";

export interface ModelCardViewModel {
  modelId: string;
  name: string;
  tasks: string[];
  specifications: string[];
  hardware: HardwareId[];
  variantCount: number;
  visualKind: CardVisualKind;
  /** The active hardware scope, or an empty string when the catalog is unscoped. */
  platform: HardwareId | "";
}

function normalized(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase();
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.trim().length > 0))];
}

function selectedHardware(platform: string): HardwareId | "" {
  if (!platform.trim()) return "";
  return normalizeHardware(platform) ?? "";
}

function isYoloFamily(model: ModelRecord): boolean {
  return /^yolov?\d+$/i.test(model.id);
}

/**
 * Turns source variant names into short, family-level specification labels.
 * Source naming conventions belong here so the DOM renderer never has to
 * infer a model identity from a display string.
 */
function specificationName(model: ModelRecord, variant: ModelVariant): string {
  if (isYoloFamily(model)) {
    return /^yolov?\d+[a-z]*/i.exec(variant.name)?.[0] ?? variant.name;
  }
  return variant.name.replace(/\s+(?:on\s+)?RDK\s+.*$/i, "");
}

function sortSpecifications(model: ModelRecord, names: string[]): string[] {
  if (!isYoloFamily(model)) return names;
  const sizeOrder = ["n", "s", "m", "l", "x", "b", "c", "e"];
  const rank = (name: string): number => {
    const size = /\d+([a-z])/i.exec(name)?.[1]?.toLowerCase();
    return size ? sizeOrder.indexOf(size) : 99;
  };
  return [...names].sort((left, right) => rank(left) - rank(right) || left.localeCompare(right));
}

function visualKind(tasks: string[]): CardVisualKind {
  const text = tasks.map(normalized).join(" ");
  if (/segmentation|semantic-segmentation|instance-segmentation/.test(text)) return "segmentation";
  if (/pose/.test(text)) return "pose";
  if (/ocr|text-detection|text-recognition/.test(text)) return "text";
  if (/object-detection|oriented-bounding-box|detection/.test(text)) return "detection";
  if (/classification|classify/.test(text)) return "classification";
  if (/audio|speech|sound|voice/.test(text)) return "audio";
  if (/robot|locomotion|control|depth|stereo/.test(text)) return "robotics";
  if (/language|llm|multimodal|embedding|caption|question-answer/.test(text)) return "language";
  return "general";
}

function scopedVariants(model: ModelRecord, platform: HardwareId | ""): ModelVariant[] {
  const variants = getModelVariants(model);
  return platform ? variants.filter((variant) => variant.hardware === platform) : variants;
}

/** Builds the presentation model used by a single model-family card. */
export function buildModelCardViewModel(model: ModelRecord, platform = ""): ModelCardViewModel {
  const activePlatform = selectedHardware(platform);
  const variants = scopedVariants(model, activePlatform);
  const taskSource = variants.length > 0 || activePlatform ? variants.map((variant) => variant.task) : model.tasks;
  const specifications = sortSpecifications(
    model,
    unique(variants.map((variant) => specificationName(model, variant)))
  );

  return {
    modelId: model.id,
    name: model.name,
    tasks: unique(taskSource),
    specifications,
    hardware: getHardwareIds(model),
    variantCount: variants.length,
    visualKind: visualKind(taskSource),
    platform: activePlatform
  };
}
