/**
 * Official model-family naming.
 *
 * The catalog must show the name each project uses for itself, not a name
 * synthesised from a directory slug. Ultralytics dropped the `v` from YOLO11
 * onward, so `YOLOv26` is wrong while `YOLOv10` is right; the sample READMEs
 * in this repository publish the authoritative spellings:
 *
 *   samples/vision/ultralytics_yolo/README.md
 *     Detection: YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13
 *   samples/vision/ultralytics_yolo26/README.md  -> "YOLO26 Model Description"
 *   samples/vision/resnet18/README.md            -> "ResNet18 Model Description"
 *   samples/vision/paddle_ocr/README.md          -> "PaddleOCR Model Description"
 */

/** YOLO major versions that officially carry a `v`, per the source READMEs. */
const YOLO_WITH_V = new Set(["5", "8", "9", "10"]);

/** Family ids whose manifest name disagrees with the sample README title. */
const FAMILY_NAME_CORRECTIONS: Record<string, string> = {
  resnet18: "ResNet18",
  resnet50: "ResNet50",
  resnet152: "ResNet152",
  paddle_ocr: "PaddleOCR",
  paddleocr: "PaddleOCR"
};

/**
 * Sample directories that publish one model family under different slugs on
 * different release lines. Without this the catalog shows two cards with the
 * same official name and splits their hardware support.
 */
const FAMILY_ID_ALIASES: Record<string, string> = {
  // X3/X5 `paddleocr` and S `paddle_ocr` are the same PaddleOCR family.
  paddle_ocr: "paddleocr"
};

/** Canonical family id, so one model is one card across release lines. */
export function canonicalFamilyId(familyId: string): string {
  return FAMILY_ID_ALIASES[familyId] ?? familyId;
}

/** Official display name for a YOLO major version, e.g. `26` -> `YOLO26`. */
export function officialYoloName(version: string): string {
  return YOLO_WITH_V.has(version) ? `YOLOv${version}` : `YOLO${version}`;
}

/**
 * Official family display name. Family ids stay unchanged so published deep
 * links (`?model=yolov26`) keep working; only the label is corrected.
 */
export function officialFamilyName(familyId: string, fallbackName: string): string {
  const yolo = /^yolov?(\d+)$/.exec(familyId);
  if (yolo) return officialYoloName(yolo[1]!);
  return FAMILY_NAME_CORRECTIONS[familyId] ?? fallbackName;
}

/**
 * Removes qualifiers a row name should not repeat: the hardware ("on RDK
 * S100", already stated by the hardware tab) and a trailing input size
 * ("640x640", already stated by the input-size column). "YOLOv10n Detect
 * 640x640 on RDK S100" becomes "YOLOv10n Detect".
 */
export function stripHardwareSuffix(name: string): string {
  const stripped = name
    .replace(/[\s·|-]*(?:on\s+)?RDK\s+[A-Za-z0-9-]+\s*$/i, "")
    // A trailing input size such as "640x640"; a bare number ("224" in
    // "SigLIP base patch16 224") is part of the model name, not a size.
    .replace(/[\s·|-]+[0-9]{2,4}\s*[x×*]\s*[0-9]{2,4}\s*$/i, "")
    .trim();
  return stripped || name.trim();
}
