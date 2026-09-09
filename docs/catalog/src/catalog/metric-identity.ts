import type { Locale, MetricUnit } from "./types";

/**
 * Accuracy metric identity and labelling.
 *
 * Source tables across the three release lines name the same measurement
 * differently (`TOP1` / `top-1`, `cosine_similarity` / `cosine-similarity`,
 * `kpt-all mAP` / `keypoints-all-map-50-95`, `map` / `bbox-map` /
 * `bbox-all-map-50-95`). Identity is canonicalised here once, so a single
 * measurement never becomes two table columns, and so the rendered column
 * label follows the wording of the original benchmark table instead of the
 * raw manifest key.
 *
 * Canonicalisation never changes a value, a unit or a scale: it only decides
 * which measurements are the same measurement.
 */

function fold(value: string): string {
  return value.normalize("NFKC").trim().toLocaleLowerCase().replace(/[\s_]+/g, "-");
}

/** Detection/segmentation/pose mAP over the 0.5:0.95 IoU range. */
const MAP_50_95 = /^(bbox|mask|keypoints|pose|kpt)?-?(all|small|medium|large)?-?map(?:-50-95|@?\.?50:?\.?95)?$/;

/** Canonical name for the mAP family; `kind` defaults to bbox for detection. */
function canonicalMap(kind: string | undefined, size: string | undefined): string {
  const resolvedKind = kind === undefined || kind === "" ? "bbox" : kind === "kpt" ? "keypoints" : kind;
  const resolvedSize = size === undefined || size === "" || size === "all" ? "all" : size;
  return `${resolvedKind}-${resolvedSize}-map-50-95`;
}

/**
 * Canonical accuracy metric name. Unknown names are returned folded so that
 * casing, spacing and underscore differences cannot split one metric in two.
 */
export function canonicalMetricName(metric: string): string {
  const folded = fold(metric);
  switch (folded) {
    case "top1":
    case "top-1-accuracy":
      return "top-1";
    case "top5":
    case "top-5-accuracy":
      return "top-5";
    case "cosine-similarity":
    case "cosine":
      return "cosine-similarity";
    case "map":
    case "map-50-95":
    case "bbox-map":
      return "bbox-all-map-50-95";
    case "mask-map":
      return "mask-all-map-50-95";
    default:
      break;
  }
  const map = MAP_50_95.exec(folded);
  if (map) return canonicalMap(map[1], map[2]);
  return folded;
}

const SIZE_LABELS: Record<string, string> = {
  all: "",
  small: " (small)",
  medium: " (medium)",
  large: " (large)"
};

const KIND_LABELS: Record<string, string> = {
  bbox: "bbox mAP@0.5:0.95",
  mask: "mask mAP@0.5:0.95",
  keypoints: "pose mAP@0.5:0.95",
  pose: "pose mAP@0.5:0.95"
};

/** Title-case a folded metric name for display, e.g. `bev-mean-iou`. */
function titleize(value: string): string {
  return value
    .split("-")
    .filter(Boolean)
    .map((token) => {
      const upper = token.toLocaleUpperCase();
      // Keep well-known acronyms intact instead of capitalising one letter.
      if (["IOU", "MSE", "MAE", "RMSE", "CER", "WER", "FPS", "PSNR", "SSIM", "BEV", "OCR"].includes(upper)) return upper;
      return token.charAt(0).toLocaleUpperCase() + token.slice(1);
    })
    .join(" ");
}

/**
 * Human label for a canonical accuracy metric, following the wording used by
 * the source benchmark tables.
 */
export function metricDisplayLabel(canonicalMetric: string, locale: Locale): string {
  const map = /^(bbox|mask|keypoints|pose)-(all|small|medium|large)-map-50-95$/.exec(canonicalMetric);
  if (map) {
    const kind = KIND_LABELS[map[1]!] ?? `${map[1]} mAP@0.5:0.95`;
    const size = SIZE_LABELS[map[2]!] ?? "";
    return `${kind}${size}`;
  }
  switch (canonicalMetric) {
    case "top-1":
      return locale === "zh" ? "Top-1 精度" : "Top-1";
    case "top-5":
      return locale === "zh" ? "Top-5 精度" : "Top-5";
    case "cosine-similarity":
      return locale === "zh" ? "余弦相似度" : "Cosine similarity";
    case "retention":
      return locale === "zh" ? "保持率" : "Retention";
    default:
      break;
  }
  // `<subject>-cosine-similarity` keeps its subject, e.g. `bev-cosine-similarity`.
  const subject = /^(.*)-cosine-similarity$/.exec(canonicalMetric);
  if (subject?.[1]) {
    return `${titleize(subject[1])} ${locale === "zh" ? "余弦相似度" : "cosine similarity"}`;
  }
  return titleize(canonicalMetric);
}

/**
 * Scale hint shown once in a column header instead of a `%` on every cell.
 * Values are displayed exactly as the source records them, so the reader needs
 * to know whether `0.342` and `34.2` are the same scale.
 */
export function unitScaleLabel(unit: MetricUnit, locale: Locale): string {
  if (unit === "percent") return "%";
  if (unit === "ratio") return locale === "zh" ? "0–1" : "0–1";
  if (unit === "fps") return "FPS";
  return unit;
}

/** Retention is the only accuracy value that is rendered with a `%` suffix. */
export function isRetentionMetricName(metric: string): boolean {
  const folded = fold(metric);
  return folded === "retention" || folded.endsWith("-retention") || folded.includes("retention");
}
