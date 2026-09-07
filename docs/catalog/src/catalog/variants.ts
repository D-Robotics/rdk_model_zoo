import type {
  BenchmarkRecord,
  ModelRecord,
  ModelVariant,
  PlatformModelRecord
} from "./types";

export const HARDWARE_IDS = ["x3", "x5", "s100", "s100p", "s600"] as const;

const RUNNABLE_FORMATS = new Set(["bin", "hbm"]);

type HardwareId = (typeof HARDWARE_IDS)[number];
type AssetRecord = ModelRecord["assets"][number];

export interface BuildVariantOptions {
  /** Hardware implied by a release line whose files have no board directory. */
  defaultHardware?: HardwareId;
  releaseTag?: string;
}

function normalized(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase();
}

/**
 * Converts source manifest hardware labels and verified S-series artifact
 * conventions to the five public hardware ids. Compound or toolchain-only
 * labels intentionally stay unmapped.
 */
export function normalizeHardware(value: string): HardwareId | undefined {
  const text = normalized(value).trim();
  if (!text) return undefined;

  // These describe a comparison or the OpenExplorer compiler, not a board.
  if (
    /openexplore|toolchain|quantization report/.test(text)
    || /s100\s*\/\s*s100p\s*\/\s*s600/.test(text)
    || /x3\s*\/\s*x5|x5\s*\/\s*x3/.test(text)
  ) {
    return undefined;
  }

  // Chip-specific paths override the shared rdk_s100 archive directory.
  const chips = [...new Set([...text.matchAll(/(?:^|[^a-z0-9])nash[-_]?([emp])(?=[^a-z0-9]|$)/g)].map((match) => match[1]))];
  if (chips.length > 1) return undefined;
  if (chips.length === 1) return ({ e: "s100", m: "s100p", p: "s600" } as const)[chips[0] as "e" | "m" | "p"];
  const boards = HARDWARE_IDS.filter((hardware) =>
    new RegExp(`(^|[^a-z0-9])${hardware}([^a-z0-9]|$)`).test(text)
    || (hardware === "x3" && text.includes("bernoulli2"))
  );
  return boards.length === 1 ? boards[0] : undefined;
}

export function isRunnableAsset(asset: AssetRecord): boolean {
  const format = normalized(asset.format).replace(/^\./, "");
  if (RUNNABLE_FORMATS.has(format)) return true;
  const filename = normalized(asset.filename);
  return /\.(?:bin|hbm)$/.test(filename);
}

function assetHardware(asset: AssetRecord): HardwareId | undefined {
  return normalizeHardware(`${asset.filename} ${asset.url ?? ""}`);
}

function assetKey(asset: AssetRecord): string {
  return asset.url ?? asset.filename;
}

function uniqueAssets(assets: AssetRecord[]): AssetRecord[] {
  const seen = new Set<string>();
  return assets.filter((asset) => {
    const key = asset.url ?? asset.filename;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function familyFromText(value: string): { id: string; name: string } | undefined {
  const text = normalized(value);
  const yolo = /yolov?(\d+)/.exec(text);
  if (yolo) return { id: `yolov${yolo[1]}`, name: `YOLOv${yolo[1]}` };
  // Treat MobileNet as a family only when it is a standalone model token.
  // Semantic-segmentation samples such as `unet_mobilenet` use MobileNet as
  // a backbone and must not be pulled into the MobileNet classifier card.
  if (
    !/(^|[^a-z0-9])unet[_-]?mobilenet/.test(text)
    && /(^|[^a-z0-9])mobilenet(?:v?\d+)?(?=$|[^a-z0-9])/.test(text)
  ) {
    return { id: "mobilenet", name: "MobileNet" };
  }
  return undefined;
}

function taskCandidates(value: string): string[] {
  const text = normalized(value);
  const candidates: string[] = [];
  // Ultralytics files named `*_cls_detect_*` are detector classifier-head
  // artifacts. Keep them under object detection instead of reclassifying the
  // file as an image-classification model merely because `cls` appears first.
  if (/(^|[-_])cls[-_]?detect([-_.]|$)/.test(text)) {
    candidates.push("object-detection");
  }
  if (/(^|[-_])(?:obb|oriented|rotated)(?:[-_]|$)/.test(text)) {
    candidates.push("oriented-bounding-box-detection");
  }
  if (/(^|[-_])semantic[-_]?seg(?:mentation)?([-_.]|$)/.test(text)) {
    candidates.push("semantic-segmentation", "instance-segmentation");
  } else if (text.includes("instance_seg")) {
    candidates.push("instance-segmentation", "semantic-segmentation");
  } else if (/(^|[-_])seg(?:mentation)?([-_.]|$)/.test(text)) {
    candidates.push("instance-segmentation", "semantic-segmentation");
  }
  if (/(^|[-_])pose([-_.]|$)/.test(text)) candidates.push("pose-estimation");
  if (/(^|[-_])cls(?:assification)?([-_.]|$)/.test(text)) candidates.push("image-classification");
  if (/(^|[-_])(?:ocr|ppocr)[-_]?(?:det|detect|detection)([-_.]|$)/.test(text)) {
    candidates.push("ocr-text-detection", "object-detection");
  } else if (/(^|[-_])det(?:ect|ection)?([-_.]|$)/.test(text)) {
    candidates.push("object-detection", "ocr-text-detection");
  }
  if (/(^|[-_])rec(?:ognition)?([-_.]|$)/.test(text)) candidates.push("ocr-text-recognition");
  return candidates;
}

function taskFor(model: ModelRecord, recordText: string): string {
  const task = taskCandidates(recordText).find((candidate) => model.tasks.includes(candidate));
  return task ?? model.tasks[0] ?? "unknown";
}

function inferredShape(value: string): number[] | undefined {
  const match = /(^|[^0-9])(\d{2,4})x(\d{2,4})(?=$|[^0-9])/i.exec(value);
  if (match) return [Number(match[2]), Number(match[3])];
  // Benchmark variant ids commonly shorten a square input to `-640` or
  // `-224`; accept only the dimensions used by the published artifacts so a
  // model version such as YOLO26 is never treated as an input size.
  const square = /(^|[-_])((?:224|240|256|260|300|320|380|384|448|512|640|672|768|896|1024))(?=$|[-_.])/i.exec(value);
  return square ? [Number(square[2]), Number(square[2])] : undefined;
}

function inferredInputFormat(value: string): string | undefined {
  const text = normalized(value);
  if (text.includes("nv12")) return "nv12";
  if (text.includes("rgb")) return "rgb";
  return undefined;
}

function inputFor(record: BenchmarkRecord | undefined, fallbackText: string): BenchmarkRecord["input"] {
  if (record?.input) return record.input;
  const shape = inferredShape(fallbackText);
  const format = inferredInputFormat(fallbackText);
  return shape || format ? { shape, format } : undefined;
}

function stripMeasurementSuffix(value: string): string {
  return value
    .replace(/-(?:x3|x5|s100p?|s600)$/i, "")
    .replace(/-(?:performance|accuracy)$/i, "")
    .replace(/-(?:single|two|four|eight|twelve)-thread$/i, "");
}

function variantIdentity(record: BenchmarkRecord): string {
  const value = stripMeasurementSuffix(record.variant_id.trim());
  const shape = shapeKey(record.input?.shape, record.input?.layout);
  const dimensionsPresent = shape?.split("x").every((dimension) => value.includes(dimension)) ?? false;
  return shape && !dimensionsPresent ? `${value}-${shape}` : value;
}

function compact(value: string): string {
  return normalized(value)
    .replace(/(\d)x(?=\d)/g, "$1")
    .replace(/yolov/g, "yolo")
    .replace(/[^a-z0-9]+/g, "");
}

function taskTokens(task: string): string[] {
  switch (task) {
    case "object-detection":
    case "ocr-text-detection":
      return ["detection", "detect", "det"];
    case "oriented-bounding-box-detection":
      return ["oriented", "rotated", "obb", "detection", "detect", "det"];
    case "instance-segmentation":
    case "semantic-segmentation":
      return ["segmentation", "segment", "seg", "instance"];
    case "pose-estimation":
      return ["pose"];
    case "image-classification":
      return ["classification", "classify", "cls", "class"];
    case "ocr-text-recognition":
      return ["recognition", "recognize", "rec"];
    default:
      return [];
  }
}

function shapeKey(shape: number[] | undefined, layout?: string): string | undefined {
  if (!shape || shape.length === 0) return undefined;
  let spatial: number[];
  if (shape.length === 4) {
    const normalizedLayout = normalized(layout ?? "").replace(/[^a-z]/g, "");
    if (normalizedLayout === "nhwc") spatial = [shape[1]!, shape[2]!];
    else if (normalizedLayout === "nchw") spatial = [shape[2]!, shape[3]!];
    else spatial = shape;
  } else {
    spatial = shape.length >= 2 ? shape.slice(-2) : shape;
  }
  if (spatial.some((value) => !Number.isFinite(value) || value <= 0)) return undefined;
  return spatial.join("x");
}

function shapeFromText(value: string): string | undefined {
  return shapeKey(inferredShape(value));
}

function inputSignature(input: BenchmarkRecord["input"] | undefined): string {
  const shape = shapeKey(input?.shape, input?.layout) ?? "";
  const format = normalized(input?.format ?? "").replace(/[^a-z0-9]+/g, "");
  const layout = normalized(input?.layout ?? "").replace(/[^a-z0-9]+/g, "");
  return `shape=${shape};format=${format};layout=${layout}`;
}

/**
 * Produces a deterministic tuple from a benchmark id or a published asset
 * filename. Both X5 `bayese` files and S `nash-*` files are source naming
 * conventions; no basename or prefix ranking is used here.
 */
function canonicalTuple(
  value: string,
  familyId: string,
  task: string,
  input?: BenchmarkRecord["input"]
): string {
  let text = compact(value)
    .replace(/nashe|nashm|nashp/g, "")
    // Some S sample records use a board directory (`s100/...`) while the
    // Ultralytics package uses the equivalent nash directory. The directory
    // is hardware metadata, not a model specifier, and is already represented
    // by the variant hardware field.
    .replace(/s100p|s100|s600/g, "")
    .replace(/bernoulli2/g, "")
    .replace(/bayese/g, "")
    .replace(/(?:single|multi|two|four|eight|twelve)thread/g, "")
    .replace(/evaluator|modified/g, "")
    .replace(/^enppocr/, "ppocr")
    .replace(/tag/g, "")
    .replace(/(?:bin|hbm|onnx)$/g, "")
    .replace(/nv12|rgb/g, "");

  const compactFamily = compact(familyId);
  if (compactFamily.startsWith("yolo")) {
    const version = /yolo(\d+)/.exec(compactFamily)?.[1];
    if (version) text = text.replace(`yolo${version}`, "");
  } else {
    text = text.replace(compactFamily, "");
  }

  for (const token of taskTokens(task)) text = text.replace(token, "");
  // Fixed source conventions used by the SAM package filenames.
  text = text.replace(/vitt|imageencodernorm|imageencoder/g, "encoder");
  const actualShape = shapeKey(input?.shape, input?.layout) ?? shapeFromText(value);
  if (actualShape) {
    for (const dimension of actualShape.split("x")) text = text.replace(dimension, "");
  }
  // The published X5 MobileNetV3 artifact is named simply
  // `MobileNetV3_224x224_nv12.bin`, while the benchmark calls the same
  // model `mobilenetv3-large-224`. The sample README explicitly identifies
  // that artifact as MobileNetV3-Large, so this is a source-specific exact
  // alias rather than a prefix or score-based match.
  if (compactFamily === "mobilenet" && text === "v3") text = "v3large";
  const actualFormat = normalized(input?.format ?? "") || inferredInputFormat(value) || "";
  const actualLayout = normalized(input?.layout ?? "") || "";
  return `${familyId}|${task}|${text}|shape=${actualShape ?? ""}|format=${actualFormat}|layout=${actualLayout}`;
}

function benchmarkTuple(seed: VariantSeed): string {
  return canonicalTuple(seed.identity, seed.familyId, seed.task, seed.input);
}

function assetTuple(asset: AssetRecord, seed: VariantSeed): string {
  return canonicalTuple(asset.filename, seed.familyId, seed.task, inputFor(undefined, asset.filename));
}

function assetVariantIdentity(
  asset: AssetRecord,
  familyId: string,
  task: string,
  input?: BenchmarkRecord["input"]
): string {
  const tuple = canonicalTuple(asset.filename, familyId, task, input);
  const suffix = tuple.split("|")[2] || "variant";
  const shape = shapeKey(input?.shape, input?.layout) ?? shapeFromText(asset.filename);
  const format = normalized(input?.format ?? "") || inferredInputFormat(asset.filename);
  // Keep the family/version token in an asset-only id. The tuple itself is
  // still used for association; this name is only for stable, readable URLs
  // and UI keys such as `yolov8-ncls-640x640-nv12`.
  return [familyId, suffix, shape, format].filter(Boolean).join("-");
}

function tupleField(tuple: string, key: string): string | undefined {
  const value = tuple.split("|").find((part) => part.startsWith(`${key}=`));
  return value?.slice(key.length + 1) || undefined;
}

function tuplesCompatible(left: string, right: string): boolean {
  const leftParts = left.split("|");
  const rightParts = right.split("|");
  // Family, task, and parsed model spec are required exact matches. Shape,
  // format, and layout are metadata fields: an omitted value is unknown and
  // can match the known side, while two contradictory values cannot.
  if (leftParts.slice(0, 3).some((part, index) => part !== rightParts[index])) return false;
  return ["shape", "format", "layout"].every((key) => {
    const leftValue = tupleField(left, key);
    const rightValue = tupleField(right, key);
    return leftValue === undefined || rightValue === undefined || leftValue === rightValue;
  });
}

function tuplesConflict(left: string, right: string): boolean {
  return ["shape", "format", "layout"].some((key) => {
    const leftValue = tupleField(left, key);
    const rightValue = tupleField(right, key);
    return leftValue !== undefined && rightValue !== undefined && leftValue !== rightValue;
  });
}

function taskLabel(task: string): string {
  return {
    "object-detection": "Detect",
    "oriented-bounding-box-detection": "OBB",
    "instance-segmentation": "Seg",
    "semantic-segmentation": "Seg",
    "pose-estimation": "Pose",
    "image-classification": "Classify",
    "ocr-text-detection": "Text Detect",
    "ocr-text-recognition": "Text Recognize"
  }[task] ?? task;
}

function assetDisplayName(asset: AssetRecord, model: ModelRecord, task: string): string {
  const basename = asset.filename.split(/[\\/]/).pop() ?? asset.filename;
  const family = familyFromText(asset.filename);
  const shape = shapeFromText(asset.filename);
  const classifierHead = /(?:^|[_-])cls[_-]?detect(?:[-_.]|$)/i.test(basename);
  if (family?.id.startsWith("yolov")) {
    const match = /yolov?(\d+)([a-z]+)/i.exec(basename);
    if (match) {
      const version = `YOLOv${match[1]}${match[2]}`;
      const label = classifierHead ? `${taskLabel(task)} (classification head)` : taskLabel(task);
      return [version, label, shape].filter(Boolean).join(" ");
    }
  }
  if (family?.id === "mobilenet") {
    const match = /mobilenet(v?\d+)(?:[_-]([a-z]+))?/i.exec(basename);
    if (match) {
      const version = match[1];
      if (!version) return model.name;
      return [`MobileNetv${version.replace(/^v/i, "")}`, match[2], shape]
        .filter(Boolean)
        .join(" ");
    }
  }
  const readable = basename
    .replace(/\.(?:bin|hbm)$/i, "")
    .replace(/(?:nashe|nashm|nashp|bayese)/gi, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return readable ? `${model.name} · ${readable}` : model.name;
}

interface VariantSeed {
  key: string;
  familyId: string;
  identity: string;
  name: string;
  task: string;
  hardware: HardwareId;
  input?: BenchmarkRecord["input"];
  records: BenchmarkRecord[];
  assets: AssetRecord[];
}

function modelFamilyId(model: ModelRecord): string {
  return familyFromText(model.id)?.id ?? normalized(model.id);
}

function seedFor(
  seeds: Map<string, VariantSeed>,
  model: ModelRecord,
  record: BenchmarkRecord,
  hardware: HardwareId
): VariantSeed {
  const task = taskFor(model, `${record.variant_id} ${record.display_name}`);
  const identity = variantIdentity(record);
  const input = inputFor(record, `${record.variant_id} ${record.display_name}`);
  const key = `${task}\u0000${identity}\u0000${hardware}\u0000${inputSignature(input)}`;
  const existing = seeds.get(key);
  if (existing) {
    existing.records.push(record);
    return existing;
  }
  const seed: VariantSeed = {
    key,
    familyId: modelFamilyId(model),
    identity,
    name: record.display_name,
    task,
    hardware,
    input,
    records: [record],
    assets: []
  };
  seeds.set(key, seed);
  return seed;
}

function attachExactAssets(
  model: ModelRecord,
  seeds: Map<string, VariantSeed>,
  assigned: Set<string>,
  defaultHardware: HardwareId | undefined
): void {
  const assetsByFilename = new Map(model.assets.map((asset) => [asset.filename, asset]));
  for (const seed of seeds.values()) {
    for (const record of seed.records) {
      if (!record.asset_filename) continue;
      const asset = assetsByFilename.get(record.asset_filename);
      if (!asset || !isRunnableAsset(asset)) continue;
      const hardware = assetHardware(asset) ?? defaultHardware;
      if (hardware !== undefined && hardware !== seed.hardware) continue;
      if (!seed.assets.some((candidate) => candidate.filename === asset.filename)) seed.assets.push(asset);
      assigned.add(assetKey(asset));
    }
  }
}

function attachInferredAssets(
  model: ModelRecord,
  seeds: Map<string, VariantSeed>,
  assigned: Set<string>,
  defaultHardware: HardwareId | undefined
): void {
  const familyId = modelFamilyId(model);
  const declaredFamily = familyFromText(model.id)?.id;
  const candidatesByAsset = new Map<string, VariantSeed[]>();
  for (const asset of model.assets.filter(isRunnableAsset)) {
    if (assigned.has(assetKey(asset))) continue;
    const hardware = assetHardware(asset) ?? defaultHardware;
    if (hardware === undefined) continue;
    const assetFamily = familyFromText(asset.filename)?.id;
    if (assetFamily !== undefined && declaredFamily !== undefined && assetFamily !== familyId) continue;
    const candidates = [...seeds.values()].filter((seed) => {
      if (seed.hardware !== hardware) return false;
      if (assetFamily !== undefined && assetFamily !== seed.familyId && declaredFamily !== undefined) return false;
      return tuplesCompatible(assetTuple(asset, seed), benchmarkTuple(seed));
    });
    candidatesByAsset.set(assetKey(asset), candidates);
  }

  const candidatesBySeed = new Map<VariantSeed, Array<{ asset: AssetRecord; tuple: string }>>();
  for (const asset of model.assets.filter(isRunnableAsset)) {
    if (assigned.has(assetKey(asset))) continue;
    const candidates = candidatesByAsset.get(assetKey(asset)) ?? [];
    // Do not pick the first same-prefix file. An asset is linked only when the
    // parsed family/task/spec/shape/hardware tuple identifies one seed.
    if (candidates.length !== 1) continue;
    const seed = candidates[0]!;
    const values = candidatesBySeed.get(seed) ?? [];
    values.push({ asset, tuple: assetTuple(asset, seed) });
    candidatesBySeed.set(seed, values);
  }
  for (const [seed, candidates] of candidatesBySeed) {
    // If the benchmark omits shape/format/layout, two otherwise matching
    // assets with contradictory metadata are ambiguous. Keep both as
    // explicit asset-only rows rather than attaching both to one benchmark.
    if (candidates.some((candidate, index) =>
      candidates.slice(index + 1).some((other) => tuplesConflict(candidate.tuple, other.tuple))
    )) continue;
    for (const { asset } of candidates) {
      seed.assets.push(asset);
      assigned.add(assetKey(asset));
    }
  }

  // S SAM benchmark rows explicitly describe an encoder-decoder package while
  // the manifest lists both component files and references its encoder. Attach
  // the companion only for that explicit single-package configuration.
  const packageSeeds = [...seeds.values()].filter((seed) =>
    seed.identity.toLowerCase().includes("encoder-decoder")
  );
  for (const seed of packageSeeds) {
    const peers = packageSeeds.filter((candidate) =>
      candidate.familyId === seed.familyId && candidate.task === seed.task && candidate.hardware === seed.hardware
    );
    if (peers.length !== 1) continue;
    for (const asset of model.assets.filter(isRunnableAsset)) {
      if (assigned.has(assetKey(asset)) || assetHardware(asset) !== seed.hardware) continue;
      const assetFamily = familyFromText(asset.filename)?.id;
      if (assetFamily !== undefined && declaredFamily !== undefined && assetFamily !== seed.familyId) continue;
      if (/(?:encoder|decoder)/i.test(asset.filename)) {
        seed.assets.push(asset);
        assigned.add(assetKey(asset));
      }
    }
  }
}

function appendUnmeasuredAssets(
  model: ModelRecord,
  seeds: Map<string, VariantSeed>,
  assigned: Set<string>,
  defaultHardware: HardwareId | undefined
): void {
  const familyId = modelFamilyId(model);
  const declaredFamily = familyFromText(model.id)?.id;
  for (const asset of model.assets.filter(isRunnableAsset)) {
    if (assigned.has(assetKey(asset))) continue;
    const hardware = assetHardware(asset) ?? defaultHardware;
    const assetFamily = familyFromText(asset.filename)?.id;
    if (
      hardware === undefined
      || (assetFamily !== undefined && declaredFamily !== undefined && assetFamily !== familyId)
    ) continue;
    const task = taskFor(model, asset.filename);
    const input = inputFor(undefined, asset.filename);
    const tuple = canonicalTuple(asset.filename, familyId, task, input);
    const identity = assetVariantIdentity(asset, familyId, task, input);
    const key = `${task}\u0000asset:${tuple}\u0000${hardware}`;
    const existing = seeds.get(key);
    if (existing) {
      existing.assets.push(asset);
    } else {
      seeds.set(key, {
        key,
        familyId,
        identity,
        name: assetDisplayName(asset, model, task),
        task,
        hardware,
        input,
        records: [],
        assets: [asset]
      });
    }
    assigned.add(assetKey(asset));
  }
}

function makeVariant(seed: VariantSeed, model: ModelRecord, releaseTag: string): ModelVariant {
  const id = `${seed.identity || "variant"}-${seed.task}-${seed.hardware}`.replace(/[^a-zA-Z0-9._:-]+/g, "-");
  return {
    id,
    name: seed.name,
    hardware: seed.hardware,
    task: seed.task,
    input: seed.input,
    assets: uniqueAssets(seed.assets),
    benchmarks: seed.records,
    sample_path: model.sample_path,
    release_tag: releaseTag
  };
}

function defaultHardwareForModel(model: ModelRecord & Partial<PlatformModelRecord>): HardwareId | undefined {
  const platform = model.platform;
  if (platform === "x3") return "x3";
  if (platform === "x5") return "x5";
  return undefined;
}

/** Builds the hardware-local, runnable variants for a family/platform record. */
export function buildModelVariants(
  model: ModelRecord & Partial<PlatformModelRecord>,
  options: BuildVariantOptions = {}
): ModelVariant[] {
  const defaultHardware = options.defaultHardware ?? defaultHardwareForModel(model);
  const releaseTag = options.releaseTag ?? model.release_tag ?? "";
  const seeds = new Map<string, VariantSeed>();
  for (const record of model.benchmarks) {
    const hardware = normalizeHardware(record.environment.hardware);
    if (hardware === undefined) continue;
    seedFor(seeds, model, record, hardware);
  }

  const assigned = new Set<string>();
  attachExactAssets(model, seeds, assigned, defaultHardware);
  attachInferredAssets(model, seeds, assigned, defaultHardware);
  appendUnmeasuredAssets(model, seeds, assigned, defaultHardware);

  return [...seeds.values()]
    .sort((left, right) => HARDWARE_IDS.indexOf(left.hardware) - HARDWARE_IDS.indexOf(right.hardware) || left.task.localeCompare(right.task) || left.identity.localeCompare(right.identity))
    .map((seed) => makeVariant(seed, model, releaseTag));
}

/**
 * Runtime compatibility helper. New generated catalogs carry `variants`; old
 * fixtures and hand-authored catalogs are derived from their benchmark/assets
 * records without promoting a release-line `s` label to all S hardware.
 */
export function getModelVariants(model: ModelRecord): ModelVariant[] {
  if (model.variants && model.variants.length > 0) return model.variants;
  if (model.platforms && model.platforms.length > 0) {
    return model.platforms.flatMap((platform) => getModelVariants(platform));
  }
  return buildModelVariants(model);
}

export function getHardwareIds(model: ModelRecord): HardwareId[] {
  const ids = new Set(getModelVariants(model).map((variant) => variant.hardware));
  return HARDWARE_IDS.filter((hardware) => ids.has(hardware));
}
