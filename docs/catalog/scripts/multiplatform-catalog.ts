import { readFile, access } from "node:fs/promises";
import { resolve } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { parse } from "yaml";
import { validateReleaseSummary } from "./release-summary";
import { correctSArtifacts } from "./source-corrections";
import { buildModelVariants, getModelVariants } from "../src/catalog/variants";
import { canonicalFamilyId, officialFamilyName, officialYoloName } from "../src/catalog/model-naming";
import type { BenchmarkRecord, Catalog, CatalogPlatform, ModelRecord, ModelVariant, PlatformModelRecord } from "../src/catalog/types";

const execFileAsync = promisify(execFile);

interface ManifestModel extends Omit<ModelRecord, "benchmarks" | "platforms"> {}
interface SourceDocument { release: { tag: string; platform: CatalogPlatform; version: string }; summary?: Record<string, number>; models: ManifestModel[]; }
interface BenchmarkDocument { release: { tag: string; platform: string; version: string }; benchmarks: BenchmarkRecord[]; }

interface FamilyEntry {
  model: ManifestModel;
  benchmarks: BenchmarkRecord[];
  familyIds: string[];
}

async function gitShow(repositoryRoot: string, ref: string, path: string): Promise<string> {
  const result = await execFileAsync("git", ["-C", repositoryRoot, "show", `${ref}:${path}`], { encoding: "utf8" });
  return result.stdout;
}

async function manifestPair(repositoryRoot: string, ref: string, tag: string): Promise<{ models: SourceDocument; benchmarks: BenchmarkDocument }> {
  // Existing release tags keep their original paths. Read both layouts
  // without changing immutable history or keeping duplicate live manifests.
  const directory = ref === "WORKTREE"
    ? await access(resolve(repositoryRoot, "docs/release/models.yaml")).then(() => "docs/release", () => "release")
    : await execFileAsync("git", ["-C", repositoryRoot, "cat-file", "-e", `${ref}:docs/release/models.yaml`])
      .then(() => "docs/release", () => "release");
  const read = (path: string): Promise<string> => ref === "WORKTREE"
    ? readFile(resolve(repositoryRoot, path), "utf8") : gitShow(repositoryRoot, ref, path);
  const [modelsText, benchmarksText] = await Promise.all([
    read(`${directory}/models.yaml`), read(`${directory}/benchmarks.yaml`)
  ]);
  const models = parse(modelsText) as SourceDocument;
  const benchmarks = parse(benchmarksText) as BenchmarkDocument;
  validateReleaseSummary(models.models, benchmarks.benchmarks, models.summary);
  if (models.release.tag !== tag || benchmarks.release.tag !== tag
    || models.release.version !== benchmarks.release.version || models.release.platform !== benchmarks.release.platform) {
    throw new Error(`Release identity mismatch for ${tag}`);
  }
  if (JSON.stringify({ models, benchmarks }).toLowerCase().includes("yoloe")) throw new Error("YOLOE is excluded from the catalog");
  return { models, benchmarks };
}

function toPlatformModel(platform: CatalogPlatform, tag: string, model: ManifestModel, benchmarks: BenchmarkRecord[]): PlatformModelRecord {
  const platformModel: PlatformModelRecord = { ...model, platform, release_tag: tag, benchmarks };
  const variants = buildModelVariants(platformModel, { releaseTag: tag });
  platformModel.variants = variants;
  // The source Ultralytics record is a shared directory index with all task
  // names. A family/platform payload must advertise only tasks represented by
  // its resolved variants, otherwise YOLOv10, YOLOv11, etc. inherit unrelated
  // classification/segmentation labels.
  if (variants.length > 0) {
    platformModel.tasks = [...new Set(variants.map((variant) => variant.task))];
  }
  return platformModel;
}

function cardFromPlatforms(id: string, name: string, platforms: PlatformModelRecord[]): ModelRecord {
  const representative = platforms.find((platform) => platform.platform === "x5") ?? platforms[0]!;
  const assets = platforms.flatMap((platform) => platform.assets).filter((asset, index, all) => {
    const key = asset.url ?? asset.filename;
    return all.findIndex((candidate) => (candidate.url ?? candidate.filename) === key) === index;
  });
  const benchmarks = platforms.flatMap((platform) => platform.benchmarks);
  const variants = platforms.flatMap((platform) => getModelVariants(platform));
  return {
    ...representative,
    id,
    name,
    tasks: [...new Set(platforms.flatMap((platform) => platform.tasks))],
    download_scripts: [...new Set(platforms.flatMap((platform) => platform.download_scripts))],
    assets,
    benchmarks,
    variants,
    platforms
  };
}

function familyIdentity(model: ManifestModel, benchmark?: BenchmarkRecord): { id: string; name: string } {
  const text = `${model.id} ${benchmark?.variant_id ?? ""}`.toLowerCase();
  const yolo = /yolov?(\d+)/.exec(text);
  if (yolo) return { id: `yolov${yolo[1]}`, name: officialYoloName(yolo[1]!) };
  // A MobileNet backbone in a segmentation sample (for example
  // `unet_mobilenet`) is not the MobileNet classifier family.
  if (
    !/(^|[^a-z0-9])unet[_-]?mobilenet/.test(text)
    && /(^|[^a-z0-9])mobilenet(?:v?\d+)?(?=$|[^a-z0-9])/.test(text)
  ) {
    return { id: "mobilenet", name: "MobileNet" };
  }
  return {
    id: canonicalFamilyId(model.id),
    name: officialFamilyName(canonicalFamilyId(model.id), model.name)
  };
}

function assetFamilyId(asset: ManifestModel["assets"][number]): string | undefined {
  const text = `${asset.filename} ${asset.url ?? ""}`.toLowerCase();
  const yolo = /yolov?(\d+)/.exec(text);
  if (yolo) return `yolov${yolo[1]}`;
  if (
    !/(^|[^a-z0-9])unet[_-]?mobilenet/.test(text)
    && /(^|[^a-z0-9])mobilenet(?:v?\d+)?(?=$|[^a-z0-9])/.test(text)
  ) return "mobilenet";
  return undefined;
}

function familyIdentityFromAsset(asset: ManifestModel["assets"][number]): { id: string; name: string } | undefined {
  const text = `${asset.filename} ${asset.url ?? ""}`.toLowerCase();
  const yolo = /yolov?(\d+)/.exec(text);
  if (yolo) return { id: `yolov${yolo[1]}`, name: officialYoloName(yolo[1]!) };
  if (
    !/(^|[^a-z0-9])unet[_-]?mobilenet/.test(text)
    && /(^|[^a-z0-9])mobilenet(?:v?\d+)?(?=$|[^a-z0-9])/.test(text)
  ) {
    return { id: "mobilenet", name: "MobileNet" };
  }
  return undefined;
}

function assetsForFamily(entry: FamilyEntry, familyId: string): ManifestModel["assets"] {
  const referenced = new Set(
    entry.benchmarks
      .map((benchmark) => benchmark.asset_filename)
      .filter((filename): filename is string => filename !== undefined)
  );
  return entry.model.assets.filter((asset) => {
    if (referenced.has(asset.filename)) return true;
    const inferredFamily = assetFamilyId(asset);
    // A source model containing one family may use unqualified filenames;
    // when it contains several families, only the family token is accepted.
    return inferredFamily === familyId || (entry.familyIds.length === 1 && inferredFamily === undefined);
  });
}

function uniqueAssets(assets: ManifestModel["assets"]): ManifestModel["assets"] {
  return assets.filter((asset, index, all) => {
    const key = asset.url ?? asset.filename;
    return all.findIndex((candidate) => (candidate.url ?? candidate.filename) === key) === index;
  });
}

function variantKey(variant: ModelVariant): string {
  return JSON.stringify([
    variant.id,
    variant.hardware,
    variant.task,
    variant.input?.shape ?? null,
    variant.input?.format?.toLowerCase() ?? null,
    variant.input?.layout?.toLowerCase() ?? null
  ]);
}

function mergeVariants(platformModels: PlatformModelRecord[]): ModelVariant[] {
  const merged = new Map<string, ModelVariant>();
  for (const platformModel of platformModels) {
    for (const variant of getModelVariants(platformModel)) {
      const key = variantKey(variant);
      const existing = merged.get(key);
      if (!existing) {
        merged.set(key, {
          ...variant,
          assets: [...variant.assets],
          benchmarks: [...variant.benchmarks]
        });
        continue;
      }
      existing.assets = uniqueAssets([...existing.assets, ...variant.assets]);
      const seenBenchmarks = new Set(existing.benchmarks.map((record) => record.id));
      existing.benchmarks.push(...variant.benchmarks.filter((record) => {
        if (seenBenchmarks.has(record.id)) return false;
        seenBenchmarks.add(record.id);
        return true;
      }));
    }
  }
  return [...merged.values()];
}

function mergePlatformModels(platform: CatalogPlatform, tag: string, familyId: string, entries: FamilyEntry[]): PlatformModelRecord {
  // Build each source entry independently first. A family can span several
  // sample directories (for example X3 YOLOv8 detect and YOLOv8-Seg), and a
  // single representative sample_path would otherwise leak into every row.
  const platformModels = entries.map((entry) => {
    const assets = assetsForFamily(entry, familyId);
    const model: ManifestModel = {
      ...entry.model,
      id: familyId,
      assets,
      tasks: [...new Set(entry.model.tasks)],
      download_scripts: [...new Set(entry.model.download_scripts)]
    };
    return toPlatformModel(platform, tag, model, entry.benchmarks);
  });
  const first = platformModels[0]!;
  const assets = uniqueAssets(platformModels.flatMap((platformModel) => platformModel.assets));
  const benchmarks = platformModels.flatMap((platformModel) => platformModel.benchmarks);
  const variants = mergeVariants(platformModels);
  return {
    ...first,
    assets,
    benchmarks,
    tasks: [...new Set(platformModels.flatMap((platformModel) => platformModel.tasks))],
    download_scripts: [...new Set(platformModels.flatMap((platformModel) => platformModel.download_scripts))],
    variants
  };
}

export async function buildMultiplatformCatalog(repositoryRoot: string): Promise<Catalog> {
  // X5 comes from this checkout (the exact release tag in CI), never a local
  // or remote branch that may have moved. Other hardware lines are locked.
  const current = parse(await readFile(resolve(repositoryRoot, "docs/release/models.yaml"), "utf8")) as SourceDocument;
  const version = (await readFile(resolve(repositoryRoot, "VERSION"), "utf8")).trim();
  if (current.release.platform !== "x5" || current.release.version !== version || current.release.tag !== `x5-v${version}`) {
    throw new Error("X5 VERSION and manifest release identity disagree");
  }
  const pins = JSON.parse(await readFile(resolve(repositoryRoot, "docs/release/catalog-sources.json"), "utf8")) as Record<"s" | "x3", string>;
  for (const platform of ["s", "x3"] as const) {
    if (!new RegExp(`^${platform}-v[0-9]+\\.[0-9]+\\.[0-9]+$`).test(pins[platform])) {
      throw new Error(`Catalog source must be an immutable platform tag: ${platform}`);
    }
    const type = await execFileAsync("git", ["-C", repositoryRoot, "cat-file", "-t", pins[platform]]);
    if (type.stdout.trim() !== "tag") throw new Error(`Annotated source tag required: ${pins[platform]}`);
  }
  const tags: Array<[CatalogPlatform, string, string]> = [
    ["x5", "WORKTREE", current.release.tag], ["s", pins.s, pins.s], ["x3", pins.x3, pins.x3]
  ];
  const loaded = new Map<CatalogPlatform, Awaited<ReturnType<typeof manifestPair>>>();
  for (const [platform, ref, tag] of tags) loaded.set(platform, await manifestPair(repositoryRoot, ref, tag));
  const sSource = loaded.get("s");
  if (sSource) correctSArtifacts(sSource.models, { benchmarks: sSource.benchmarks.benchmarks });
  const families = new Map<string, { name: string; platforms: Map<CatalogPlatform, FamilyEntry[]> }>();
  for (const [platform] of tags) {
    const source = loaded.get(platform)!;
    for (const model of source.models.models) {
      const byFamily = new Map<string, BenchmarkRecord[]>();
      for (const benchmark of source.benchmarks.benchmarks.filter((record) => record.sample_id === model.id)) {
        const family = familyIdentity(model, benchmark);
        const records = byFamily.get(family.id) ?? [];
        records.push(benchmark);
        byFamily.set(family.id, records);
      }
      // A release can publish a runnable artifact before its board benchmark
      // is available. Seed those families from the asset names so the online
      // catalog still exposes a hardware row marked “尚未实测”.
      for (const asset of model.assets) {
        const family = familyIdentityFromAsset(asset);
        if (family && !byFamily.has(family.id)) byFamily.set(family.id, []);
      }
      if (byFamily.size === 0) byFamily.set(familyIdentity(model).id, []);
      const familyIds = [...byFamily.keys()];
      for (const [familyId, benchmarks] of byFamily) {
        const identity = familyIdentity(model, benchmarks[0]);
        const name = officialFamilyName(familyId, familyId === "mobilenet" ? "MobileNet" : identity.name);
        const family = families.get(familyId) ?? { name, platforms: new Map() };
        const platformModels = family.platforms.get(platform) ?? [];
        platformModels.push({ model, benchmarks, familyIds });
        family.platforms.set(platform, platformModels);
        families.set(familyId, family);
      }
    }
  }
  const cards: ModelRecord[] = [...families.entries()].map(([id, family]) => cardFromPlatforms(
    id,
    family.name,
    tags.filter(([platform]) => family.platforms.has(platform)).map(([platform]) =>
       mergePlatformModels(platform, loaded.get(platform)!.models.release.tag, id, family.platforms.get(platform)!)
    )
  ));
  const runnableAssets = uniqueAssets(cards.flatMap((card) => getModelVariants(card).flatMap((variant) => variant.assets)));
  const assetCount = runnableAssets.length;
  const downloadableAssetCount = runnableAssets.filter((asset) => asset.url).length;
  return {
    schema_version: 1,
    release: { platform: "multi", version: current.release.version, tag: tags.map(([, , tag]) => tag).join("+"), platform_tags: Object.fromEntries(tags.map(([platform, , tag]) => [platform, tag])) },
    summary: {
      sample_count: cards.length,
      asset_count: assetCount,
      downloadable_asset_count: downloadableAssetCount,
      benchmark_count: cards.reduce((count, card) => count + card.benchmarks.length, 0)
    },
    models: cards.sort((left, right) => left.name.localeCompare(right.name))
  };
}
