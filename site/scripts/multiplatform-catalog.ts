import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { parse } from "yaml";
import type { BenchmarkRecord, Catalog, CatalogPlatform, ModelRecord, PlatformModelRecord } from "../src/catalog/types";

const execFileAsync = promisify(execFile);

interface ManifestModel extends Omit<ModelRecord, "benchmarks" | "platforms"> {}
interface SourceDocument { release: { tag: string; platform: CatalogPlatform; version: string }; summary?: Record<string, number>; models: ManifestModel[]; }
interface BenchmarkDocument { release: { tag: string }; benchmarks: BenchmarkRecord[]; }

async function gitShow(repositoryRoot: string, ref: string, path: string): Promise<string> {
  const result = await execFileAsync("git", ["-C", repositoryRoot, "show", `${ref}:${path}`], { encoding: "utf8" });
  return result.stdout;
}

async function manifestPair(repositoryRoot: string, ref: string, tag: string): Promise<{ models: SourceDocument; benchmarks: BenchmarkDocument }> {
  const [modelsText, benchmarksText] = await Promise.all([
    gitShow(repositoryRoot, ref, "release/models.yaml"),
    gitShow(repositoryRoot, ref, "release/benchmarks.yaml")
  ]);
  const models = parse(modelsText) as SourceDocument;
  const benchmarks = parse(benchmarksText) as BenchmarkDocument;
  if (models.release.tag !== tag || benchmarks.release.tag !== tag || models.release.tag !== benchmarks.release.tag) {
    throw new Error(`Release identity mismatch for ${tag}`);
  }
  if (JSON.stringify({ models, benchmarks }).toLowerCase().includes("yoloe")) throw new Error("YOLOE is excluded from the catalog");
  return { models, benchmarks };
}

function toPlatformModel(platform: CatalogPlatform, tag: string, model: ManifestModel, benchmarks: BenchmarkRecord[]): PlatformModelRecord {
  return { ...model, platform, release_tag: tag, benchmarks };
}

function cardFromPlatforms(id: string, name: string, platforms: PlatformModelRecord[]): ModelRecord {
  const representative = platforms.find((platform) => platform.platform === "x5") ?? platforms[0]!;
  return { ...representative, id, name, platforms };
}

function familyIdentity(model: ManifestModel, benchmark?: BenchmarkRecord): { id: string; name: string } {
  const text = `${model.id} ${benchmark?.variant_id ?? ""}`.toLowerCase();
  const yolo = /yolov?(\d+)/.exec(text);
  if (yolo) return { id: `yolov${yolo[1]}`, name: `YOLOv${yolo[1]}` };
  if (text.includes("mobilenet")) return { id: "mobilenet", name: "MobileNet" };
  return { id: model.id, name: model.name };
}

function mergePlatformModels(platform: CatalogPlatform, tag: string, models: Array<{ model: ManifestModel; benchmarks: BenchmarkRecord[] }>): PlatformModelRecord {
  const first = models[0]!;
  const assets = models.flatMap((entry) => entry.model.assets).filter((asset, index, all) =>
    all.findIndex((candidate) => candidate.filename === asset.filename) === index
  );
  const benchmarks = models.flatMap((entry) => entry.benchmarks);
  return toPlatformModel(platform, tag, { ...first.model, assets, tasks: [...new Set(models.flatMap((entry) => entry.model.tasks))] }, benchmarks);
}

export async function buildMultiplatformCatalog(repositoryRoot: string): Promise<Catalog> {
  // x5-v1.0.0 predates the first committed X5 manifests.  Its immutable
  // release identity is retained while rdk_x5 supplies that release's data.
  const tags: Array<[CatalogPlatform, string, string]> = [["x5", "rdk_x5", "x5-v1.0.0"], ["s", "s-v1.0.0", "s-v1.0.0"], ["x3", "x3-v1.0.0", "x3-v1.0.0"]];
  const loaded = new Map<CatalogPlatform, Awaited<ReturnType<typeof manifestPair>>>();
  for (const [platform, ref, tag] of tags) loaded.set(platform, await manifestPair(repositoryRoot, ref, tag));
  const families = new Map<string, { name: string; platforms: Map<CatalogPlatform, Array<{ model: ManifestModel; benchmarks: BenchmarkRecord[] }>> }>();
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
      if (byFamily.size === 0) byFamily.set(familyIdentity(model).id, []);
      for (const [familyId, benchmarks] of byFamily) {
        const identity = familyIdentity(model, benchmarks[0]);
        const family = families.get(familyId) ?? { name: identity.name, platforms: new Map() };
        const platformModels = family.platforms.get(platform) ?? [];
        platformModels.push({ model, benchmarks });
        family.platforms.set(platform, platformModels);
        families.set(familyId, family);
      }
    }
  }
  const cards: ModelRecord[] = [...families.entries()].map(([id, family]) => cardFromPlatforms(
    id,
    family.name,
    tags.filter(([platform]) => family.platforms.has(platform)).map(([platform]) =>
      mergePlatformModels(platform, loaded.get(platform)!.models.release.tag, family.platforms.get(platform)!)
    )
  ));
  return {
    schema_version: 1,
    release: { platform: "multi", version: "1.0.0", tag: "x5-v1.0.0+s-v1.0.0+x3-v1.0.0" },
    summary: { sample_count: cards.length, asset_count: cards.reduce((count, card) => count + card.assets.length, 0), benchmark_count: cards.reduce((count, card) => count + card.benchmarks.length, 0) },
    models: cards.sort((left, right) => left.name.localeCompare(right.name))
  };
}
