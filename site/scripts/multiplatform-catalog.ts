import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parse } from "yaml";
import type { BenchmarkRecord, Catalog, CatalogPlatform, ModelRecord, PlatformModelRecord } from "../src/catalog/types";

const execFileAsync = promisify(execFile);

interface ManifestModel extends Omit<ModelRecord, "benchmarks" | "platforms"> {}
interface SourceDocument { release: { tag: string; platform: CatalogPlatform; version: string }; summary?: Record<string, number>; models: ManifestModel[]; }
interface BenchmarkDocument { release: { tag: string }; benchmarks: BenchmarkRecord[]; }
interface IdentitySource { platform: CatalogPlatform; sample_id: string; variant_id: string; }
interface Identity { id: string; name: string; sources: IdentitySource[]; }

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

function recordsFor(model: ManifestModel, benchmarks: BenchmarkDocument, variantId?: string): BenchmarkRecord[] {
  return benchmarks.benchmarks.filter((benchmark) => benchmark.sample_id === model.id && (variantId === undefined || benchmark.variant_id === variantId));
}

function toPlatformModel(platform: CatalogPlatform, tag: string, model: ManifestModel, benchmarks: BenchmarkRecord[]): PlatformModelRecord {
  return { ...model, platform, release_tag: tag, benchmarks };
}

function cardFromPlatforms(id: string, name: string, platforms: PlatformModelRecord[]): ModelRecord {
  const representative = platforms.find((platform) => platform.platform === "x5") ?? platforms[0]!;
  return { ...representative, id, name, platforms };
}

export async function buildMultiplatformCatalog(repositoryRoot: string): Promise<Catalog> {
  // x5-v1.0.0 predates the first committed X5 manifests.  Its immutable
  // release identity is retained while rdk_x5 supplies that release's data.
  const tags: Array<[CatalogPlatform, string, string]> = [["x5", "rdk_x5", "x5-v1.0.0"], ["s", "s-v1.0.0", "s-v1.0.0"], ["x3", "x3-v1.0.0", "x3-v1.0.0"]];
  const loaded = new Map<CatalogPlatform, Awaited<ReturnType<typeof manifestPair>>>();
  for (const [platform, ref, tag] of tags) loaded.set(platform, await manifestPair(repositoryRoot, ref, tag));
  const registryText = await readFile(resolve(repositoryRoot, "release/catalog-identities.yaml"), "utf8");
  const identities = (parse(registryText) as { identities: Identity[] }).identities;
  const used = new Set<string>();
  const cards: ModelRecord[] = [];
  for (const identity of identities) {
    const platforms = identity.sources.map((source) => {
      const loadedSource = loaded.get(source.platform)!;
      const model = loadedSource.models.models.find((candidate) => candidate.id === source.sample_id);
      if (!model) throw new Error(`Identity ${identity.id} references missing sample ${source.platform}/${source.sample_id}`);
      const benchmarks = recordsFor(model, loadedSource.benchmarks, source.variant_id);
      if (benchmarks.length === 0) throw new Error(`Identity ${identity.id} references missing variant ${source.platform}/${source.variant_id}`);
      used.add(`${source.platform}:${source.sample_id}:${source.variant_id}`);
      return toPlatformModel(source.platform, loadedSource.models.release.tag, model, benchmarks);
    });
    cards.push(cardFromPlatforms(identity.id, identity.name, platforms));
  }
  for (const [platform] of tags) {
    const source = loaded.get(platform)!;
    for (const model of source.models.models) {
      const variants = new Set(recordsFor(model, source.benchmarks).map((record) => record.variant_id));
      if (variants.size === 0) {
        cards.push(cardFromPlatforms(`${platform}-${model.id}`, model.name, [toPlatformModel(platform, source.models.release.tag, model, [])]));
        continue;
      }
      for (const variantId of variants) {
        if (used.has(`${platform}:${model.id}:${variantId}`)) continue;
        const benchmarks = recordsFor(model, source.benchmarks, variantId);
        cards.push(cardFromPlatforms(`${platform}-${model.id}-${variantId}`, benchmarks[0]?.display_name ?? model.name, [toPlatformModel(platform, source.models.release.tag, model, benchmarks)]));
      }
    }
  }
  return {
    schema_version: 1,
    release: { platform: "multi", version: "1.0.0", tag: "x5-v1.0.0+s-v1.0.0+x3-v1.0.0" },
    summary: { sample_count: cards.length, asset_count: cards.reduce((count, card) => count + card.assets.length, 0), benchmark_count: cards.reduce((count, card) => count + card.benchmarks.length, 0) },
    models: cards.sort((left, right) => left.name.localeCompare(right.name))
  };
}
