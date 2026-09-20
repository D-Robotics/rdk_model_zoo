import { createHash } from "node:crypto";
import { parse } from "yaml";
import { PLATFORMS, readSourceFile, type PlatformSource } from "../sources";
import { validateNormalizedCatalog, validatePublishedManifests, type PlatformDocuments } from "./manifest-validation";
import { validateReleaseSummary } from "./release-summary";
import { correctSArtifacts } from "./source-corrections";
import { applySCatalogErrata } from "./catalog-errata";
import { applyX3CatalogErrata } from "./x3-catalog-errata";
import { buildModelVariants, canonicalTuple, getModelVariants, tuplesCompatible } from "../catalog/variants";
import { canonicalFamilyId, officialFamilyName, officialYoloName } from "../catalog/model-naming";
import type {
  BenchmarkRecord, Catalog, CatalogPlatform, CatalogSourceRecord, ModelRecord, ModelVariant, PlatformModelRecord
} from "../catalog/types";

interface ManifestModel extends Omit<ModelRecord, "benchmarks" | "platforms"> {}
interface ReleaseIdentity { tag: string; platform: CatalogPlatform; version: string | number; }
interface SourceDocument { release: ReleaseIdentity; summary?: Record<string, number>; models: ManifestModel[]; }
interface BenchmarkDocument { release: ReleaseIdentity; benchmarks: BenchmarkRecord[]; }

interface FamilyEntry {
  model: ManifestModel;
  benchmarks: BenchmarkRecord[];
  familyIds: string[];
}

function sha256(text: string): string {
  // Git may check text out as CRLF on Windows. Hash canonical LF content so
  // worktrees and git-show sources identify the same published revision.
  return createHash("sha256").update(text.replace(/\r\n/g, "\n"), "utf8").digest("hex");
}

/**
 * Reads one platform's manifest pair. Every platform is read through the same
 * path, and the manifest's own release section is the identity that must agree
 * with the platform VERSION file — no platform is taken from a website release.
 */
async function manifestPair(
  repositoryRoot: string,
  source: PlatformSource,
  repositoryUrl?: string
): Promise<{ models: SourceDocument; benchmarks: BenchmarkDocument; digest: string }> {
  const modelsText = await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/models.yaml`);
  const benchmarksText = await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/benchmarks.yaml`);
  const models = parse(modelsText) as SourceDocument;
  const benchmarks = parse(benchmarksText) as BenchmarkDocument;
  const version = String(models.release.version);
  const expectedTag = `${source.platform}-v${version}`;
  if (models.release.platform !== source.platform || benchmarks.release.platform !== source.platform) {
    throw new Error(`${source.platform}: manifest release.platform does not name its own platform`);
  }
  if (models.release.tag !== expectedTag || benchmarks.release.tag !== expectedTag) {
    throw new Error(`${source.platform}: manifest release tag must be ${expectedTag}`);
  }
  if (String(benchmarks.release.version) !== version) {
    throw new Error(`Release identity mismatch for ${expectedTag}`);
  }
  const platformVersion = (await readSourceFile(repositoryRoot, source, source.versionFile)).trim();
  if (platformVersion !== version) {
    throw new Error(`${source.platform}: VERSION ${platformVersion} and manifest release version ${version} disagree`);
  }
  validateReleaseSummary(models.models, benchmarks.benchmarks, models.summary);
  const documents = { models, benchmarks } as unknown as PlatformDocuments;
  await validatePublishedManifests({ repositoryRoot, source, documents, repositoryUrl });
  return { models, benchmarks, digest: sha256(`${modelsText}
${benchmarksText}`) };
}

/**
 * Applies the documented normalisation and errata layers for one platform.
 *
 * The layers are keyed to the release each platform published, so they apply
 * identically whether the manifest was read from the worktree or from the
 * frozen tag that first carried it. Both the published artifact and the release
 * check run this, so the documents that get validated are exactly the ones that
 * get published.
 */
export function normalizePlatformDocuments(source: PlatformSource, documents: PlatformDocuments): void {
  const tag = String(documents.models.release.tag ?? "");
  if (source.platform === "s") {
    correctSArtifacts(documents.models, { benchmarks: documents.benchmarks.benchmarks });
    applySCatalogErrata(tag, documents.models.models, documents.benchmarks.benchmarks);
  }
  if (source.platform === "x3") {
    applyX3CatalogErrata(tag, documents.benchmarks.benchmarks);
  }
}

/**
 * A platform's manifest is authoritative for its own hardware only. The X5
 * manifest also carries two RDK X3 paddleocr records (historical
 * cross-publishing) that the X3 manifest publishes as well; letting both
 * through duplicated the X3 rows on the merged card.
 */
const PLATFORM_HARDWARE: Record<CatalogPlatform, RegExp> = {
  x5: /\bX5\b/,
  s: /\b(S100P|S100|S600)\b/,
  x3: /\bX3\b/
};

function platformOwnsRecord(platform: CatalogPlatform, record: BenchmarkRecord): boolean {
  const hardware = record.environment?.hardware ?? "";
  if (!hardware.trim()) return true;
  // Reject explicit cross-published board records, but keep toolchain and
  // unspecified-hardware evidence on its source platform. Variant resolution
  // must not turn that evidence into a board performance claim.
  return PLATFORM_HARDWARE[platform].test(hardware)
    || !Object.values(PLATFORM_HARDWARE).some(pattern => pattern.test(hardware));
}

function toPlatformModel(platform: CatalogPlatform, source: PlatformSource, tag: string, model: ManifestModel, benchmarks: BenchmarkRecord[]): PlatformModelRecord {
  const platformModel: PlatformModelRecord = { ...model, platform, release_tag: tag, benchmarks };
  const variants = buildModelVariants(platformModel, {
    releaseTag: tag,
    // The layout that actually holds `sample_path` for this platform, so a
    // reader on the migration branch gets a resolving link while a pinned build
    // keeps the frozen tag's original, prefix-free paths.
    sourceRef: source.linkRef,
    sourcePathPrefix: source.linkPrefix
  });
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
  // ResNet18/50/152 across release lines are one classifier family, exactly
  // like MobileNet v1-v4; `3dresnet` is a different model.
  if (
    !/(^|[^a-z0-9])(?:3d[-_]?resnet|unet[-_]?resnet)/.test(text)
    && /(^|[^a-z0-9])resnet(?:\d{2,3})?(?=$|[^a-z0-9])/.test(text)
  ) {
    return { id: "resnet", name: "ResNet" };
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
  if (
    !/(^|[^a-z0-9])(?:3d[-_]?resnet|unet[-_]?resnet)/.test(text)
    && /(^|[^a-z0-9])resnet(?:\d{2,3})?(?=$|[^a-z0-9])/.test(text)
  ) return "resnet";
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
  if (
    !/(^|[^a-z0-9])(?:3d[-_]?resnet|unet[-_]?resnet)/.test(text)
    && /(^|[^a-z0-9])resnet(?:\d{2,3})?(?=$|[^a-z0-9])/.test(text)
  ) {
    return { id: "resnet", name: "ResNet" };
  }
  return undefined;
}

function assetsForFamily(entry: FamilyEntry, familyId: string): ManifestModel["assets"] {
  const referenced = new Set(
    entry.benchmarks
      .map((benchmark) => benchmark.asset_filename)
      .filter((filename): filename is string => filename !== undefined)
  );
  // Assets whose family token matches must go to the entry that owns this
  // family's benchmarks. When an entry carries several families (the
  // Ultralytics directory indexes several YOLO versions), a foreign family's
  // assets would otherwise seed asset-only rows here while the entry holding
  // the benchmarks cannot attach them.
  const ownsFamily = entry.familyIds.includes(familyId);
  if (ownsFamily && entry.familyIds.length === 1) return entry.model.assets;
  return entry.model.assets.filter((asset) => {
    if (referenced.has(asset.filename)) return true;
    const inferredFamily = assetFamilyId(asset);
    // A source model containing one family may use unqualified filenames;
    // when it contains several families, only the family token is accepted.
    return (ownsFamily && inferredFamily === familyId)
      || (entry.familyIds.length === 1 && inferredFamily === undefined);
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

function targetVariantId(row: ModelVariant): string {
  const byVariant = new Map<string, number>();
  for (const record of row.benchmarks) byVariant.set(record.variant_id, (byVariant.get(record.variant_id) ?? 0) + 1);
  return [...byVariant.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0]
    ?? row.name;
}

function mergeVariants(platformModels: PlatformModelRecord[], familyId: string): ModelVariant[] {
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

  // A family can be spread over several sample directories (the Ultralytics
  // index publishes YOLO26 files while a second sample carries the YOLO26
  // benchmark rows). The benchmark rows and the asset-only rows then name the
  // same model differently (`yolo26l-detect-640` versus
  // `yolov26-l-640x640-nv12`), so the id-based key above keeps two rows: one
  // with measurements and no download, one with a download and no
  // measurements. Move the assets onto the measured row via the same tuple
  // identity the single-sample builder uses, and drop the emptied row.
  const rows = [...merged.values()];
  const assetOnly = rows.filter((row) => row.benchmarks.length === 0 && row.assets.length > 0);
  if (assetOnly.length > 0) {
    const taken = new Set<string>();
    for (const source of assetOnly) {
      // Match with the same identity the single-sample builder uses: the
      // asset filename alone, not the row id (ids embed task words that
      // canonicalTuple strips only once and pollute the identity).
      const sourceTuple = canonicalTuple(
        source.assets.map((asset) => asset.filename).join(" "),
        familyId,
        source.task,
        source.input
      );
      const target = rows.find((row) => row !== source
        && row.benchmarks.length > 0
        && row.hardware === source.hardware
        && row.task === source.task
        && tuplesCompatible(
          sourceTuple,
          canonicalTuple(
            // A single variant id, as the single-sample builder uses: joining
            // several ids leaves the second family token unstripped and
            // pollutes the identity.
            targetVariantId(row),
            familyId,
            row.task,
            row.input
          )
        ));
      if (!target) continue;
      const newAssets = uniqueAssets([...target.assets, ...source.assets]);
      const mergedAll = newAssets.length === target.assets.length + source.assets.length;
      const redundant = newAssets.length === target.assets.length;
      if (mergedAll) {
        target.assets = newAssets;
        taken.add(source.id);
      } else if (redundant && target.assets.length > 0) {
        // Every asset of the asset-only row is already attached to the
        // measured row (the n size carries an explicit asset_filename): the
        // empty duplicate row is pure noise and must be dropped.
        taken.add(source.id);
      }
    }
    return rows.filter((row) => !taken.has(row.id));
  }
  return rows;
}

function mergePlatformModels(platform: CatalogPlatform, source: PlatformSource, tag: string, familyId: string, entries: FamilyEntry[]): PlatformModelRecord {
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
    return toPlatformModel(platform, source, tag, model, entry.benchmarks);
  });
  const first = platformModels[0]!;
  const assets = uniqueAssets(platformModels.flatMap((platformModel) => platformModel.assets));
  const benchmarks = platformModels.flatMap((platformModel) => platformModel.benchmarks);
  const variants = mergeVariants(platformModels, familyId);
  return {
    ...first,
    assets,
    benchmarks,
    tasks: [...new Set(platformModels.flatMap((platformModel) => platformModel.tasks))],
    download_scripts: [...new Set(platformModels.flatMap((platformModel) => platformModel.download_scripts))],
    variants
  };
}

export interface BuildMultiplatformCatalogOptions {
  repositoryRoot: string;
  /** Resolved platform distributions; every platform is read the same way. */
  sources: PlatformSource[];
  /** Repository that hosts the platform trees, recorded for consumers. */
  repository?: string;
}

export async function buildMultiplatformCatalog(options: BuildMultiplatformCatalogOptions): Promise<Catalog> {
  const { repositoryRoot } = options;
  const ordered = PLATFORMS.map((platform) => {
    const source = options.sources.find((candidate) => candidate.platform === platform);
    if (!source) throw new Error(`Catalog build is missing the ${platform} platform source`);
    return source;
  });

  const loaded = new Map<CatalogPlatform, Awaited<ReturnType<typeof manifestPair>>>();
  for (const source of ordered) loaded.set(source.platform, await manifestPair(repositoryRoot, source, options.repository));
  const provenance: Record<string, CatalogSourceRecord> = {};
  for (const source of ordered) {
    const manifest = loaded.get(source.platform)!;
    provenance[source.platform] = {
      kind: source.kind,
      ref: source.linkRef,
      path: source.kind === "worktree" ? source.treePrefix : [source.treePrefix, source.manifestDirectory].filter(Boolean).join("/"),
      manifest_directory: source.manifestDirectory,
      release_tag: manifest.models.release.tag,
      manifest_sha256: manifest.digest
    };
  }

  for (const source of ordered) {
    const manifest = loaded.get(source.platform)!;
    normalizePlatformDocuments(source, manifest as unknown as PlatformDocuments);
  }

  // Asset references and cited evidence are checked on the normalized
  // documents, so a corrected or added record must still resolve to real files.
  for (const source of ordered) {
    const manifest = loaded.get(source.platform)!;
    await validateNormalizedCatalog({
      repositoryRoot,
      source,
      documents: manifest as unknown as PlatformDocuments,
      repositoryUrl: options.repository,
      onWarning: (message) => console.warn(`${source.platform}: ${message}`)
    });
  }

  const tags: Array<[CatalogPlatform, string, string]> = ordered.map((source) => [
    source.platform, source.linkRef, loaded.get(source.platform)!.models.release.tag
  ]);
  const families = new Map<string, { name: string; platforms: Map<CatalogPlatform, FamilyEntry[]> }>();
  for (const [platform] of tags) {
    const source = loaded.get(platform)!;
    for (const model of source.models.models) {
      const byFamily = new Map<string, BenchmarkRecord[]>();
      for (const benchmark of source.benchmarks.benchmarks.filter(
        (record) => record.sample_id === model.id && platformOwnsRecord(platform, record)
      )) {
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
        // A pipeline's detector/backbone is an asset of that pipeline. Only
        // the aggregate Ultralytics sample intentionally spans foreign families.
        const ownsAssetFamily = model.id === "ultralytics_yolo" || family?.id === familyIdentity(model).id;
        if (family && ownsAssetFamily && !byFamily.has(family.id)) byFamily.set(family.id, []);
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
    tags.filter(([platform]) => family.platforms.has(platform)).map(([platform]) => mergePlatformModels(
      platform,
      ordered.find((source) => source.platform === platform)!,
      loaded.get(platform)!.models.release.tag,
      id,
      family.platforms.get(platform)!
    ))
  ));
  const runnableAssets = uniqueAssets(cards.flatMap((card) => getModelVariants(card).flatMap((variant) => variant.assets)));
  const assetCount = runnableAssets.length;
  const downloadableAssetCount = runnableAssets.filter((asset) => asset.url).length;
  // Catalog format evolves independently of every platform release.
  const version = "1.0.0";
  const platformTags = Object.fromEntries(tags.map(([platform, , tag]) => [platform, tag]));
  const sortedCards = cards.sort((left, right) => left.name.localeCompare(right.name));
  const revision = sha256(JSON.stringify({ version, platformTags, provenance, models: sortedCards }));
  const catalogVersion = `catalog-v${version}-${revision.slice(0, 16)}`;
  return {
    schema_version: 1,
    release: {
      platform: "multi",
      version,
      tag: catalogVersion,
      // Changes on any platform, including errata, produce a new identity.
      catalog_version: catalogVersion,
      ...(options.repository ? { repository: options.repository } : {}),
      platform_tags: platformTags
    },
    sources: provenance,
    summary: {
      sample_count: cards.length,
      asset_count: assetCount,
      downloadable_asset_count: downloadableAssetCount,
      benchmark_count: cards.reduce((count, card) => count + card.benchmarks.length, 0)
    },
    models: sortedCards
  };
}
