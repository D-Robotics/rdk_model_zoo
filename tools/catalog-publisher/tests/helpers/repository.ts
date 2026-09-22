import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";
import type { Catalog, CatalogPlatform } from "../../src/catalog/types";
import { buildMultiplatformCatalog } from "../../src/pipeline/multiplatform-catalog";
import type { PlatformDocuments } from "../../src/pipeline/manifest-validation";
import { loadSourcesDocument, readSourceFile, resolvePlatformSources, type PlatformSource } from "../../src/sources";

export const publisherRoot = fileURLToPath(new URL("../../", import.meta.url));
export const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));

let cached: Promise<Catalog> | undefined;

/**
 * Builds the shipped catalog from the checked-out platform distributions once
 * per run. Every suite that inspects real release data shares this instance so
 * the manifests are read a single time.
 */
export function repositoryCatalog(): Promise<Catalog> {
  cached ??= (async () => {
    const sources = await loadSourcesDocument(resolve(publisherRoot, "sources.json"));
    const resolved = await resolvePlatformSources({ repositoryRoot, sources });
    return buildMultiplatformCatalog({ repositoryRoot, sources: resolved, repository: sources.repository });
  })();
  return cached;
}

/** One platform's slice of a family card, with the empty defaults spelled out. */
export function platformSlice(catalog: Catalog, platform: CatalogPlatform, familyId: string) {
  const family = catalog.models.find((model) => model.id === familyId);
  const entry = family?.platforms?.find((candidate) => candidate.platform === platform);
  return {
    family,
    entry,
    variants: entry?.variants ?? [],
    benchmarks: entry?.benchmarks ?? []
  };
}

/** Every benchmark record one platform published, across all family cards. */
export function platformBenchmarks(catalog: Catalog, platform: CatalogPlatform) {
  return catalog.models.flatMap((model) => model.platforms ?? [])
    .filter((entry) => entry.platform === platform)
    .flatMap((entry) => entry.benchmarks);
}

/** A platform source that reads one fixture release directory instead of the worktree. */
export function fixtureSource(variant: string, platform: CatalogPlatform = "x5"): PlatformSource {
  return {
    platform,
    kind: "worktree",
    worktreeRoot: "tools/catalog-publisher/tests/fixtures",
    treePrefix: "",
    manifestDirectory: `release/${variant}`,
    versionFile: "VERSION",
    linkRef: "main",
    linkPrefix: ""
  };
}

export interface FixtureRelease {
  source: PlatformSource;
  documents: PlatformDocuments;
}

/** Reads one fixture release directory as a platform manifest pair. */
export async function fixtureRelease(variant: string): Promise<FixtureRelease> {
  const source = fixtureSource(variant);
  const models = parse(await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/models.yaml`)) as PlatformDocuments["models"];
  const benchmarks = parse(await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/benchmarks.yaml`)) as PlatformDocuments["benchmarks"];
  return { source, documents: { models, benchmarks } };
}
