/**
 * Release check for the model data this repository is authoritative for.
 *
 * Every platform distribution is validated the same way — schema, release
 * identity, unique ids, asset references, and the repository file each
 * benchmark cites — before any catalog artifact is published.
 *
 * Usage: tsx scripts/validate-sources.ts [--root <dir>]
 */
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readSourceFile, loadSourcesDocument, resolvePlatformSources } from "../src/sources";
import { validateNormalizedCatalog, validatePublishedManifests, type PlatformDocuments } from "../src/pipeline/manifest-validation";
import { normalizePlatformDocuments } from "../src/pipeline/multiplatform-catalog";
import { validateReleaseSummary } from "../src/pipeline/release-summary";
import { parse } from "yaml";

const publisherRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rootArgument = process.argv.indexOf("--root");
const repositoryRoot = rootArgument === -1 ? resolve(publisherRoot, "../..") : resolve(process.argv[rootArgument + 1]!);

const sourcesDocument = await loadSourcesDocument(resolve(publisherRoot, "sources.json"));
const sources = await resolvePlatformSources({ repositoryRoot, sources: sourcesDocument });

let failures = 0;
for (const source of sources) {
  const label = `${source.platform} (${source.kind} ${source.linkRef}/${source.manifestDirectory})`;
  try {
    const models = parse(await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/models.yaml`)) as PlatformDocuments["models"];
    const benchmarks = parse(await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/benchmarks.yaml`)) as PlatformDocuments["benchmarks"];
    const documents: PlatformDocuments = { models, benchmarks };
    await validatePublishedManifests({ repositoryRoot, source, documents });
    validateReleaseSummary(models.models, benchmarks.benchmarks, models.summary);
    // The release check validates the same normalized documents the artifact is
    // built from, so a correction cannot validate in one path and not the other.
    normalizePlatformDocuments(source, documents);
    await validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl: sourcesDocument.repository });
    console.log(
      `ok ${label}: ${models.models.length} samples, ${benchmarks.benchmarks.length} benchmark records, `
      + `${benchmarks.benchmarks.flatMap((record) => record.performance ?? []).length} performance metrics, `
      + `${benchmarks.benchmarks.flatMap((record) => record.accuracy ?? []).length} accuracy metrics`
    );
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${label}: ${(error as Error).message}`);
  }
}

if (failures > 0) {
  console.error(`${failures} platform source(s) failed validation.`);
  process.exit(1);
}
console.log("All platform sources validated.");
