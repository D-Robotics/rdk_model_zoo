/**
 * Catalog data CLI.
 *
 * Reads the three platform distributions that live in this repository, applies
 * the documented normalisation layers, validates every manifest against the
 * schema shipped with its own platform, and writes a versioned `catalog.json`
 * plus the `catalog.meta.json` checksum contract that consumers lock against.
 *
 * Usage:
 *   tsx scripts/build-catalog.ts [--root <dir>] [--out <dir>]
 *                                [--pin <platform>=<tag>[:<tree-prefix>]]... [--check]
 */
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildMultiplatformCatalog } from "../src/pipeline/multiplatform-catalog";
import { CATALOG_FILE, CATALOG_METADATA_FILE, serializeCatalog, writeCatalogArtifact } from "../src/artifact";
import { loadSourcesDocument, resolvePlatformSources, type CatalogPlatform, type SourcePin } from "../src/sources";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const publisherRoot = resolve(scriptDirectory, "..");
const defaultRoot = resolve(publisherRoot, "../..");

interface Cli {
  root: string;
  out: string;
  pins: Partial<Record<CatalogPlatform, SourcePin>>;
  check: boolean;
}

function parseArguments(argv: string[]): Cli {
  const cli: Cli = { root: defaultRoot, out: resolve(publisherRoot, "dist"), pins: {}, check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]!;
    const value = (): string => {
      const next = argv[index + 1];
      if (next === undefined || next.startsWith("--")) throw new Error(`${argument} requires a value`);
      index += 1;
      return next;
    };
    if (argument === "--root") cli.root = resolve(value());
    else if (argument === "--out") cli.out = resolve(value());
    else if (argument === "--check") cli.check = true;
    else if (argument === "--pin") {
      const [platform, target] = value().split("=");
      if (!platform || !target) throw new Error("--pin expects <platform>=<tag>[:<tree-prefix>]");
      const [tag, treePrefix] = target.split(":");
      cli.pins[platform as CatalogPlatform] = { tag: tag!, treePrefix: treePrefix ?? "" };
    } else if (argument === "--help" || argument === "-h") {
      console.log("Usage: tsx scripts/build-catalog.ts [--root <dir>] [--out <dir>] [--pin p=tag[:prefix]]... [--check]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  return cli;
}

const cli = parseArguments(process.argv.slice(2));
const sourcesDocument = await loadSourcesDocument(resolve(publisherRoot, "sources.json"));
const sources = await resolvePlatformSources({
  repositoryRoot: cli.root,
  sources: sourcesDocument,
  pins: cli.pins
});

for (const source of sources) {
  console.log(
    `source ${source.platform}: ${source.kind} ${source.linkRef}`
    + ` -> ${source.kind === "worktree" ? source.worktreeRoot : source.ref}/${source.manifestDirectory}`
  );
}

const catalog = await buildMultiplatformCatalog({
  repositoryRoot: cli.root,
  sources,
  repository: sourcesDocument.repository
});

if (cli.check) {
  const existing = await readFile(resolve(cli.out, CATALOG_FILE), "utf8").catch(() => undefined);
  const existingMetadata = await readFile(resolve(cli.out, CATALOG_METADATA_FILE), "utf8").catch(() => undefined);
  if (existing === undefined || existingMetadata === undefined) {
    console.error(`No artifact to check in ${cli.out}; run the build first.`);
    process.exit(1);
  }
  const expected = serializeCatalog(catalog);
  if (existing !== expected) {
    console.error("Catalog is not reproducible: the checked artifact differs from a fresh build.");
    process.exit(1);
  }
  const { verifyArtifact } = await import("../src/artifact");
  verifyArtifact(existing, JSON.parse(existingMetadata) as never);
  console.log(`Catalog artifact is reproducible and its checksum contract verifies (${catalog.release.catalog_version}).`);
  process.exit(0);
}

const artifact = await writeCatalogArtifact(catalog, cli.out, "tools/catalog-publisher");
console.log(`Wrote ${artifact.catalogPath}`);
console.log(`Wrote ${artifact.metadataPath}`);
console.log(`catalog_version: ${artifact.metadata.catalog_version}`);
console.log(`sha256: ${artifact.metadata.sha256} (${artifact.metadata.bytes} bytes)`);
console.log(`families: ${catalog.summary.sample_count}, benchmarks: ${catalog.summary.benchmark_count}`);
