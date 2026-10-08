import { isAbsolute, win32 } from "node:path";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { readRepositoryBlob, readSourceFile, repositoryBlobExists, type PlatformSource } from "../sources";
import type { BenchmarkRecord, ModelRecord } from "../catalog/types";

export class CatalogValidationError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message);
    this.name = "CatalogValidationError";
  }
}

export interface PlatformDocuments {
  models: { schema_version: number; release: Record<string, unknown>; summary?: Record<string, number>; models: ManifestModelRecord[] };
  benchmarks: { release: Record<string, unknown>; benchmarks: BenchmarkRecord[] };
}

interface ManifestModelRecord extends Omit<ModelRecord, "benchmarks"> {}

function formatSchemaErrors(errors: { instancePath?: string; message?: string }[] | null | undefined): string {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`).join("; ");
}

function validateFiniteNumbers(value: unknown, code: string, path = "$"): void {
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new CatalogValidationError(code, `${code}: ${path} must be a finite number`);
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => validateFiniteNumbers(item, code, `${path}[${index}]`));
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      validateFiniteNumbers(item, code, `${path}.${key}`);
    }
  }
}

/** Validates a parsed manifest against the JSON Schema its own platform ships. */
async function validateWithSchema(
  repositoryRoot: string,
  source: PlatformSource,
  document: unknown,
  schemaFile: string
): Promise<void> {
  const code = schemaFile.includes("models") ? "MODELS_SCHEMA" : "BENCHMARKS_SCHEMA";
  let schemaText: string;
  try {
    schemaText = await readSourceFile(repositoryRoot, source, `${source.manifestDirectory}/schemas/${schemaFile}`);
  } catch {
    throw new CatalogValidationError(code, `${source.platform}: schema not found: ${source.manifestDirectory}/schemas/${schemaFile}`);
  }
  const ajv = new Ajv2020({ allErrors: true, strict: true });
  addFormats(ajv);
  const validate = ajv.compile(JSON.parse(schemaText) as object);
  if (!validate(document)) {
    throw new CatalogValidationError(code, `${source.platform} ${code}: ${formatSchemaErrors(validate.errors)}`);
  }
  validateFiniteNumbers(document, code);
}

function validateReleaseTags(modelsTag: unknown, benchmarksTag: unknown, platform: string): void {
  if (modelsTag !== benchmarksTag) {
    throw new CatalogValidationError("RELEASE_TAG_MISMATCH", `${platform}: release tags differ: ${modelsTag} and ${benchmarksTag}`);
  }
}

function validateSourceRefs(releaseTag: string, benchmarks: BenchmarkRecord[]): void {
  for (const benchmark of benchmarks) {
    const sourceRef = benchmark.source.ref;
    if (sourceRef !== releaseTag && !/^[a-f0-9]{40}$/i.test(sourceRef)) {
      throw new CatalogValidationError(
        "INVALID_SOURCE_REF",
        `Benchmark ${benchmark.id} source ref must match release tag ${releaseTag} or be a full Git commit SHA: ${sourceRef}`
      );
    }
  }
}

function validateUniqueCollection(records: Array<{ id: string }>, label: string, code: string): void {
  const ids = new Set<string>();
  for (const record of records) {
    if (ids.has(record.id)) throw new CatalogValidationError(code, `Duplicate ${label} id: ${record.id}`);
    ids.add(record.id);
  }
}

function validateUniqueIds(models: ManifestModelRecord[], benchmarks: BenchmarkRecord[]): void {
  validateUniqueCollection(models, "model", "DUPLICATE_MODEL_ID");
  validateUniqueCollection(benchmarks, "benchmark", "DUPLICATE_BENCHMARK_ID");
}

function validateModelAndAssetReferences(models: ManifestModelRecord[], benchmarks: BenchmarkRecord[]): void {
  const modelsById = new Map(models.map((model) => [model.id, model]));
  for (const benchmark of benchmarks) {
    const model = modelsById.get(benchmark.sample_id);
    if (!model) {
      throw new CatalogValidationError("UNKNOWN_SAMPLE", `Benchmark ${benchmark.id} references unknown sample ${benchmark.sample_id}`);
    }
    if (benchmark.asset_filename && !model.assets.some((asset) => asset.filename === benchmark.asset_filename)) {
      throw new CatalogValidationError("UNKNOWN_ASSET", `Benchmark ${benchmark.id} references unknown asset ${benchmark.asset_filename}`);
    }
  }
}

function isSafeRelativePath(path: string): boolean {
  return !isAbsolute(path) && !win32.isAbsolute(path) && win32.parse(path).root === "" && !path.split(/[\\/]+/).includes("..");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** True when `content` contains `section` as an exact ATX heading outside a code fence. */
export function hasExactMarkdownAtxHeading(content: string, section: string): boolean {
  const normalizedSection = section.trim();
  if (!/^#{1,6}(?:[\t ]+|$)/.test(normalizedSection)) return false;

  const headingLine = new RegExp(`^ {0,3}${escapeRegExp(normalizedSection)}[\\t ]*$`);
  let fence: { character: string; length: number } | undefined;

  for (const line of content.split(/\r?\n/)) {
    const fenceMatch = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (fenceMatch) {
      const marker = fenceMatch[1]!;
      const remainder = fenceMatch[2]!;
      if (!fence) {
        fence = { character: marker[0]!, length: marker.length };
      } else if (marker[0] === fence.character && marker.length >= fence.length && remainder.trim() === "") {
        fence = undefined;
      }
      continue;
    }
    if (!fence && headingLine.test(line)) return true;
  }
  return false;
}

/** Text files whose `section` must name a real heading. */
const MARKDOWN_EXTENSION = /\.(?:md|mdx)$/i;

/**
 * Verifies that every benchmark cites evidence this repository actually holds.
 *
 * A record names `(ref, path, section)` and the ref is the immutable revision
 * the evidence was published at, so the blob is read from the repository's
 * object store (`git show <ref>:<path>`), not from the checked-out worktree.
 * On the unified branch a cited sample may not be migrated yet, and a migrated
 * sample may legitimately have rewritten its README; the ref, not the worktree,
 * is the provenance the record vouches for. Reading at the ref also keeps the
 * check working for pinned-tag builds, whose records cite the same object
 * store.
 *
 * A benchmark points at one of three kinds of source, and each is checked as
 * strictly as it can be:
 *
 * - A Markdown blob at the cited ref must contain the named section as an
 *   exact ATX heading, so a stale anchor cannot survive into the artifact.
 * - A record that names another repository (`source.repository_url`) is located
 *   outside this tree; only a non-empty path and locator are required.
 * - A non-text artifact (an evaluation screenshot, for example) is located by a
 *   caption it carries itself, which cannot be matched as a heading; the blob
 *   still has to exist at the cited ref.
 */
async function validateRepositorySources(
  repositoryRoot: string,
  repositoryUrl: string | undefined,
  source: PlatformSource,
  benchmarks: BenchmarkRecord[]
): Promise<void> {
  for (const benchmark of benchmarks) {
    const cited = benchmark.source;
    const sourcePath = cited.path;
    const section = cited.section;
    if (!sourcePath.trim() || !section.trim()) {
      throw new CatalogValidationError(
        "INVALID_SOURCE_LOCATOR",
        `Benchmark ${benchmark.id} must cite a non-empty source path and section`
      );
    }
    if (cited.repository_url && cited.repository_url !== repositoryUrl) {
      // The evidence lives in another repository, whose checkout this build has
      // no access to; the record is only required to name where it lives.
      continue;
    }
    if (!isSafeRelativePath(sourcePath)) {
      // The blob is addressed inside the git tree, but the traversal rules of a
      // worktree path still apply: no absolute paths, no `..` escapes.
      throw new CatalogValidationError("INVALID_SOURCE_PATH", `Benchmark ${benchmark.id} has an unsafe source path: ${sourcePath}`);
    }
    // A non-text artifact (an evaluation screenshot, for example) is located by
    // a caption it carries itself, which cannot be matched as a heading; only
    // its existence at the cited ref is required, so the blob is never read.
    if (!MARKDOWN_EXTENSION.test(sourcePath)) {
      if (!(await repositoryBlobExists(repositoryRoot, cited.ref, sourcePath))) {
        throw new CatalogValidationError(
          "SOURCE_NOT_FOUND",
          `${source.platform}: benchmark ${benchmark.id} source does not exist at ref ${cited.ref}: ${sourcePath}`
        );
      }
      continue;
    }
    let content: string;
    try {
      content = await readRepositoryBlob(repositoryRoot, cited.ref, sourcePath);
    } catch {
      throw new CatalogValidationError(
        "SOURCE_NOT_FOUND",
        `${source.platform}: benchmark ${benchmark.id} source does not exist at ref ${cited.ref}: ${sourcePath}`
      );
    }
    if (!hasExactMarkdownAtxHeading(content, section)) {
      throw new CatalogValidationError(
        "SOURCE_SECTION_NOT_FOUND",
        `${source.platform}: benchmark ${benchmark.id} source section is missing: ${sourcePath} ${section}`
      );
    }
  }
}

function warnOnIncompleteAccuracy(benchmarks: BenchmarkRecord[], onWarning?: (message: string) => void): void {
  if (!onWarning) return;
  const incompleteCount = benchmarks.reduce(
    (count, benchmark) => count + (benchmark.accuracy ?? []).filter((metric) => !metric.dataset).length,
    0
  );
  if (incompleteCount > 0) {
    onWarning(`Catalog warning: ${incompleteCount} accuracy metrics have no published dataset.`);
  }
}

export interface ValidateDocumentsOptions {
  repositoryRoot: string;
  source: PlatformSource;
  documents: PlatformDocuments;
  /** URL of the repository that hosts the platform trees. */
  repositoryUrl?: string;
  onWarning?: (message: string) => void;
}

/**
 * First pass, run on the manifest exactly as published: JSON Schema, release
 * identity, unique ids, and source references. Every platform runs the same
 * checks, so a source revision cannot enter the catalog unvalidated.
 */
export async function validatePublishedManifests(options: ValidateDocumentsOptions): Promise<void> {
  const { source, documents } = options;
  await validateWithSchema(options.repositoryRoot, source, documents.models, "models.schema.json");
  await validateWithSchema(options.repositoryRoot, source, documents.benchmarks, "benchmarks.schema.json");
  const releaseTag = String((documents.models.release as { tag?: unknown }).tag ?? "");
  validateReleaseTags(releaseTag, (documents.benchmarks.release as { tag?: unknown }).tag, source.platform);
  validateSourceRefs(releaseTag, documents.benchmarks.benchmarks);
  validateUniqueIds(documents.models.models, documents.benchmarks.benchmarks);
}

/**
 * Second pass, run after the documented normalisation and errata layers: asset
 * references and the evidence each benchmark cites, read at the immutable ref
 * the record names from this repository's object store.
 */
export async function validateNormalizedCatalog(options: ValidateDocumentsOptions): Promise<void> {
  const { source, documents } = options;
  validateModelAndAssetReferences(documents.models.models, documents.benchmarks.benchmarks);
  await validateRepositorySources(
    options.repositoryRoot,
    options.repositoryUrl,
    source,
    documents.benchmarks.benchmarks
  );
  warnOnIncompleteAccuracy(documents.benchmarks.benchmarks, options.onWarning);
}
