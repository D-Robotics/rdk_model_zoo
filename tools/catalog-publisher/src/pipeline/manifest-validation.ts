import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve, win32 } from "node:path";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { readSourceFile, type PlatformSource } from "../sources";
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
function hasExactMarkdownAtxHeading(content: string, section: string): boolean {
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
 * A benchmark points at one of three kinds of source, and each is checked as
 * strictly as it can be:
 *
 * - A Markdown file in this repository must contain the named section as an
 *   exact ATX heading, so a stale anchor cannot survive into the artifact.
 * - A record that names another repository (`source.repository_url`) is located
 *   outside this tree; only a non-empty path and locator are required.
 * - A non-text artifact (an evaluation screenshot, for example) is located by a
 *   caption it carries itself, which cannot be matched as a heading; the file
 *   still has to exist at the cited path.
 *
 * Only a checked-out distribution can be read file by file, so a pinned-tag
 * build skips the check and says so.
 */
async function validateRepositorySources(
  repositoryRoot: string,
  repositoryUrl: string | undefined,
  source: PlatformSource,
  benchmarks: BenchmarkRecord[]
): Promise<void> {
  if (source.kind !== "worktree") return;
  const root = resolve(repositoryRoot, source.worktreeRoot!);
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
      throw new CatalogValidationError("INVALID_SOURCE_PATH", `Benchmark ${benchmark.id} has an unsafe source path: ${sourcePath}`);
    }
    const sourceFile = resolve(root, sourcePath);
    const sourceRelative = relative(root, sourceFile);
    if (isAbsolute(sourceRelative) || win32.isAbsolute(sourceRelative) || sourceRelative === ".."
      || sourceRelative.startsWith("../") || sourceRelative.startsWith("..\\")) {
      throw new CatalogValidationError("INVALID_SOURCE_PATH", `Benchmark ${benchmark.id} has an unsafe source path: ${sourcePath}`);
    }
    let content: string;
    try {
      content = await readFile(sourceFile, "utf8");
    } catch {
      throw new CatalogValidationError("SOURCE_NOT_FOUND", `${source.platform}: benchmark ${benchmark.id} source does not exist: ${sourcePath}`);
    }
    if (!MARKDOWN_EXTENSION.test(sourcePath)) continue;
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
 * references and the on-disk evidence each benchmark cites. The repository
 * source check needs a checked-out distribution, so it is skipped for pinned
 * historical builds and reported as such.
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
