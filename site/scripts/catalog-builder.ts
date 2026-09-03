import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, win32 } from "node:path";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { parse } from "yaml";
import type { BenchmarkRecord, Catalog, ModelRecord } from "../src/catalog/types";

export interface BuildCatalogOptions {
  repositoryRoot: string;
  modelsPath: string;
  benchmarksPath: string;
  modelsSchemaPath: string;
  benchmarksSchemaPath: string;
}

export class CatalogValidationError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message);
    this.name = "CatalogValidationError";
  }
}

interface ModelsDocument {
  schema_version: number;
  release: Catalog["release"];
  summary?: Record<string, number>;
  models: ModelManifestRecord[];
}

interface ModelManifestRecord extends Omit<ModelRecord, "assets" | "benchmarks"> {
  assets: Array<{ filename: string; format: string; url?: string | null; sha256?: string | null }>;
}

type ModelManifestAsset = ModelManifestRecord["assets"][number];

interface BenchmarksDocument {
  release: { tag: string };
  benchmarks: BenchmarkRecord[];
}

async function readYaml(path: string): Promise<unknown> {
  return parse(await readFile(path, "utf8"));
}

function formatSchemaErrors(errors: { instancePath?: string; message?: string }[] | null | undefined): string {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`).join("; ");
}

async function validateWithSchema(document: unknown, schemaPath: string, code: string): Promise<void> {
  const schema = JSON.parse(await readFile(schemaPath, "utf8")) as object;
  const ajv = new Ajv2020({ allErrors: true, strict: true });
  addFormats(ajv);
  const validate = ajv.compile(schema);
  if (!validate(document)) {
    throw new CatalogValidationError(code, `${code}: ${formatSchemaErrors(validate.errors)}`);
  }
  validateFiniteNumbers(document, code);
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

function validateReleaseTags(modelsTag: string, benchmarksTag: string): void {
  if (modelsTag !== benchmarksTag) {
    throw new CatalogValidationError("RELEASE_TAG_MISMATCH", `Release tags differ: ${modelsTag} and ${benchmarksTag}`);
  }
}

function validateUniqueIds(models: ModelsDocument["models"], benchmarks: BenchmarkRecord[]): void {
  validateUniqueCollection(models, "model", "DUPLICATE_MODEL_ID");
  validateUniqueCollection(benchmarks, "benchmark", "DUPLICATE_BENCHMARK_ID");
}

function validateUniqueCollection(records: Array<{ id: string }>, label: string, code: string): void {
  const ids = new Set<string>();
  for (const record of records) {
    if (ids.has(record.id)) {
      throw new CatalogValidationError(code, `Duplicate ${label} id: ${record.id}`);
    }
    ids.add(record.id);
  }
}

function validateModelAndAssetReferences(models: ModelsDocument["models"], benchmarks: BenchmarkRecord[]): void {
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

async function validateRepositorySources(repositoryRoot: string, benchmarks: BenchmarkRecord[]): Promise<void> {
  const root = resolve(repositoryRoot);
  for (const benchmark of benchmarks) {
    const sourcePath = benchmark.source.path;
    if (!isSafeRelativePath(sourcePath)) {
      throw new CatalogValidationError("INVALID_SOURCE_PATH", `Benchmark ${benchmark.id} has an unsafe source path: ${sourcePath}`);
    }
    const sourceFile = resolve(root, sourcePath);
    const sourceRelative = relative(root, sourceFile);
    if (isAbsolute(sourceRelative) || win32.isAbsolute(sourceRelative) || sourceRelative === ".." || sourceRelative.startsWith("../") || sourceRelative.startsWith("..\\")) {
      throw new CatalogValidationError("INVALID_SOURCE_PATH", `Benchmark ${benchmark.id} has an unsafe source path: ${sourcePath}`);
    }
    let content: string;
    try {
      content = await readFile(sourceFile, "utf8");
    } catch {
      throw new CatalogValidationError("SOURCE_NOT_FOUND", `Benchmark ${benchmark.id} source does not exist: ${sourcePath}`);
    }
    if (!content.includes(benchmark.source.section)) {
      throw new CatalogValidationError("SOURCE_SECTION_NOT_FOUND", `Benchmark ${benchmark.id} source section is missing: ${benchmark.source.section}`);
    }
  }
}

function validateNoYoloe(modelsDocument: ModelsDocument, benchmarksDocument: BenchmarksDocument): void {
  const catalogContent = JSON.stringify({ modelsDocument, benchmarksDocument }).toLowerCase();
  if (catalogContent.includes("yoloe")) {
    throw new CatalogValidationError("EXCLUDED_YOLOE", "YOLOE entries are excluded from the catalog");
  }
}

function normalizeAsset({ url, ...asset }: ModelManifestAsset): ModelRecord["assets"][number] {
  return typeof url === "string" ? { ...asset, url } : asset;
}

function joinCatalog(modelsDocument: ModelsDocument, benchmarksDocument: BenchmarksDocument): Catalog {
  const benchmarksBySample = new Map<string, BenchmarkRecord[]>();
  for (const benchmark of benchmarksDocument.benchmarks) {
    const records = benchmarksBySample.get(benchmark.sample_id) ?? [];
    records.push(benchmark);
    benchmarksBySample.set(benchmark.sample_id, records);
  }
  return {
    schema_version: modelsDocument.schema_version,
    release: modelsDocument.release,
    summary: modelsDocument.summary ?? {},
    models: modelsDocument.models.map(({ assets, ...model }) => ({
      ...model,
      assets: assets.map(normalizeAsset),
      benchmarks: benchmarksBySample.get(model.id) ?? []
    }))
  };
}

export async function buildCatalog(options: BuildCatalogOptions): Promise<Catalog> {
  const modelsDocument = await readYaml(resolve(options.repositoryRoot, options.modelsPath));
  const benchmarksDocument = await readYaml(resolve(options.repositoryRoot, options.benchmarksPath));
  await validateWithSchema(modelsDocument, resolve(options.repositoryRoot, options.modelsSchemaPath), "MODELS_SCHEMA");
  await validateWithSchema(benchmarksDocument, resolve(options.repositoryRoot, options.benchmarksSchemaPath), "BENCHMARKS_SCHEMA");
  const models = modelsDocument as ModelsDocument;
  const benchmarks = benchmarksDocument as BenchmarksDocument;
  validateReleaseTags(models.release.tag, benchmarks.release.tag);
  validateUniqueIds(models.models, benchmarks.benchmarks);
  validateModelAndAssetReferences(models.models, benchmarks.benchmarks);
  await validateRepositorySources(options.repositoryRoot, benchmarks.benchmarks);
  validateNoYoloe(models, benchmarks);
  return joinCatalog(models, benchmarks);
}

export async function writeCatalog(options: BuildCatalogOptions & { outputPath: string }): Promise<Catalog> {
  const catalog = await buildCatalog(options);
  await mkdir(dirname(options.outputPath), { recursive: true });
  await writeFile(options.outputPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  return catalog;
}
