// @vitest-environment node
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildCatalog } from "../scripts/catalog-builder";
import type { BenchmarkRecord, MetricRecord } from "../src/catalog/types";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const EXPECTED_MODEL_COUNT = 36;
const EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS = new Set([
  "clip",
  "vargconvnet",
  "yoloworld"
]);

function buildRepositoryCatalog() {
  return buildCatalog({
    repositoryRoot,
    modelsPath: "release/models.yaml",
    benchmarksPath: "release/benchmarks.yaml",
    modelsSchemaPath: "release/schemas/models.schema.json",
    benchmarksSchemaPath: "release/schemas/benchmarks.schema.json"
  });
}

function recordContext(record: BenchmarkRecord) {
  return [
    record.source.ref,
    record.source.path,
    record.source.section,
    record.sample_id,
    record.variant_id,
    record.asset_filename ?? null,
    record.model_format ?? null,
    record.precision ?? null,
    record.input?.shape ?? null,
    record.input?.layout ?? null,
    record.input?.format ?? null,
    record.environment.hardware,
    record.environment.rdk_os ?? null,
    record.environment.runtime ?? null,
    record.environment.cpu_mode ?? null,
    record.environment.bpu_cores ?? null
  ];
}

function metricContext(kind: "performance" | "accuracy", metric: MetricRecord) {
  return [
    kind,
    metric.metric,
    metric.unit,
    metric.value,
    metric.qualifier ?? null,
    metric.statistic ?? null,
    metric.model_stage ?? null,
    metric.dataset ?? null,
    metric.concurrency ?? null,
    metric.scope ?? null
  ];
}

function metricKeys(record: BenchmarkRecord) {
  const context = recordContext(record);
  return (["performance", "accuracy"] as const).flatMap((kind) =>
    (record[kind] ?? []).map((metric) => JSON.stringify([...context, ...metricContext(kind, metric)]))
  );
}

function recordKey(record: BenchmarkRecord) {
  return JSON.stringify([
    ...recordContext(record),
    ...metricKeys(record).sort()
  ]);
}

function duplicates(values: string[]) {
  const seen = new Set<string>();
  const repeated = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      repeated.add(value);
    }
    seen.add(value);
  }
  return [...repeated];
}

describe("audited benchmark coverage", () => {
  it("accounts for the complete release inventory", async () => {
    const catalog = await buildRepositoryCatalog();
    const emptyModelIds = catalog.models
      .filter((model) => model.benchmarks.length === 0)
      .map((model) => model.id)
      .sort();
    const expectedEmptyModelIds = [...EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS].sort();

    expect(catalog.models).toHaveLength(EXPECTED_MODEL_COUNT);
    expect(emptyModelIds).toEqual(expectedEmptyModelIds);
    expect(catalog.models.filter(
      (model) => model.benchmarks.length === 0 && !EXPECTED_WITHOUT_PUBLISHED_BENCHMARKS.has(model.id)
    )).toEqual([]);
  });

  it("keeps benchmark identifiers and semantic evidence unique", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.flatMap((model) => model.benchmarks);

    expect(duplicates(records.map((record) => record.id))).toEqual([]);
    expect(duplicates(records.map(recordKey))).toEqual([]);
    expect(duplicates(records.flatMap(metricKeys))).toEqual([]);
  });

  it("keeps immutable sources resolvable and the prohibited family absent", async () => {
    const catalog = await buildRepositoryCatalog();
    const records = catalog.models.flatMap((model) => model.benchmarks);

    expect(records.every((record) => record.source.ref === catalog.release.tag)).toBe(true);
    expect(records.every((record) => record.source.path.length > 0 && record.source.section.length > 0)).toBe(true);

    const prohibitedFamily = ["yolo", "e"].join("");
    expect(JSON.stringify(catalog).toLowerCase()).not.toContain(prohibitedFamily);
  });
});
