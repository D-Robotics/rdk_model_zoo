// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildCatalog } from "../scripts/catalog-builder";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

function buildRepositoryCatalog() {
  return buildCatalog({
    repositoryRoot,
    modelsPath: "release/models.yaml",
    benchmarksPath: "release/benchmarks.yaml",
    modelsSchemaPath: "release/schemas/models.schema.json",
    benchmarksSchemaPath: "release/schemas/benchmarks.schema.json"
  });
}

describe("buildCatalog", () => {
  it("includes exact published evidence and excludes YOLOE", async () => {
    const catalog = await buildRepositoryCatalog();
    const himloco = catalog.models.find((model) => model.id === "himloco");
    const cpp = himloco?.benchmarks.find((record) => record.id === "himloco-cpp-runtime-x5");
    expect(cpp?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 2853.09,
      unit: "fps",
      qualifier: "exact"
    }));
    const text = JSON.stringify(catalog).toLowerCase();
    expect(text).not.toContain("yoloe");
  });

  it("keeps plus-suffixed throughput as reported rather than inventing an exact value", async () => {
    const catalog = await buildRepositoryCatalog();
    const atto = catalog.models.find((model) => model.id === "convnext")
      ?.benchmarks.find((record) => record.variant_id === "convnext-atto-224");
    expect(atto?.performance).toContainEqual(expect.objectContaining({
      metric: "throughput",
      value: 732,
      unit: "fps",
      qualifier: "lower-bound"
    }));
  });

  it("joins models and benchmarks and preserves the release tag", async () => {
    const catalog = await buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.valid.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    });
    expect(catalog.release.tag).toBe("x5-v1.0.0");
    expect(catalog.models[0]?.benchmarks[0]?.sample_id).toBe("convnext");
  });

  it("rejects a benchmark whose sample_id is absent from models.yaml", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/invalid-reference/", import.meta.url)),
      modelsPath: "models.yaml",
      benchmarksPath: "benchmarks.yaml",
      modelsSchemaPath: "../../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "UNKNOWN_SAMPLE"
    }));
  });

  it("rejects a benchmark whose source ref is a mutable branch", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.mutable-ref.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "INVALID_SOURCE_REF"
    }));
  });
});
