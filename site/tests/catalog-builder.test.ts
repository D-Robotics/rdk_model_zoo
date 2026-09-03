// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildCatalog } from "../scripts/catalog-builder";

describe("buildCatalog", () => {
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
});
