import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { writeCatalog } from "./catalog-builder";

const siteDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(siteDirectory, "../..");

await writeCatalog({
  repositoryRoot,
  modelsPath: "release/models.yaml",
  benchmarksPath: "release/benchmarks.yaml",
  modelsSchemaPath: "release/schemas/models.schema.json",
  benchmarksSchemaPath: "release/schemas/benchmarks.schema.json",
  outputPath: resolve(siteDirectory, "../public/data/catalog.json")
});
