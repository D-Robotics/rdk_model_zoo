import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import { buildMultiplatformCatalog } from "./multiplatform-catalog";

const siteDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(siteDirectory, "../../..");

const catalog = await buildMultiplatformCatalog(repositoryRoot);
const outputPath = resolve(siteDirectory, "../public/data/catalog.json");
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
