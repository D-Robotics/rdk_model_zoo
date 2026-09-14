// @vitest-environment node
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  CATALOG_FILE, CATALOG_METADATA_FILE, readVerifiedArtifact, writeCatalogArtifact
} from "../src/artifact";
import type { CatalogArtifactMetadata } from "../src/artifact";
import type { Catalog } from "../src/catalog/types";
import { repositoryCatalog } from "./helpers/repository";

const GENERATOR = "tools/catalog-publisher";

async function scratch(): Promise<string> {
  return mkdtemp(join(tmpdir(), "catalog-artifact-"));
}

async function readMetadata(directory: string): Promise<CatalogArtifactMetadata> {
  return JSON.parse(await readFile(join(directory, CATALOG_METADATA_FILE), "utf8")) as CatalogArtifactMetadata;
}

async function writeMetadata(directory: string, metadata: CatalogArtifactMetadata): Promise<void> {
  await writeFile(join(directory, CATALOG_METADATA_FILE), `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
}

describe("published catalog artifact", () => {
  it("writes a checksummed payload and the metadata that describes it", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    const { metadata } = await writeCatalogArtifact(catalog, directory, GENERATOR);

    const payload = await readFile(join(directory, CATALOG_FILE), "utf8");
    const onDisk = await readMetadata(directory);

    expect(metadata.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(onDisk.sha256).toBe(metadata.sha256);
    expect(onDisk.bytes).toBe(Buffer.byteLength(payload, "utf8"));
    expect(onDisk.payload).toBe(CATALOG_FILE);
    expect(onDisk.catalog_version).toBe(catalog.release.catalog_version);
    expect(onDisk.release_tag).toBe(catalog.release.tag);
    expect(onDisk.platform_tags).toEqual(catalog.release.platform_tags);
    expect(onDisk.summary).toEqual(catalog.summary);
    expect(onDisk.generator).toEqual({ name: "catalog-publisher", path: GENERATOR });
  });

  it("is reproducible: the same catalog serialises to the same bytes", async () => {
    const catalog = await repositoryCatalog();
    const first = await scratch();
    const second = await scratch();
    const a = await writeCatalogArtifact(catalog, first, GENERATOR);
    const b = await writeCatalogArtifact(catalog, second, GENERATOR);

    expect(a.metadata.sha256).toBe(b.metadata.sha256);
    expect(await readFile(join(first, CATALOG_FILE), "utf8"))
      .toBe(await readFile(join(second, CATALOG_FILE), "utf8"));
  });

  it("round-trips through the verified reader", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    await writeCatalogArtifact(catalog, directory, GENERATOR);

    const read = await readVerifiedArtifact(directory);
    const parsed = JSON.parse(read.payload) as Catalog;
    expect(parsed.release.catalog_version).toBe(catalog.release.catalog_version);
    expect(parsed.models).toHaveLength(catalog.models.length);
  });

  it("fails when the payload no longer matches its checksum", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    await writeCatalogArtifact(catalog, directory, GENERATOR);

    const payloadPath = join(directory, CATALOG_FILE);
    const payload = JSON.parse(await readFile(payloadPath, "utf8")) as Catalog;
    payload.summary = { ...payload.summary, sample_count: payload.summary.sample_count! + 1 };
    await writeFile(payloadPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

    await expect(readVerifiedArtifact(directory)).rejects.toThrow(/checksum mismatch/i);
  });

  it("fails when the payload is truncated even though it still parses", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    await writeCatalogArtifact(catalog, directory, GENERATOR);

    const payloadPath = join(directory, CATALOG_FILE);
    const payload = await readFile(payloadPath, "utf8");
    // Trailing whitespace keeps the JSON valid but changes the byte count.
    await writeFile(payloadPath, `${payload}\n`, "utf8");

    await expect(readVerifiedArtifact(directory)).rejects.toThrow(/checksum mismatch/i);
  });

  it("fails when the metadata was generated for a different data revision", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    await writeCatalogArtifact(catalog, directory, GENERATOR);

    const metadata = await readMetadata(directory);
    metadata.catalog_version = "multi-0.0.0+x5:x5-v0.0.0+s:s-v0.0.0+x3:x3-v0.0.0";
    await writeMetadata(directory, metadata);

    await expect(readVerifiedArtifact(directory))
      .rejects.toThrow(/version mismatch/i);
  });

  it("refuses a metadata contract it does not understand", async () => {
    const catalog = await repositoryCatalog();
    const directory = await scratch();
    await writeCatalogArtifact(catalog, directory, GENERATOR);

    const metadata = await readMetadata(directory);
    (metadata as { schema_version: number }).schema_version = 99;
    await writeMetadata(directory, metadata);

    await expect(readVerifiedArtifact(directory)).rejects.toThrow(/Unsupported catalog metadata/);
  });
});
