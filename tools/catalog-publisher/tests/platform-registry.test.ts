// @vitest-environment node
import { access, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parse } from "yaml";
import { describe, expect, it } from "vitest";
import { loadSourcesDocument } from "../src/sources";
import { publisherRoot, repositoryRoot } from "./helpers/repository";

interface RegistryEntry {
  id: string;
  display_name: string;
  path: string;
  release_line: string;
  release_tag: string;
  version: string;
  hardware: string[];
  manifest_directory: string;
  release_notes: string;
  readme: string;
  readme_cn: string;
  guidelines: string | null;
  license: string | null;
}

async function registry(): Promise<{ schema_version: number; platforms: RegistryEntry[] }> {
  return JSON.parse(await readFile(join(repositoryRoot, "platforms", "registry.json"), "utf8"));
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(resolve(repositoryRoot, path));
    return true;
  } catch {
    return false;
  }
}

/**
 * `platforms/registry.json` is the published statement of what main ships. It
 * is checked against the distributions themselves so a stale or optimistic
 * entry cannot survive.
 */
describe("platform registry", () => {
  it("registers every platform the catalog build reads, and nothing else", async () => {
    const document = await registry();
    const sources = await loadSourcesDocument(resolve(publisherRoot, "sources.json"));

    expect(document.schema_version).toBe(1);
    expect(document.platforms.map((entry) => entry.id).sort()).toEqual(Object.keys(sources.sources).sort());
    for (const entry of document.platforms) {
      expect(entry.path).toBe(`platforms/${entry.id}`);
      expect(sources.sources[entry.id as keyof typeof sources.sources]!.path).toBe(entry.path);
      expect(sources.sources[entry.id as keyof typeof sources.sources]!.manifest_root)
        .toBe(entry.manifest_directory.slice(`platforms/${entry.id}/`.length));
      expect((await readFile(join(repositoryRoot, "platforms", entry.id, "VERSION"), "utf8")).trim())
        .toBe(entry.version);
    }
  });

  it("publishes the release identity each manifest actually declares", async () => {
    for (const entry of (await registry()).platforms) {
      const manifest = parse(
        await readFile(resolve(repositoryRoot, entry.manifest_directory, "models.yaml"), "utf8")
      ) as { release: { platform: string; version: string; tag: string; branch: string } };

      expect(manifest.release.platform).toBe(entry.id);
      expect(`v${manifest.release.version}`).toBe(`v${entry.version}`);
      expect(manifest.release.tag).toBe(entry.release_tag);
      expect(manifest.release.branch).toBe(entry.release_line);
      expect(entry.hardware.length).toBeGreaterThan(0);
    }
  });

  it("points every documented file at a path that exists", async () => {
    for (const entry of (await registry()).platforms) {
      expect(await exists(entry.readme)).toBe(true);
      expect(await exists(entry.readme_cn)).toBe(true);
      expect(await exists(entry.release_notes)).toBe(true);
      expect(await exists(`${entry.manifest_directory}/benchmarks.yaml`)).toBe(true);
      expect(await exists(`${entry.manifest_directory}/schemas/models.schema.json`)).toBe(true);
      expect(await exists(`${entry.manifest_directory}/schemas/benchmarks.schema.json`)).toBe(true);
      if (entry.guidelines) expect(await exists(entry.guidelines)).toBe(true);
      if (entry.license) expect(await exists(entry.license)).toBe(true);
    }
  });

  it("does not claim a license the distribution never published", async () => {
    const entries = (await registry()).platforms;
    const x3 = entries.find((entry) => entry.id === "x3")!;

    // Upstream RDK X3 ships no license file; the registry must say so rather
    // than inventing a path or silently borrowing another platform's license.
    expect(x3.license).toBeNull();
    expect(await exists("platforms/x3/LICENSE")).toBe(false);
    expect(entries.filter((entry) => entry.license !== null).map((entry) => entry.id).sort())
      .toEqual(["s", "x5"]);
  });
});
