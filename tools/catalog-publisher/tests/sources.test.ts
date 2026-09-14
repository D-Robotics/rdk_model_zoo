// @vitest-environment node
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { PLATFORMS, loadSourcesDocument, resolvePlatformSources, type SourcesDocument } from "../src/sources";
import { buildMultiplatformCatalog } from "../src/pipeline/multiplatform-catalog";
import { publisherRoot, repositoryRoot } from "./helpers/repository";

async function sourcesDocument(): Promise<SourcesDocument> {
  return loadSourcesDocument(resolve(publisherRoot, "sources.json"));
}

describe("platform source resolution", () => {
  it("resolves every platform from the checked-out distribution by default", async () => {
    const sources = await resolvePlatformSources({ repositoryRoot, sources: await sourcesDocument() });

    expect(sources.map((source) => source.platform)).toEqual(PLATFORMS);
    for (const source of sources) {
      expect(source.kind).toBe("worktree");
      expect(source.worktreeRoot).toBe(`platforms/${source.platform}`);
      expect(source.linkRef).toBe("main");
      expect(source.linkPrefix).toBe(`platforms/${source.platform}`);
    }
    // The historical X3 layout keeps its manifests at the platform root; the
    // probe finds each distribution's own manifest directory, so no platform
    // needs a special case in the build.
    expect(Object.fromEntries(sources.map((source) => [source.platform, source.manifestDirectory])))
      .toEqual({ x5: "docs/release", s: "docs/release", x3: "release" });
  });

  it("reads a pinned platform from its frozen tag with the layout that tag carries", async () => {
    const sources = await resolvePlatformSources({
      repositoryRoot,
      sources: await sourcesDocument(),
      pins: { x3: { tag: "x3-v1.1.2" } }
    });
    const x3 = sources.find((source) => source.platform === "x3")!;

    expect(x3.kind).toBe("tag");
    expect(x3.ref).toBe("x3-v1.1.2");
    // The tag predates the platform split, so its tree has no prefix.
    expect(x3.treePrefix).toBe("");
    expect(x3.linkRef).toBe("x3-v1.1.2");
    expect(x3.linkPrefix).toBe("");
    expect(x3.manifestDirectory).toBe("release");
  });

  it("keeps pinned-tag source links prefix-free while the worktree build is prefixed", async () => {
    const document = await sourcesDocument();
    const pinned = await buildMultiplatformCatalog({
      repositoryRoot,
      sources: await resolvePlatformSources({ repositoryRoot, sources: document, pins: { x3: { tag: "x3-v1.1.2" } } })
    });
    const x3Variants = pinned.models.flatMap((model) => model.variants ?? [])
      .filter((variant) => variant.hardware === "x3");

    expect(x3Variants.length).toBeGreaterThan(0);
    expect(x3Variants.every((variant) => variant.source_ref === "x3-v1.1.2")).toBe(true);
    expect(x3Variants.every((variant) => variant.source_path_prefix === "")).toBe(true);
    expect(pinned.release.platform_tags!.x3).toBe("x3-v1.1.2");
  });

  it("rejects a pin that does not name an annotated tag", async () => {
    await expect(resolvePlatformSources({
      repositoryRoot,
      sources: await sourcesDocument(),
      // A branch is mutable, so it cannot stand in for a released distribution.
      pins: { s: { tag: "HEAD" } }
    })).rejects.toThrow(/Annotated source tag required/);
  });

  it("rejects a sources document with an unsupported schema version", async () => {
    const document = await sourcesDocument();
    const directory = await mkdtemp(join(tmpdir(), "catalog-sources-"));
    const path = join(directory, "sources.json");
    await writeFile(path, `${JSON.stringify({ ...document, schema_version: 2 }, null, 2)}\n`, "utf8");

    await expect(loadSourcesDocument(path)).rejects.toThrow(/Unsupported catalog sources schema_version/);
    expect(document.schema_version).toBe(1);
    expect(document.repository).toMatch(/^https:\/\/github\.com\//);
  });

  it("rejects a sources document that omits a platform", async () => {
    const document = await sourcesDocument();
    const incomplete: SourcesDocument = { ...document, sources: { ...document.sources } };
    delete incomplete.sources.x3;

    await expect(resolvePlatformSources({ repositoryRoot, sources: incomplete }))
      .rejects.toThrow(/do not describe platform x3/);
  });

  it("reports a platform whose manifest cannot be found anywhere it probes", async () => {
    const document = await sourcesDocument();
    const broken: SourcesDocument = {
      ...document,
      sources: { ...document.sources, x5: { ...document.sources.x5!, path: "platforms/x5/samples" } }
    };

    await expect(resolvePlatformSources({ repositoryRoot, sources: broken }))
      .rejects.toThrow(/no models.yaml found/);
  });
});
