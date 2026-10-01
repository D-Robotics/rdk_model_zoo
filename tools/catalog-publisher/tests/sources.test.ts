// @vitest-environment node
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  PLATFORMS, loadSourcesDocument, readSourceFile, resolvePlatformSources,
  type PlatformSource, type SourcePin, type SourcesDocument
} from "../src/sources";
import { buildMultiplatformCatalog } from "../src/pipeline/multiplatform-catalog";
import { publisherRoot, repositoryRoot } from "./helpers/repository";
import { createTaggedRepository, fixtureManifests, fixtureSourcesDocument } from "./helpers/tag-repository";

/** The full commit the removed historical platforms/x3 tree is pinned to. */
const X3_PIN = "6fcef2b87c12435e11fbd7327ea70d4efd917b1c";

async function sourcesDocument(): Promise<SourcesDocument> {
  return loadSourcesDocument(resolve(publisherRoot, "sources.json"));
}

describe("platform source resolution", () => {
  it("resolves the unified worktree platforms and the commit-pinned x3", async () => {
    const sources = await resolvePlatformSources({ repositoryRoot, sources: await sourcesDocument() });

    expect(sources.map((source) => source.platform)).toEqual(PLATFORMS);
    for (const source of sources) {
      // X5 and S are unified into the repository root and read from the
      // worktree; the historical X3 tree was removed from the active branch
      // and is read from its pinned full-SHA commit instead.
      const unified = source.platform !== "x3";
      expect(source.kind).toBe(unified ? "worktree" : "commit");
      expect(source.worktreeRoot).toBe(unified ? "." : undefined);
      expect(source.linkRef).toBe(unified ? "develop" : X3_PIN);
      expect(source.linkPrefix).toBe(unified ? "" : "platforms/x3");
    }
    // Each distribution's own manifest directory: the unified groups live at
    // docs/release/<platform>, the pinned X3 keeps its historical root under
    // the tree prefix its commit carries.
    expect(Object.fromEntries(sources.map((source) => [source.platform, source.manifestDirectory])))
      .toEqual({ x5: "docs/release/x5", s: "docs/release/s", x3: "release" });
    // The version file travels with the manifests; the commit-pinned X3
    // resolves VERSION from the layout the pin itself carries.
    expect(Object.fromEntries(sources.map((source) => [source.platform, source.versionFile])))
      .toEqual({ x5: "docs/release/x5/VERSION", s: "docs/release/s/VERSION", x3: "VERSION" });
  });

  it("reads the removed x3 tree from the pinned commit byte-for-byte", async () => {
    const sources = await resolvePlatformSources({ repositoryRoot, sources: await sourcesDocument() });
    const x3 = sources.find((source) => source.platform === "x3")!;

    expect(x3.kind).toBe("commit");
    expect(x3.ref).toBe(X3_PIN);
    expect(x3.treePrefix).toBe("platforms/x3");
    const manifest = await readSourceFile(repositoryRoot, x3, "release/models.yaml");
    // The pinned bytes equal what `git show` itself returns for the pin.
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const git = promisify(execFile);
    const direct = await git("git", ["-C", repositoryRoot, "show", `${X3_PIN}:platforms/x3/release/models.yaml`], { encoding: "utf8" });
    expect(manifest).toBe(direct.stdout);
  });

  it("reports the exact fetch requirement when the pinned commit object is absent", async () => {
    // A fixture repository has no platforms history: the real sources.json
    // pin is missing from its object store, and the error must name the
    // object and the fetch command instead of returning an empty success.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "docs/release/x5"),
      "docs/release/x5/VERSION": "1.0.0\n"
    });

    await expect(resolvePlatformSources({
      repositoryRoot: root, sources: await sourcesDocument()
    })).rejects.toThrow(
      new RegExp(`Commit source ${X3_PIN} is not present in the local Git object store[\\s\\S]*git fetch origin ${X3_PIN}`));
  });

  it("rejects a commit pin that is not a full 40-hex SHA", async () => {
    const document = await sourcesDocument();
    const short: SourcesDocument = {
      ...document,
      sources: { ...document.sources, x3: { ...document.sources.x3!, mode: "commit", commit: "deadbeef" } }
    };

    await expect(resolvePlatformSources({ repositoryRoot, sources: short }))
      .rejects.toThrow(/full 40-hex commit SHA/);
  });

  it("rejects a commit pin that resolves to a non-commit ref", async () => {
    // An annotated tag is immutable but is not a commit; commit mode pins
    // commits so shallow-clone fetch requirements stay exact.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "docs/release/x5"),
      "docs/release/x5/VERSION": "1.0.0\n"
    });
    const document = await sourcesDocument();
    // The annotated tag OBJECT's own SHA is 40-hex, so the hex check passes
    // and the object-type check is what rejects it.
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const git = promisify(execFile);
    const { stdout } = await git("git", ["-C", root, "rev-parse", "x5-v9.9.9"], { encoding: "utf8" });
    const tagged: SourcesDocument = {
      ...document,
      sources: { ...document.sources, x3: { ...document.sources.x3!, mode: "commit", commit: stdout.trim() } }
    };

    await expect(resolvePlatformSources({ repositoryRoot: root, sources: tagged }))
      .rejects.toThrow(/Commit source required: .* is a tag/);
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
    // The tag carries a root VERSION, resolved from its own layout.
    expect(x3.versionFile).toBe("VERSION");
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

describe("pinned platform VERSION resolution", () => {
  async function resolveX5(repository: string, pin: SourcePin): Promise<PlatformSource> {
    // Fixture repositories resolve x5 through a tag pin and x3 through a
    // commit pin on the fixture's own HEAD (the real 6fcef2b8 object is not
    // part of the fixture's history).
    const sources = await resolvePlatformSources({
      repositoryRoot: repository,
      sources: await fixtureSourcesDocument(repository),
      pins: { x5: pin }
    });
    return sources.find((source) => source.platform === "x5")!;
  }

  it("resolves a pinned platform's VERSION from the layout its tag carries", async () => {
    // Regression (H8-CATALOG-R2): the advertised pin reads a historical tag
    // whose VERSION sits at its root. The worktree's colocated
    // docs/release/x5/VERSION path must not leak into the pinned read.
    const x5 = await resolveX5(repositoryRoot, { tag: "x5-v1.1.2" });

    expect(x5.kind).toBe("tag");
    expect(x5.manifestDirectory).toBe("docs/release");
    expect(x5.versionFile).toBe("VERSION");
  });

  it("builds the catalog from the advertised historical pin", async () => {
    const document = await sourcesDocument();
    const pinned = await buildMultiplatformCatalog({
      repositoryRoot,
      sources: await resolvePlatformSources({ repositoryRoot, sources: document, pins: { x5: { tag: "x5-v1.1.2" } } })
    });

    expect(pinned.release.platform_tags!.x5).toBe("x5-v1.1.2");
    const x5Variants = pinned.models.flatMap((model) => model.variants ?? [])
      .filter((variant) => variant.hardware === "x5");
    expect(x5Variants.length).toBeGreaterThan(0);
    expect(x5Variants.every((variant) => variant.source_ref === "x5-v1.1.2")).toBe(true);
  });

  it("resolves a prefixed legacy tag layout", async () => {
    // A tag that carries the platform under platforms/<id>, with the legacy
    // root VERSION inside that prefix: the probe applies the tree prefix to
    // both the manifest directory and the VERSION candidates.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "platforms/x5/docs/release"),
      "platforms/x5/VERSION": "1.0.0\n"
    });

    const x5 = await resolveX5(root, { tag: "x5-v9.9.9", treePrefix: "platforms/x5" });

    expect(x5.manifestDirectory).toBe("docs/release");
    expect(x5.versionFile).toBe("VERSION");
    expect(x5.linkPrefix).toBe("platforms/x5");
  });

  it("resolves a unified-layout tag's colocated VERSION", async () => {
    // A future tag already using the unified layout carries the version next
    // to the manifests it belongs to.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "2.0.0", "docs/release/x5"),
      "docs/release/x5/VERSION": "2.0.0\n"
    });

    const x5 = await resolveX5(root, { tag: "x5-v9.9.9" });

    expect(x5.manifestDirectory).toBe("docs/release/x5");
    expect(x5.versionFile).toBe("docs/release/x5/VERSION");
  });

  it("rejects a pinned tag whose VERSION disagrees with its manifest", async () => {
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v1.0.0", "1.0.0", "docs/release/x5"),
      "docs/release/x5/VERSION": "0.0.1\n"
    });
    const resolved = await resolvePlatformSources({
      repositoryRoot: root,
      sources: await fixtureSourcesDocument(root),
      pins: { x5: { tag: "x5-v9.9.9" } }
    });

    // The version file the tag layout selected is still checked strictly
    // against the manifest identity the release published.
    await expect(buildMultiplatformCatalog({ repositoryRoot: root, sources: resolved }))
      .rejects.toThrow(/VERSION 0\.0\.1 and manifest release version 1\.0\.0 disagree/);
  });

  it("resolves VERSION next to a fallback manifest directory", async () => {
    // The tag's manifests resolve through the probe fallback (`release/`),
    // and its VERSION sits in that same directory. The VERSION probe must use
    // the resolved directory, not the sources.json preferred one.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "release"),
      "release/VERSION": "1.0.0\n"
    });

    const x5 = await resolveX5(root, { tag: "x5-v9.9.9" });

    expect(x5.manifestDirectory).toBe("release");
    expect(x5.versionFile).toBe("release/VERSION");
  });

  it("prefers the colocated VERSION over a stale root VERSION", async () => {
    // Same fallback layout, but the tag also carries an outdated root VERSION
    // from a predecessor layout: the file colocated with the resolved
    // manifests wins, and the stale root copy must not be selected.
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "release"),
      "release/VERSION": "1.0.0\n",
      "VERSION": "0.0.1\n"
    });

    const x5 = await resolveX5(root, { tag: "x5-v9.9.9" });

    expect(x5.manifestDirectory).toBe("release");
    expect(x5.versionFile).toBe("release/VERSION");
  });

  it("rejects a tag that carries manifests but no VERSION", async () => {
    const root = await createTaggedRepository("x5-v9.9.9", {
      ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "docs/release/x5")
    });

    await expect(resolveX5(root, { tag: "x5-v9.9.9" }))
      .rejects.toThrow(/no VERSION found under x5-v9\.9\.9 \(tried docs\/release\/x5\/VERSION, VERSION\)/);
  });

  it("keeps worktree VERSION reads strict on the configured path", async () => {
    const document = await sourcesDocument();
    const broken: SourcesDocument = {
      ...document,
      sources: {
        ...document.sources,
        x5: { ...document.sources.x5!, version_file: "docs/release/x5/NONEXISTENT" }
      }
    };
    const sources = await resolvePlatformSources({ repositoryRoot, sources: broken });
    const x5 = sources.find((source) => source.platform === "x5")!;

    // No layout probing for worktree sources: the configured file is the one
    // read, and a missing one fails the build instead of silently falling
    // back to a colocated or root VERSION.
    expect(x5.versionFile).toBe("docs/release/x5/NONEXISTENT");
    await expect(readSourceFile(repositoryRoot, x5, x5.versionFile)).rejects.toThrow();
  });
});
