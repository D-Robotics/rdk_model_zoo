import { execFile } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";
import type { SourcesDocument } from "../../src/sources";

const run = promisify(execFile);

/**
 * Minimal one-platform manifest pair; enough for source resolution probes.
 * `directory` places the pair inside the tag tree (e.g. "docs/release/x5").
 */
export function fixtureManifests(
  platform: string,
  tag: string,
  version = "1.0.0",
  directory = "."
): Record<string, string> {
  const release = `schema_version: 1\nrelease:\n  platform: ${platform}\n  version: ${version}\n  tag: ${tag}\n`;
  const prefix = directory.replace(/^\.\/?|\/$/g, "");
  return {
    [prefix ? `${prefix}/models.yaml` : "models.yaml"]: `${release}models: []\n`,
    [prefix ? `${prefix}/benchmarks.yaml` : "benchmarks.yaml"]: `${release}benchmarks: []\n`
  };
}

/**
 * Creates a throwaway git repository holding one annotated tag with the given
 * file layout, plus the stub worktree manifests the other two platforms' fixed
 * sources.json paths probe for. The repository is fully isolated from the
 * model zoo checkout: resolution and builds that read the fixture never touch
 * real tags, refs or files.
 */
export async function createTaggedRepository(
  tag: string,
  files: Record<string, string>,
  stubOtherPlatforms = true
): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "catalog-tag-fixture-"));
  const all: Record<string, string> = { ...files };
  if (stubOtherPlatforms) {
    all["docs/release/s/models.yaml"] = "schema_version: 1\nrelease:\n  platform: s\n  version: 1.0.0\n  tag: s-v1.0.0\nmodels: []\n";
    all["platforms/x3/release/models.yaml"] = "schema_version: 1\nrelease:\n  platform: x3\n  version: 1.0.0\n  tag: x3-v1.0.0\nmodels: []\n";
    // Commit-mode x3 resolution probes VERSION inside the pinned tree.
    all["platforms/x3/VERSION"] = "1.0.0\n";
  }
  for (const [path, content] of Object.entries(all)) {
    await mkdir(dirname(join(root, path)), { recursive: true });
    await writeFile(join(root, path), content);
  }
  const git = (args: string[]) => run("git", [
    "-C", root,
    "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
    ...args
  ]);
  await git(["init", "-q"]);
  await git(["add", "-A"]);
  await git(["commit", "-qm", "fixture layout"]);
  await git(["tag", "-a", tag, "-m", `${tag} fixture`]);
  return root;
}

/**
 * A sources document for fixture repositories: X5/S keep the unified
 * worktree layout, and X3 uses commit mode pinned to the fixture's own
 * HEAD (the fixture commits its `platforms/x3` stub), mirroring how the
 * real sources.json pins the removed historical tree to a full commit SHA.
 */
export async function fixtureSourcesDocument(root: string): Promise<SourcesDocument> {
  const { stdout } = await run("git", ["-C", root, "rev-parse", "HEAD"]);
  const commit = stdout.trim();
  return {
    schema_version: 1,
    repository: "https://github.com/D-Robotics/rdk_model_zoo",
    sources: {
      x5: {
        mode: "worktree", path: ".", manifest_root: "docs/release/x5",
        version_file: "docs/release/x5/VERSION", link_ref: "develop", link_prefix: ""
      },
      s: {
        mode: "worktree", path: ".", manifest_root: "docs/release/s",
        version_file: "docs/release/s/VERSION", link_ref: "develop", link_prefix: ""
      },
      x3: {
        mode: "commit", commit, tree_prefix: "platforms/x3", manifest_root: "release",
        link_ref: commit, link_prefix: "platforms/x3"
      }
    }
  };
}
