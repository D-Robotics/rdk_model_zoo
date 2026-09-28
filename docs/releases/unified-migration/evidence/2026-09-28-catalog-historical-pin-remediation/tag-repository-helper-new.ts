import { execFile } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

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
