// @vitest-environment node
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { repositoryRoot } from "./helpers/repository";

const WORKFLOW_DIRECTORY = join(repositoryRoot, ".github", "workflows");
const DATA_WORKFLOW = join(WORKFLOW_DIRECTORY, "model-catalog-data.yml");

async function workflowNames(): Promise<string[]> {
  return (await readdir(WORKFLOW_DIRECTORY)).filter((name) => /\.ya?ml$/.test(name)).sort();
}

/**
 * The catalog data package is the only thing this repository publishes from CI.
 * These checks keep a website deployment from creeping back into a model
 * branch: the data workflow may build and upload, never deploy. The sample
 * contract workflow only runs host-side checks and uploads nothing.
 */
describe("catalog release workflow", () => {
  it("runs exactly the data and sample-contract workflows and deploys no website", async () => {
    expect(await workflowNames()).toEqual(["model-catalog-data.yml", "sample-contract.yml"]);

    const workflows = await Promise.all((await workflowNames()).map((name) => readFile(join(WORKFLOW_DIRECTORY, name), "utf8")));
    for (const workflow of workflows) {
      for (const forbidden of [
        "actions/deploy-pages",
        "actions/upload-pages-artifact",
        "actions/configure-pages",
        "pages: write",
        "id-token: write",
        "environment:"
      ]) {
        expect(workflow).not.toContain(forbidden);
      }
    }
  });

  it("reads every platform distribution through the publisher", async () => {
    const workflow = await readFile(DATA_WORKFLOW, "utf8");

    expect(workflow).toContain('working-directory: tools/catalog-publisher');
    expect(workflow).toContain("run: npm ci");
    expect(workflow).toContain("run: npm run check");
    // Benchmark records cite immutable commits, so the source check needs the
    // history rather than a shallow clone.
    expect(workflow).toContain("fetch-depth: 0");
  });

  it("uploads the catalog package with its checksum contract", async () => {
    const workflow = await readFile(DATA_WORKFLOW, "utf8");

    expect(workflow).toContain("actions/upload-artifact");
    expect(workflow).toContain("tools/catalog-publisher/dist/catalog.json");
    expect(workflow).toContain("tools/catalog-publisher/dist/catalog.meta.json");
  });

  it("enforces the checksum contract through the publisher, not a second implementation", async () => {
    const manifest = JSON.parse(
      await readFile(join(repositoryRoot, "tools", "catalog-publisher", "package.json"), "utf8")
    ) as { scripts: Record<string, string> };
    const workflow = await readFile(DATA_WORKFLOW, "utf8");

    // `verifyArtifact` is the single implementation of the metadata contract.
    // The workflow must reach it by running the publisher's own check, not by
    // re-deriving the digest inline.
    expect(manifest.scripts.check).toContain("catalog:check");
    expect(manifest.scripts.check).toContain("validate:sources");
    expect(workflow).toContain("npm run check");
    expect(workflow).not.toContain("createHash");
    expect(workflow).not.toContain("digest(");
  });
});
