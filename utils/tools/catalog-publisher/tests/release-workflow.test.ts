// @vitest-environment node
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { repositoryRoot } from "./helpers/repository";

const WORKFLOW_DIRECTORY = join(repositoryRoot, ".github", "workflows");
const DATA_WORKFLOW = join(WORKFLOW_DIRECTORY, "model-catalog-data.yml");

async function workflowNames(): Promise<string[]> {
  return (await readdir(WORKFLOW_DIRECTORY)).filter((name) => /\.ya?ml$/.test(name)).sort();
}

/**
 * The data and host workflows remain separate from the website publisher.
 * Only model-zoo-web-pages.yml may deploy; the data workflow still builds
 * and uploads the retained catalog package without publishing a website.
 */
describe("catalog release workflow", () => {
  it("keeps data and validation workflows separate from website deployment", async () => {
    expect(await workflowNames()).toEqual([
      "host-validation.yml",
      "model-catalog-data.yml",
      "model-zoo-web-ci.yml",
      "model-zoo-web-pages.yml",
      "sample-contract.yml",
    ]);

    const workflows = await Promise.all((await workflowNames())
      .filter((name) => name !== "model-zoo-web-pages.yml")
      .map((name) => readFile(join(WORKFLOW_DIRECTORY, name), "utf8")));
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

  it("publishes the website only for release tags whose commit belongs to main", async () => {
    const source = await readFile(join(WORKFLOW_DIRECTORY, "model-zoo-web-pages.yml"), "utf8");
    const workflow = parse(source);
    expect(workflow.on).toEqual({ push: { tags: ["web-v*"] } });
    const steps = workflow.jobs.build.steps;
    const ancestry = steps.find((step: { name: string }) => step.name === "Verify release tag and branch ancestry");
    expect(ancestry.run).toContain('git merge-base --is-ancestor "$GITHUB_SHA" refs/remotes/origin/main');
    expect(steps.find((step: { uses?: string }) => step.uses?.startsWith("actions/upload-pages-artifact")).with.path)
      .toBe("model_zoo_web/dist");
    expect(source).not.toContain("origin/model_zoo_web");
    expect(source).not.toContain("origin/develop");
  });

  it("reads every platform distribution through the publisher", async () => {
    const workflow = await readFile(DATA_WORKFLOW, "utf8");

    expect(workflow).toContain('working-directory: utils/tools/catalog-publisher');
    expect(workflow).toContain("run: npm ci");
    expect(workflow).toContain("run: npm run check");
    // Benchmark records cite immutable commits, so the source check needs the
    // history rather than a shallow clone.
    expect(workflow).toContain("fetch-depth: 0");
  });

  it("uploads the catalog package with its checksum contract", async () => {
    const workflow = await readFile(DATA_WORKFLOW, "utf8");

    expect(workflow).toContain("actions/upload-artifact");
    expect(workflow).toContain("utils/tools/catalog-publisher/dist/catalog.json");
    expect(workflow).toContain("utils/tools/catalog-publisher/dist/catalog.meta.json");
  });

  it("enforces the checksum contract through the publisher, not a second implementation", async () => {
    const manifest = JSON.parse(
      await readFile(join(repositoryRoot, "utils", "tools", "catalog-publisher", "package.json"), "utf8")
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
