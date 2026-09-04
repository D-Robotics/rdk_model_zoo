// @vitest-environment node
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

const workflowPath = fileURLToPath(new URL("../../.github/workflows/model-catalog-pages.yml", import.meta.url));

interface PagesWorkflow {
  on: {
    release: { types: string[] };
    workflow_dispatch: { inputs: Record<string, unknown> };
  };
  jobs: {
    build: { if: string };
    deploy: { needs: string };
  };
}

describe("model catalog Pages workflow", () => {
  it("runs release deployments only for X5 catalog tags", async () => {
    const workflow = parse(await readFile(workflowPath, "utf8")) as PagesWorkflow;

    expect(workflow.on.release.types).toEqual(["published"]);
    expect(workflow.on.workflow_dispatch.inputs).toHaveProperty("catalog_ref");
    expect(workflow.jobs.build.if).toBe(
      "github.event_name == 'workflow_dispatch' || (github.event_name == 'release' && startsWith(github.event.release.tag_name, 'x5-v'))"
    );
    expect(workflow.jobs.deploy.needs).toBe("build");
  });
});
