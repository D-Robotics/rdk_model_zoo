// @vitest-environment node
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { repositoryRoot } from "./helpers/repository";

const REDIRECT_DIRECTORY = join(repositoryRoot, "tools", "catalog-redirect");

async function redirectPage(): Promise<string> {
  return readFile(join(REDIRECT_DIRECTORY, "index.html"), "utf8");
}

/**
 * The dashboard used to be published from this repository at `/rdk_model_zoo/`.
 * The redirect that preserves those deep links is stored here but deliberately
 * not deployed; these checks keep it correct and keep it unpublished.
 */
describe("legacy catalog redirect artifact", () => {
  it("forwards the query string and fragment unchanged", async () => {
    const page = await redirectPage();

    // Every deep link the old single-page app published lived in the query.
    expect(page).toContain("window.location.search");
    expect(page).toContain("window.location.hash");
    expect(page).toMatch(/TARGET\s*\+\s*window\.location\.search\s*\+\s*window\.location\.hash/);
    expect(page).toContain("window.location.replace(");
  });

  it("points at the documentation site without hardcoding a workstation path", async () => {
    const page = await redirectPage();

    expect(page).toContain("https://d-robotics.github.io/model_zoo_doc/models/");
    for (const forbidden of ["D:\\", "C:\\", "file://", "/Users/", "/home/"]) {
      expect(page).not.toContain(forbidden);
    }
  });

  it("still works with JavaScript disabled", async () => {
    const page = await redirectPage();

    // The inline script rewrites the fallback link, so it must exist and carry
    // a usable href before the script ever runs.
    expect(page).toMatch(/<a id="destination" href="https:\/\/d-robotics\.github\.io\/model_zoo_doc\/models\/"/);
    expect(page).toContain("<noscript>");
    expect(page).toContain('rel="canonical"');
  });

  it("says so in its own README and is never picked up by a workflow", async () => {
    const readme = await readFile(join(REDIRECT_DIRECTORY, "README.md"), "utf8");
    const workflowDirectory = join(repositoryRoot, ".github", "workflows");
    const workflows = (await readdir(workflowDirectory)).filter((name) => /\.ya?ml$/.test(name));

    expect(readme).toContain("not published");
    for (const name of workflows) {
      const workflow = await readFile(join(workflowDirectory, name), "utf8");
      expect(workflow).not.toContain("catalog-redirect");
      expect(workflow).not.toContain("upload-pages-artifact");
    }
  });
});
