#!/usr/bin/env python3
"""Static claim check for the H8 catalog-docs README correction.

Verifies every factual statement of the rewritten tools/catalog-publisher/README.md
against the actual sources (sources.json, CI workflow, src/, VERSION files,
generated dist metadata). Pure static comparison - no npm, no network, no build.
Run from the repository root:
    python3 docs/releases/unified-migration/evidence/2026-09-28-catalog-docs-remediation/check_claims.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path.cwd()
PUB = REPO / "tools" / "catalog-publisher"
FAILURES = []
PASSES = []


def check(name, condition, detail=""):
    (PASSES if condition else FAILURES).append(f"{name}{': ' + detail if detail else ''}")


readme = (PUB / "README.md").read_text()
before = (Path(__file__).parent / "readme-before.md").read_text()
sources = json.loads((PUB / "sources.json").read_text())

# --- 1. README no longer claims main/platforms reads -------------------------
check("old false paragraph removed",
      "All current platform sources are read from `main`" not in readme
      and "New models, fixes and measurements are maintained there" not in readme)
check("README states worktree mode", "All three currently use `worktree` mode" in readme)

# --- 2. sources.json: mode, paths, manifests, versions, link refs ------------
for platform, manifest_root, version_file, link_ref, link_prefix, path in [
    ("x5", "docs/release/x5", "docs/release/x5/VERSION", "develop", "", "."),
    ("s", "docs/release/s", "docs/release/s/VERSION", "develop", "", "."),
    ("x3", "release", "platforms/x3/VERSION", "main", "platforms/x3", "platforms/x3"),
]:
    entry = sources["sources"][platform]
    check(f"{platform}: mode worktree", entry["mode"] == "worktree", entry["mode"])
    check(f"{platform}: path", entry["path"] == path, entry.get("path"))
    check(f"{platform}: manifest_root", entry["manifest_root"] == manifest_root, entry["manifest_root"])
    check(f"{platform}: version_file", entry.get("version_file", "VERSION") == version_file
          or (entry.get("version_file") is None and (REPO / version_file).name == "VERSION"))
    check(f"{platform}: link_ref", entry["link_ref"] == link_ref, entry["link_ref"])
    check(f"{platform}: link_prefix", entry["link_prefix"] == link_prefix, repr(entry["link_prefix"]))
    # files the build actually reads exist in the worktree
    check(f"{platform}: models.yaml exists", (REPO / path / manifest_root / "models.yaml").is_file())
    check(f"{platform}: benchmarks.yaml exists", (REPO / path / manifest_root / "benchmarks.yaml").is_file())
    check(f"{platform}: VERSION exists", (REPO / version_file).is_file())
    check(f"{platform}: README table row cites link",
          f"blob/{link_ref}/..." in readme or f"blob/{link_ref}/" in readme)
    # README's claim: build rejects VERSION/manifest disagreement (src check below)
    # current tree actually agrees:
    version = (REPO / version_file).read_text().strip()
    import_text = (REPO / path / manifest_root / "models.yaml").read_text()
    manifest_version = re.search(r"^\s*version:\s*[\"']?([^\"'\s]+)", import_text, re.M)
    check(f"{platform}: VERSION {version} agrees with manifest version",
          manifest_version is not None and manifest_version.group(1) == version)

# --- 3. archived snapshots exist but are never referenced by sources.json ----
check("archived x5 snapshot exists", (REPO / "platforms/x5/docs/release/models.yaml").is_file())
check("archived s snapshot exists", (REPO / "platforms/s/docs/release/models.yaml").is_file())
blob = json.dumps(sources)
check("sources.json never reads platforms/x5 or platforms/s snapshots",
      "platforms/x5" not in blob and "platforms/s" not in blob)
check("README warns archived snapshots are not read",
      "`sources.json` never reads them" in readme and "archived frozen snapshots" in readme)
check("README names docs/release/{x5,s} as maintenance location",
      "New models, fixes and measurements for X5 and S are maintained in the unified manifests `docs/release/{x5,s}`" in readme)

# --- 4. worktree read vs generated link (src mechanics) ----------------------
src = (PUB / "src" / "sources.ts").read_text()
check("sources.ts documents link_ref as link-only",
      "Git ref used for repository source links (`blob/<ref>/...`)" in src)
check("sources.ts worktree read uses filesystem readFile",
      re.search(r'kind === "worktree"\)\s*\{\s*return readFile\(', src, re.S) is not None)
variants = (PUB / "src" / "catalog" / "variants.ts").read_text()
check("variants.ts emits source_ref from sourceRef option",
      "source_ref: options.sourceRef ?? releaseTag" in variants
      and "source_path_prefix: options.sourcePathPrefix" in variants)
mp = (PUB / "src" / "pipeline" / "multiplatform-catalog.ts").read_text()
check("provenance ref comes from linkRef", "ref: source.linkRef" in mp)
check("build rejects VERSION/manifest disagreement",
      "and manifest release version" in mp and "disagree" in mp)
check("README explains worktree-read vs generated-link split",
      "`link_ref`/`link_prefix` do not select what is read" in readme
      and "do not prove that `develop` already contains those candidates" in readme)
check("README cites manifest_sha256 as what was read",
      "manifest_sha256" in readme and "manifest_sha256" in mp)

# --- 5. dist metadata agrees (worktree refs are link labels) -----------------
meta_path = PUB / "dist" / "catalog.meta.json"
if meta_path.is_file():
    meta = json.loads(meta_path.read_text())
    for platform, ref in [("x5", "develop"), ("s", "develop"), ("x3", "main")]:
        prov = meta["sources"][platform]
        check(f"dist provenance {platform} ref={ref}", prov["ref"] == ref, prov["ref"])
        check(f"dist provenance {platform} kind=worktree", prov["kind"] == "worktree")
        check(f"dist provenance {platform} has manifest_sha256",
              len(prov.get("manifest_sha256", "")) == 64)
else:
    check("dist metadata present", False, "dist/catalog.meta.json missing (pre-build state)")

# --- 6. CI triggers ----------------------------------------------------------
wf = (REPO / ".github" / "workflows" / "model-catalog-data.yml").read_text()
check("workflow: pull_request trigger", re.search(r"^  pull_request:", wf, re.M) is not None)
push = re.search(r"^  push:\n    branches: \[main\]", wf, re.M)
check("workflow: push restricted to main", push is not None)
check("workflow: workflow_dispatch trigger", re.search(r"^  workflow_dispatch:", wf, re.M) is not None)
check("workflow: upload only on non-PR runs", "if: github.event_name != 'pull_request'" in wf)
check("workflow: uploads exactly the two dist files",
      "tools/catalog-publisher/dist/catalog.json" in wf
      and "tools/catalog-publisher/dist/catalog.meta.json" in wf)
check("README states the three triggers and non-PR upload",
      "`pull_request` (path-filtered), pushes to `main` only (path-filtered), and manual `workflow_dispatch`" in readme
      and "uploads the two artifact files only for non-PR runs" in readme)
check("README denies workbranch-push publication",
      "Pushes of integration work branches trigger no workflow" in readme)
check("README keeps separate-publication boundary",
      "Enabling that workflow remotely and publishing the website are separate actions" in readme)

# --- 7. preserved sections ----------------------------------------------------
def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0] if start in text and end in text else None

for label, start, end in [
    ("Node requirement", "Use Node.js 22.12", "```sh"),
    ("npm script block", "cd tools/catalog-publisher", "The check validates"),
    ("data identity + pins", "The catalog format version is independent", "## Import into"),
    ("docs import block", "After the source check succeeds", "Commit the imported snapshot"),
    ("no-synthesis rule", "Static manifest checks do not replace", "no values are synthesized."),
]:
    check(f"preserved byte-identical: {label}", section(before, start, end) == section(readme, start, end))

check("catalog-redirect pointer kept",
      "The old website redirect is in `../catalog-redirect`" in readme
      and (REPO / "tools" / "catalog-redirect").is_dir())
check("engines matches Node claim",
      json.loads((PUB / "package.json").read_text())["engines"]["node"] == ">=22.12 <23")
check("README no longer says 'changing the main branch'",
      "changing the main branch" not in readme)

# --- 8. relative links in README resolve --------------------------------------
for target in re.findall(r"\]\(([^)#h][^)]*)\)|`(\.\./[^`]+)`", readme):
    raw = target[0] or target[1]
    resolved = (PUB / raw).resolve()
    check(f"README link resolves: {raw}", resolved.exists())

# --- 9. scope: only README.md changed inside tools/catalog-publisher ----------
status = Path(".git").is_dir()
git_run = __import__("subprocess").run(
    ["git", "status", "--porcelain", "tools/catalog-publisher"],
    capture_output=True, text=True, cwd=REPO)
if git_run.returncode != 0:
    FAILURES.append(f"git status failed (rc={git_run.returncode}): {git_run.stderr.strip()}")
    git_status = []
else:
    # rstrip only: porcelain's leading status column (" M" = unstaged modify)
    # must survive, or the M is misread as a staged change.
    git_status = git_run.stdout.rstrip("\n").splitlines()
modified = [line[3:] for line in git_status if line.startswith(" M")]
check("only tools/catalog-publisher/README.md modified in module",
      modified == ["tools/catalog-publisher/README.md"], str(modified))
check("no other tracked change in module",
      all(not line.startswith(("M ", "A ", "D ", "R ")) for line in git_status))

# --- report -------------------------------------------------------------------
for item in PASSES:
    print(f"ok   {item}")
for item in FAILURES:
    print(f"FAIL {item}")
print(f"\n{len(PASSES)} passed, {len(FAILURES)} failed")
sys.exit(1 if FAILURES else 0)
