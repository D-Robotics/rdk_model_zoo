# S/X3 Baseline Releases Implementation Plan

> **Execution:** Use isolated worktrees and subagents for the independent S and X3 release-preparation commits. The primary agent owns the shared workflow guard, final validation, tags, pushes, and GitHub Releases.

**Goal:** Publish `s-v1.0.0` and `x3-v1.0.0` as formal, immutable GitHub Releases with model and benchmark manifests derived from existing repository evidence.

**Architecture:** The `rdk_x5` line remains the release-policy and catalog controller. Each platform branch carries a self-contained release snapshot and schemas. The X5 validation code is used from the controller worktree to validate each platform worktree without copying the web application into legacy branches.

**Constraints:** No RDK board tests. No invented benchmark conditions, platform support, URLs, variants, or checksums. Exclude YOLOE from release data without deleting its source. Treat X3 as legacy. Attach both YAML manifests to each GitHub Release.

---

## Task 1: Protect the existing Pages deployment from non-X5 releases

**Files:**

- Modify: `.github/workflows/model-catalog-pages.yml`
- Create: `site/tests/pages-workflow.test.ts`
- Modify: `docs/RELEASE.md`
- Modify: `docs/RELEASE_cn.md`

1. Add a failing test that reads the Pages workflow and requires the build job to run for manual dispatches and `x5-v*` Release tags only.
2. Add the job-level release-tag guard. Let the deploy job remain dependent on the guarded build job.
3. Document that S/X3 Releases publish manifests but do not automatically replace the current X5-only dashboard.
4. Run `npm test`, `npm run check`, and `git diff --check`.
5. Commit as `fix(release): scope catalog deployment to X5 tags` and push the reviewed commit to `rdk_x5` before publishing either Release.

## Task 2: Prepare the RDK S v1.0.0 release commit

**Worktree:** `.worktrees/release-s-v1.0.0`, created from the current `origin/rdk_s` tip.

**Files:**

- Create: `VERSION`
- Create or update: `CHANGELOG.md`
- Create: `release/models.yaml`
- Create: `release/benchmarks.yaml`
- Create: `release/schemas/models.schema.json`
- Create: `release/schemas/benchmarks.schema.json`
- Create: `release/README.md`
- Create: `docs/releases/s-v1.0.0.md`

1. Inventory local samples, download scripts, explicit assets, SoC variants, and available checksums from the S branch.
2. Exclude every YOLOE model and benchmark record. Do not modify the existing sample source.
3. Represent ACT/Pi0 only if the gitlink evidence fits the manifest contract; otherwise disclose their external-submodule status in the release notes and omit them from model totals.
4. Transcribe performance and accuracy records from exact tagged sections. Keep incomplete conditions absent and add plain limitations to the release notes.
5. Set all release identity fields to platform `s`, version `1.0.0`, tag/source ref `s-v1.0.0`, and branch `rdk_s`.
6. Recompute summary counts from parsed YAML. Validate schema, references, source files/headings, YOLOE exclusion, and `git diff --check` from the controller worktree.
7. Commit as `chore(release): prepare S v1.0.0 baseline`.

## Task 3: Prepare the RDK X3 v1.0.0 release commit

**Worktree:** `.worktrees/release-x3-v1.0.0`, created from the current `origin/rdk_x3` tip.

**Files:**

- Create: `VERSION`
- Create or update: `CHANGELOG.md`
- Create: `release/models.yaml`
- Create: `release/benchmarks.yaml`
- Create: `release/schemas/models.schema.json`
- Create: `release/schemas/benchmarks.schema.json`
- Create: `release/README.md`
- Create: `docs/releases/x3-v1.0.0.md`

1. Inventory the 15 logical model families and their documented concrete external assets.
2. Preserve unresolved URLs and upstream variant ambiguity; do not expand a family into variants without branch evidence.
3. Mark the release and its sample inventory as a legacy, non-normalized baseline in metadata and notes.
4. Transcribe X3-only performance and accuracy records. Do not import X5, S, or mixed-platform rows.
5. Set all release identity fields to platform `x3`, version `1.0.0`, tag/source ref `x3-v1.0.0`, and branch `rdk_x3`.
6. Recompute summary counts from parsed YAML. Validate schema, references, source files/headings, and `git diff --check` from the controller worktree.
7. Commit as `chore(release): prepare X3 v1.0.0 baseline`.

## Task 4: Review both release commits

1. Use a fresh reviewer for each branch to compare the manifest data with the tagged-source candidates and the approved design.
2. Resolve all correctness findings in the owning worktree and rerun validation.
3. Confirm both tags are absent locally and remotely, both worktrees are clean, and each release commit is based on the expected remote branch tip.
4. Perform a final cross-release review: release tag, branch, version, model totals, asset totals, benchmark totals, attachments, limitations, and absence of YOLOE.

## Task 5: Publish branches, annotated tags, and GitHub Releases

1. Push the S release commit to `origin/rdk_s` and the X3 release commit to `origin/rdk_x3`.
2. Create annotated tags `s-v1.0.0` and `x3-v1.0.0` at the reviewed commits, then push them without force.
3. Create public GitHub Releases with the tagged bilingual notes and attach `release/models.yaml` as `models.yaml` and `release/benchmarks.yaml` as `benchmarks.yaml`.
4. Verify through GitHub that each Release is public, both attachments are downloadable, each tag resolves to the intended commit, and the platform branch contains the same commit.
5. Check the Actions result to confirm S/X3 release events skip the X5-only Pages build instead of failing it.

## Task 6: Record publication results and resume catalog work

1. Record the S and X3 release URLs, immutable commit SHAs, inventory totals, validation scope, and material limitations.
2. Rebase the multi-platform catalog branch on the updated `rdk_x5` branch.
3. Start the approved exact-model-card redesign using the three immutable release tags as data sources.
