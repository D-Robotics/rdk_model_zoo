# Phase 0 review — preparation and acceptance (2026-09-20)

Scope: plan Phase 0 items 1–5 plus the review-supplement conditions (kickoff
semantics fix, progress region, session discovery/trigger evidence, CLAUDE.md
minimal correction, `inspect_repo.py` boundary record). Phase 0 covers
preparation only — it does not assert baseline, migration, or board readiness.

## 1. Skills installation and verification

**Install.** Source: `rdk_x5` @ `ac115717197920355fc390bb04299b20e6436864`.
Installed complete directories (not bare `SKILL.md`) for all 7 skills to
`~/.claude/skills/`: `rdk-model-zoo`, `rdk-model-zoo-repo`,
`rdk-model-zoo-integrate`, `rdk-model-zoo-develop`, `rdk-model-zoo-validate`,
`rdk-model-zoo-review`, `rdk-model-zoo-release`. No pre-existing copies of
these names were present before install; user-owned `rdk-docs` was left
untouched.

**Per-file comparison** (performed 2026-09-20 after the plan tightened this
requirement; the install itself preceded the requirement, so the comparison is
retrospective, not contemporaneous):

```bash
# cwd: repo root (develop)
tmp=$(mktemp -d) && git archive ac11571 skills | tar -x -C "$tmp"
diff -r "$tmp/skills/<name>" ~/.claude/skills/<name>   # for each of the 7
```

Result: all 7 directories **IDENTICAL** (zero diff lines) to
`rdk_x5@ac11571`. Record: this comparison is recorded here and was not
captured as a separate evidence JSON; it is reproducible with the command
above.

**File/tool usability.**
`python3 ~/.claude/skills/rdk-model-zoo-repo/scripts/inspect_repo.py --repo /Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo`
→ exit code **0**, `ok: true`, `head: dc5815d…`, `branch: develop`,
`dirty: true` (untracked `CLAUDE.md` only), `file_count: 2669`. Run from the
installed location, not a repo-relative path (develop has no root `skills/`).

**Session discovery and trigger.** Executing session: Claude Code session
`4afec57f-4307-4180-8324-0bf16e48a8c9` (continued across compaction; the same
session performs the migration). On 2026-09-20 the session invoked the Skill
tool for `rdk-model-zoo-repo` with a read-only repo-layout request; the
harness resolved it to base directory `/Users/Max/.claude/skills/rdk-model-zoo-repo`
(installed copy) and loaded its instructions, and the session produced the
skill-compliant read-only context report (recorded in the session transcript).
Discovery of the remaining 6 skills is evidenced by their presence and
identical content but **routing trigger for those 6: not-run** — each will be
exercised when its workflow first executes (develop/review/validate in
Phase 0.5–1; integrate/release later). File existence and script success are
not claimed as session verification for them.

**Known boundary of `inspect_repo.py` (current installed version).** On
develop it reports `branch_role: "unknown"` and `manifest_candidates: []`.
Manual check contradicts an "no manifests" reading:
`platforms/x5/docs/release/` and `platforms/s/docs/release/` both contain
`models.yaml`, `benchmarks.yaml`, `README.md`, `schemas/`. The script does not
yet understand the unified
layout; it must not be cited as evidence that manifests are absent. Unified-
layout support is scheduled in Q5, not claimed done here.

## 2. develop switch record

Worktree on `develop`, HEAD `dc5815ddd6b71cad56b669b8a010d86e01223aa6`
(kickoff commit). Status: clean except untracked `CLAUDE.md` (preserved,
then corrected in §4). No branch switches or resets performed afterwards.

## 3. Record conventions and progress region

- `2026-09-20-batch-migration-kickoff.md`: the erroneous `S → F (mapping
  verified) → closed` state-machine reading was replaced with the inventory
  semantics (S/F/H as defined in the map), a link to the new progress region,
  and closure conditions consistent with the plan (`Closed=yes` only with all
  required dimensions passed and evidence; host tests never close a row).
- `x5-s-migration-map.md`: appended the current-round progress region —
  fixed columns (Mapping/Refactor/Docs/Host/Board/Review/Closed/Evidence),
  status vocabularies, evidence-binding and not-applicable rules, and the Q3
  inclusion rule (Refactor in-progress/done). 3 pilot rows (with per-target
  board nuance from the 2026-09-17 integration review: YOLO covers X5 dual
  boards + S100 + applicable S100P; ResNet S100P has no approved asset, not
  verified; OCR S100P/S600 not-run) + 52 pending rows covering every B1–B11
  sample. Historical P0 tables untouched.
- Corrigenda recorded against the kickoff batch table:
  `depth_anything_v2`, `lanenet`, `pointnet` are S-only sources; B2's
  "efficientformer(v2)" is two separate source samples
  (`efficientformer`, `efficientformerv2`); the S-side YOLOv13 directory is
  `yolov13_imoonlab`.

## 4. CLAUDE.md minimal correction (pre-0.5, not deferred to closure)

Six targeted edits, preserving still-true user content: develop described as
the sample-centric integration line with per-sample `--target` selection and
no silent fallback; rdk_x5/rdk_s named as delivery lines during the window;
X3 historical; `docs/catalog`/`docs/RELEASE.md`/root `skills/`/Guidelines
marked as not-on-develop with their real locations (installed copies at
`~/.claude/skills/`, ADR-0001/ADR-0006 pointers); language coverage declared
per sample instead of unconditional dual-runtime; `get_soc_name()` default-
`s100` behavior attributed to delivery branches only; manifest locations
during migration (`platforms/{x5,s}/docs/release/` → A4 target) documented;
`resnet` named as the migration reference sample.

## 5. Not-run / limitations (explicit)

- Board runs: none in Phase 0 (out of scope); pilot board statuses above are
  citations of the 2026-09-17 review, not new runs.
- Session routing trigger for 6 of 7 skills: **not-run** (see §1).
- Source-drift re-check of rdk_x5/rdk_s tips vs baselines: not performed in
  Phase 0; first re-check happens at B1 preflight per the batch procedure.
- The per-file skills comparison is reproducible but was executed
  retrospectively (the tightened plan postdates the install).

## Verdict

All Phase 0 items and review-supplement conditions are closed within Phase 0's
declared scope; the not-run items above are recorded rather than hidden. No
claim is made about baseline (Q1–Q5), migration batches, or board readiness.
**Entry to Phase 0.5 is allowed.**
