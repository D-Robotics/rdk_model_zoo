# H8 (bounded): datasets README restoration — 2026-09-28

Status: **implemented by Claude Code + GLM; independent review returned changes-required (DATASET-R1/R2/R3); author remediation applied 2026-09-28 — this is author rectification only, not independent acceptance. H8 is not closed by this document.**

Scope: restore usable bilingual documentation for the shared `datasets/`
resources and retarget the two evaluator dataset-navigation sentences that
previously linked acquisition only through the archived platform snapshots.
This is documentation-only: no data, label, script, manifest, platform-archive,
sample-code or other-doc file was modified, and no download, board run,
toolchain or quantization validation was executed.

## Findings addressed

| Finding (review input) | Resolution |
| --- | --- |
| No `datasets/README.md`/`README_cn.md` index | New bilingual index (`datasets/README.md`, `datasets/README_cn.md`): per-directory inventory table, consumers, manual-acquisition boundary, class-index-vs-category-ID note, provenance |
| `datasets/coco/README.md` Chinese despite English filename; no CN pair; cwd/output semantics missing | Rewritten in English; new `README_cn.md`. Documents the script as checked in: downloads train2017/val2017/annotations via `wget -c`, writes `coco_full/` **relative to the caller's cwd**, extracts with `unzip -q`, deletes the zips; script has no executable bit, so guides show `bash download_full_coco.sh`. Script itself untouched and not run |
| `datasets/dotav1` one bare URL, no CN | Rewritten EN + new CN: 15-class list documented as a fixed repository-side listing — native DOTA annotations record category **names** (eight coordinates + category + difficulty per the official page), so no official numeric ID table exists and any converted numeric-ID dataset owns its own mapping; three example tiles, **no current unified consumer**, archived X5 YOLO26 usage as provenance, and an explicit warning that `datasets/dotav1/dota_classes.names` (repository listing) and `samples/vision/ultralytics_yolo/test_data/ultralytics_dota_classes.names` (Ultralytics model order) must not be mixed — the orders coincide only at index 0 (`plane`), 14 of 15 positions differ; OBB evaluator is prediction-only, no DOTA AP scorer exists. *(Corrected 2026-09-28 per DATASET-R1: the initial version wrongly called the listing "official DOTA annotation order" with global category IDs 1–15.)* |
| PascalVOC English README nearly empty unlike CN | Both rewritten: reference-links-only contents stated plainly; UNet VOC 2012 segmentation consumer documented (21 classes, palette index = class index, 255 = ignore); acquisition boundary and inherited links preserved |
| ImageNet references only | Rewritten: dict-literal format explained (1000 entries, keys = model indices 0–999, parsed by `samples/_shared/labels.py`), zebra example (index 340), consumer map, no-downloader boundary, `.gitignore` exclusion, original official links kept. *(Narrowed 2026-09-28 per DATASET-R3: runtime display-name `--label-file` is now distinguished from the Ultralytics classification evaluator's ordered `n########` synset-ID `--label-file`; the zebra photo is described as a smoke input with per-model recorded evidence, not a guarantee.)* |
| `datasets/yoloe` README: `main.py --label-file` without cwd; names old export path | Rewritten: explicit repository-root cwd example matching the real CLI (`--target`, not `--platform`); canonical default is `samples/vision/yoloe/test_data/classes.names` (byte-identical, SHA-256 `1a6c943d…`); stale `conversion/onnx_export/export_yoloe11seg_bpu.py` corrected to the archived X5 location, with the current `conversion/export.py`/`prepare.py` flow described; PF-index≠COCO-ID mapping section added (2163=person→COCO 1, 821=chair→COCO 62) |
| Evaluator READMEs linked acquisition only via platform snapshots | Only the dataset-navigation sentences changed in `samples/vision/{ultralytics_yolo,yoloe}/evaluator/README{,_cn}.md`: unified `datasets/` guides lead, archived snapshots retained as provenance. All commands, parameters and other text untouched |

## Verification (host, static)

- Local links + explicit anchors: all 16 touched/created READMEs resolve
  (`evidence/check_links.py`, `evidence/evidence.md` §7).
- Bilingual parity: anchor sets, heading counts and code-block counts match in
  all six pairs; executable command lines are byte-identical EN/CN
  (comments translated per repo convention) (`check_parity.py`, `check_cmds2.py`).
- `git diff --check` clean on the scoped paths.
- Affected sample checkers: `samples/vision/ultralytics_yolo` → 0 violations,
  2 policy skips; `samples/vision/yoloe` → 0 violations, 1 policy skip
  (unchanged skip set, all pre-existing CLI-layer/stability-shim policies).
- Facts checked against actual files: label counts 80/15/1000/4585; COCO
  VOC-synonym names at indices 3/4/57/58/60/62; COCO category-ID gaps read from
  `eval_common.py::COCO_CATEGORY_IDS`; DOTA dual ordering captured
  byte-for-byte; YOLOE vocabulary positions 0/821/2163; byte-identity of all
  dataset resources against X5 pin `ac11571` (resources only — READMEs were
  rewritten) and of the YOLOE vocabulary against S pin `380e1a2` sources.
- Scoped diff: exactly 12 modified + 4 new README files inside the allowed
  write paths (`evidence.md` §1).

## Remaining limits (not claimed)

- No dataset was downloaded; the COCO script behavior is documented from the
  script text, not executed. Ignore coverage is narrower than the script's
  output: `.gitignore` covers the direct `datasets/coco/val2017/*` and
  `annotations/*` layout but **not** the script's `coco_full/` output, so the
  guides recommend running the script from an out-of-checkout working
  directory or adding a local exclusion first (corrected per DATASET-R2).
- No board, SDK, OE, HMCT, calibration or dataset-accuracy statement is made or
  implied; the guides deliberately contain no quantization-validation recipe.
- External, non-in-repo facts (COCO train2017 image count, ImageNet sizes,
  DOTA image counts, annotation-archive contents, official license terms) are
  stated from the official sites as referenced; they were not verified online
  in this session and carry their source links.
- `docs/Model_Zoo_Repository_Guidelines.md` §"datasets/coco/README.md"
  (structure/download/terms requirements) is satisfied by the rewritten COCO
  guide; the guidelines file itself was not edited (out of scope).
- The archived `platforms/{x5,s}/datasets/` copies are untouched and remain
  frozen provenance; their READMEs intentionally still carry the old sparse
  content.
- H8 sub-item only. H8 overall (shared responsibilities, legacy-path
  compatibility, manifests/catalog, skills provenance, upstream deltas) and
  H0–H9 remain open; independent Codex review of this package is pending.

## 2026-09-28 — DATASET-R1/R2/R3 remediation (author rectification)

Applied after the independent review
([2026-09-28-datasets-independent-review.md](2026-09-28-datasets-independent-review.md),
changes-required). Reviewer evidence and files untouched.

- **R1 (DOTA identity)** — `datasets/dotav1/README{,_cn}.md`: dropped the
  "official DOTA-v1.0 annotation category order / category IDs 1–15" claim;
  now state native annotations carry **category names** (eight coordinates +
  category + difficulty, official page supplied by the reviewer), this file is
  a fixed repository-side listing, and any converted numeric-ID dataset owns
  its own mapping that must be stated explicitly. "Mislabels every category"
  replaced with the verified counterexample: orders coincide only at index 0
  (`plane`); 14 of 15 positions differ. Root `datasets/README{,_cn}.md` dotav1
  row and class-index bullet corrected. Author evidence §4 header corrected.
- **R2 (COCO ignore coverage)** — `datasets/coco/README{,_cn}.md`: now state
  `.gitignore` covers the direct `datasets/coco/val2017/*`/`annotations/*`
  layout only, **not** the script's `coco_full/` output (verified:
  `git check-ignore datasets/coco/coco_full/train2017/example.jpg` rc=1, while
  the direct val2017 path is ignored rc=0); recommend an out-of-checkout
  working directory or a local `.git/info/exclude` entry before running in
  place. Root index intro no longer implies downloads are automatically
  excluded. Script and `.gitignore` untouched.
- **R3 (label-guide navigation and format boundaries)** —
  `datasets/PascalVOC/README.md`: language switch no longer loops to itself
  (now `./README_cn.md`). `datasets/imagenet/README{,_cn}.md`: runtime
  display-name `--label-file` (dict literal or one-per-line, shared loader)
  is separated from the Ultralytics classification evaluator's `--label-file`
  (ordered `n########` synset IDs in model-class order; the display-name file
  cannot be passed there) and from `--val-txt` ground truth; values described
  as human-readable display names, not synset IDs; zebra photo described as a
  smoke input whose expected class is checked against each model's own
  recorded evidence (e.g. MobileNetV3 S100/S600 Top-5 includes `zebra`), not
  a guarantee for arbitrary models.

Re-verification after these edits: local links/anchors, bilingual anchor and
command parity, stale-phrase scan and `git diff --check` all pass (see
`evidence/2026-09-28-datasets-docs-remediation/evidence.md` §11–§12).
Affected sample checkers are unaffected (no sample-side text changed in this
round). Author rectification only — independent rereview of the changed
semantics is still required; H8 remains open.

### 2026-09-28 — R2 follow-up correction (final, two files only)

The R2 fix's optional in-place exclusion example wrote a literal
`.git/info/exclude` path, which is wrong for managed worktrees (`.git` is a
**file** there) and for any cwd other than the repository root, while the
guide's example cwd is `datasets/coco`. `datasets/coco/README{,_cn}.md` now
resolve the path with
`"$(git rev-parse --git-path info/exclude)"` (verified: identical absolute
path returned from the repository root and from `datasets/coco` in this
worktree) and explicitly warn against the literal path. The out-of-checkout
invocation remains the primary recommendation. No exclude file, `.gitignore`
or script was modified, and no download was run; all DATASET-R1/R3 corrections
are preserved unchanged.

## Files

- New: `datasets/README.md`, `datasets/README_cn.md`,
  `datasets/coco/README_cn.md`, `datasets/dotav1/README_cn.md`
- Rewritten: `datasets/coco/README.md`, `datasets/imagenet/README.md`,
  `datasets/imagenet/README_cn.md`, `datasets/dotav1/README.md`,
  `datasets/PascalVOC/README.md`, `datasets/PascalVOC/README_cn.md`,
  `datasets/yoloe/README.md`, `datasets/yoloe/README_cn.md`
- Navigation-only edits: `samples/vision/ultralytics_yolo/evaluator/README.md`,
  `samples/vision/ultralytics_yolo/evaluator/README_cn.md`,
  `samples/vision/yoloe/evaluator/README.md`,
  `samples/vision/yoloe/evaluator/README_cn.md`
- Evidence: `evidence/2026-09-28-datasets-docs-remediation/`
  (`evidence.md`, `check_links.py`, `check_parity.py`, `check_cmds2.py`)
