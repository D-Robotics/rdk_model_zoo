# Phase 0.5 Q3 review — automated checker with negative fixtures (2026-09-20)

Scope: plan Q3 — `tools/sample_contract/check.py` + tests + fixtures +
`.github/workflows/sample-contract.yml`.  Evidence:
[evidence/2026-09-20-phase05-q3-checker.json](evidence/2026-09-20-phase05-q3-checker.json)
(full pilot run: [evidence/2026-09-20-phase05-q3-pilot-checker-report.json](evidence/2026-09-20-phase05-q3-pilot-checker-report.json)).

## Delivered

| Artifact | Content |
| --- | --- |
| `tools/sample_contract/check.py` | 7 rules: `R-README-PAIR`, `R-README-SECTIONS`, `R-README-LINKS`, `R-CLI-DEFAULTS`, `R-I18N-PARAMS`, `R-STAGE-PURITY`, plus `R-EXEMPTION`/`R-SCOPE` bookkeeping.  Stdlib-only; reads source/docs; the only code execution is `build_parser()` import from trusted repo code (`--parser-mode static` executes nothing).  Exit 0/1/2; JSON `--report` for evidence; skips are always visible and never count as passes. |
| Section source of truth | Required anchor IDs are derived **live** from `docs/sample-standards/templates/*.md` at every run (en/zh parity enforced, duplicates fatal).  No second rule source; template changes flow into fixtures/tests automatically. |
| CLI-defaults comparison | Defaults come from the actual parser (`build_parser()` + subparser walk).  Canonical forms: `None`→`null`, bool→`true/false`, lists→JSON, repo-absolute paths→repo-relative; documented in the module docstring and `tools/sample_contract/README.md`. |
| Stage purity (AST) | `pre_process`/`forward`/`post_process`/`predict`/`forward_*`/`run_*` bodies scanned for download/network, file write/save (incl. write-mode `open`), subprocess, destructive-fs, and eval/exec calls.  `main.py`/`legacy.py` are policy-skipped with recorded reasons — visible skips, not silent exemptions. |
| Fixtures (constructed first) | `good_sample` (fully compliant) + 7 negative fixtures: `bad_sections` (missing/duplicate/out-of-order anchors), `bad_links` (dead local link, dead image, dead fragment; external URL out of scope), `bad_cli_drift` (default drift, undocumented option, phantom option; zh clean), `bad_stage_purity` (4 boundary calls incl. write-mode `open`; non-stage helper deliberately clean), `bad_i18n_params` (option-set + default mismatch), `bad_pair` (missing `README_cn.md`), `bad_import_runtime` (import failure → skip). |
| `tests/test_check.py` | 23 tests asserting exact rule IDs and fixture line numbers, static-mode skip semantics, exemption behavior (match suppresses; unused fails; missing reason exits 2), live migration-map scope resolution, and canonicalization.  Suite: **23 OK**. |
| `.github/workflows/sample-contract.yml` | unittest discovery + `--scope migration --parser-mode import --report` with artifact upload; path-filtered to samples/standards/checker/map. |
| Migration scope semantics | Parses the current-round progress region; **Refactor ∈ {in-progress, done}** (parentheticals stripped); historical P0 S/F/H never read.  Unresolvable/ambiguous rows raise `R-SCOPE` violations — ledger hygiene enforced by CI. |

## Verification

- `python3 -m unittest discover -s tools/sample_contract/tests -v` — 23 OK
  (command per plan).
- Checker on the live migration scope (3 pilots): **252 violations, exit 1** —
  all `R-README-SECTIONS` (84 per sample; the pilots' READMEs predate the Q1
  anchor mechanism).  This is the expected pre-Q4 state, now machine-recorded
  instead of assumed.  Notably `R-STAGE-PURITY` is **zero** on all three
  pilots' real task modules — the Q2 boundary work holds at the AST level.
- `R-CLI-DEFAULTS` currently records skips ("no parameters section") on the
  pilots; the rule activates when Q4 adds the anchor + tables.  No rule was
  weakened anywhere to make output green.

## Explicit limits / not-run

- GitHub Actions has not executed (no push from this worktree; authorization
  boundary).  The workflow is committed with failing-on-violation semantics.
- CI will stay red on the pilot scope until Q4 reworks resnet/paddle_ocr
  READMEs (ultralytics_yolo per the baseline exit note defers to B9 — its
  Docs column already records "pending（Q1–Q5 合规随 B9 收编验收）").
  Red-until-compliant is the honest gate the plan asks for
  ("不得通过减少文档内容…放宽规则来消除检查错误").
- AST purity is name/shape-based: it proves the absence of the denied call
  categories in stage bodies, not full semantic equivalence — prose quality,
  duplicate predict logic, and board behavior remain Q4/review dimensions.
- The checker does not parse prose file references (only markdown links);
  `directory`/`local-paths` factual agreement stays with semantic review.

## Verdict

Q3 complete: fixtures were constructed before the rules they pin, the suite
asserts both polarities with rule ID + path:line, scope resolution reads the
new progress region (not historical P0), and the checker on real pilots
produces the honest pre-Q4 baseline.  Proceed to Q4 (reference-sample
rework and dual-perspective acceptance).
