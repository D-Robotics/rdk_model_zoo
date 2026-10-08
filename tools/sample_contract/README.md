English | [简体中文](README_cn.md)

# sample-contract checker

Static contract checker for the unified samples layout. It executes the
machine-decidable rules of `docs/sample-standards/readme-contract.md` and
`docs/sample-standards/inference-contract.md`; everything not decidable
statically goes to semantic review — a skipped check is reported as a
skip and never counts as a pass.

## What it checks

| Rule | Scope |
| --- | --- |
| `R-README-PAIR` | every existing level ships both `README.md` and `README_cn.md` |
| `R-README-SECTIONS` | fixed anchor IDs (derived live from the templates) are present, unique, and in template order |
| `R-README-LINKS` | relative links/images resolve to existing files; intra-document fragments resolve to explicit anchors; external URLs are out of scope (no network) |
| `R-CLI-DEFAULTS` | runtime/python parameter tables match the real `build_parser` defaults, per language |
| `R-I18N-PARAMS` | the en/zh parameter tables agree with each other (option set and defaults) |
| `R-STAGE-PURITY` | AST scan: stage functions — canonical `preprocess`/`infer`/`postprocess`/`predict` plus the legacy `pre_process`/`forward`/`post_process` spellings, with the matching `preprocess_*`/`infer_*`/`postprocess_*`/`forward_*`/`pre_process_*`/`post_process_*`/`run_*` prefixes — contain no download, file-write/save, subprocess, or destructive calls. `main.py` and `legacy.py` are policy-skipped with a recorded reason; module-level helpers in `cli.py`/`yolo_cli.py` (e.g. a `run_prepare` that downloads on explicit request) are the CLI application boundary and are recorded as skips, while stage-named methods inside those files stay checked |

## Usage

```bash
# one sample
python3 tools/sample_contract/check.py --sample samples/vision/resnet

# CI scope: every sample row listed in the progress map
python3 tools/sample_contract/check.py --scope migration --report out.json

# never execute sample code (CLI-default checks then record skips)
python3 tools/sample_contract/check.py --sample ... --parser-mode static

# checker's own tests (fixtures first)
python3 -m unittest discover -s tools/sample_contract/tests -v
```

Exit codes: `0` no violations, `1` violations found, `2` usage/config error.
Skips (missing `main.py`, import failure, static mode, policy-skipped files,
CLI-boundary module-level helpers in `cli.py`/`yolo_cli.py`) are always
printed and included in the JSON report.

### Stage-name scope

The sample architecture uses `preprocess`/`infer`/`postprocess` as the
primary stage spellings with `pre_process`/`forward`/`post_process` kept as
thin compatibility aliases of the same bodies, so the purity scan covers both
spellings, and a download or file write belongs in neither. The CLI-boundary
exemption covers only module-level functions in the sample-local
`cli.py`/`yolo_cli.py` files with stage-shaped names, always recorded as a
named skip; class methods, other files and whole directories stay checked.

## Canonical default-value forms

The README `Default` column and the parser are compared after
canonicalization: `None` → `null` (README may write `null`/`none`), booleans
→ `true`/`false`, lists → JSON form (`[0]`, `[0, 1]`), scalars → literal,
absolute paths under the repository → repo-relative posix form. Backticks
and surrounding quotes in README cells are stripped.

## Check scope

`--scope migration` reads the progress region of
`docs/releases/unified-migration/x5-s-migration-map.md` and checks every
sample whose **Refactor** column is `in-progress` or `done` (parenthetical
annotations are ignored). A selected row without a resolvable
`samples/<domain>/<name>` directory raises an `R-SCOPE` violation, keeping
the map and the sample tree in sync.

## Exemptions

`--exemptions file.json` accepts reviewed exceptions of the form
`{"rule", "path", "line", "reason"}`; `reason` is mandatory. An optional
`"message"` field pins the entry to one exact finding message — use it for
rules that report several findings at one line (`R-README-SECTIONS` reports
every missing anchor at line 0). Each entry names one finding location;
directory-wide entries are rejected. An exemption with no matching finding
fails with `R-EXEMPTION`, so remove the entry in the same change that
removes the finding. The schema and matching behavior are covered by the
fixture tests under `tests/`.

## Boundaries

The checker reads files and — in import mode — imports trusted repository
`main.py` modules to call `build_parser`. It never downloads, never loads
a board SDK, and never executes README code blocks. Prose quality,
duplicate `predict` logic, board behavior, and conversion correctness are
covered by semantic review and board smoke testing, not by this checker.
