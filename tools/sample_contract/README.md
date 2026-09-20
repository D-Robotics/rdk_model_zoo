# sample-contract checker (Phase 0.5, Q3)

Static contract checker for the unified samples layout.  It executes the
machine-decidable rules of `docs/sample-standards/readme-contract.md` and
`docs/sample-standards/inference-contract.md`; everything not decidable
statically goes to semantic review (Q4) — a skipped check is reported as a
skip and never counts as a pass.

## What it checks

| Rule | Scope |
| --- | --- |
| `R-README-PAIR` | every existing level ships both `README.md` and `README_cn.md` |
| `R-README-SECTIONS` | fixed anchor IDs (derived live from the Q1 templates) are present, unique, and in template order |
| `R-README-LINKS` | relative links/images resolve to existing files; intra-document fragments resolve to explicit anchors; external URLs are out of scope (no network) |
| `R-CLI-DEFAULTS` | runtime/python parameter tables match the real `build_parser()` defaults, per language |
| `R-I18N-PARAMS` | the en/zh parameter tables agree with each other (option set and defaults) |
| `R-STAGE-PURITY` | AST scan: `pre_process`/`forward`/`post_process`/`predict`/`forward_*`/`run_*` functions contain no download, file-write/save, subprocess, or destructive calls (`main.py` and `legacy.py` are policy-skipped with a recorded reason, not silently ignored) |

## Usage

```bash
# one sample
python3 tools/sample_contract/check.py --sample samples/vision/resnet

# CI scope: resolved from the migration progress region
python3 tools/sample_contract/check.py --scope migration --report out.json

# never execute sample code (CLI-default checks then record skips)
python3 tools/sample_contract/check.py --sample ... --parser-mode static

# checker's own tests (fixtures first)
python3 -m unittest discover -s tools/sample_contract/tests -v
```

Exit codes: `0` no violations, `1` violations found, `2` usage/config error.
Skips (missing `main.py`, import failure, static mode, policy-skipped files)
are always printed and included in the JSON report.

## Canonical default-value forms

The README `Default` column and the parser are compared after
canonicalization: `None` → `null` (README may write `null`/`none`), booleans
→ `true`/`false`, lists → JSON form (`[0]`, `[0, 1]`), scalars → literal,
absolute paths under the repository → repo-relative posix form.  Backticks
and surrounding quotes in README cells are stripped.

## Migration scope semantics

`--scope migration` parses the current-round progress region of
`docs/releases/unified-migration/x5-s-migration-map.md` and includes every
row whose **Refactor** column is `in-progress` or `done` (parenthetical
annotations are ignored).  The historical P0 `S/F/H` columns are never read
for scope decisions.  Rows that qualify but have no resolvable
`samples/<domain>/<name>` directory raise an `R-SCOPE` violation, so ledger
hygiene is enforced by CI.

## Exemptions

`--exemptions file.json` accepts reviewed exceptions of the form
`{"rule", "path", "line", "reason"}`; `reason` is mandatory.  An exemption
that matches no finding is itself a violation (`R-EXEMPTION`), so stale
entries fail CI instead of rotting.  Broad directory exemptions are not
supported by design (plan Q3); each entry must name one finding location.

## Boundaries

The checker reads files and — in import mode — imports trusted repository
`main.py` modules to call `build_parser()`.  It never downloads, never loads
a board SDK, and never executes README code blocks.  Prose quality,
duplicate `predict` logic, board behavior, and conversion correctness stay
with semantic review (Q4) and board smoke.
