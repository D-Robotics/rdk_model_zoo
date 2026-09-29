# Review status documentation remediation — author record

Implementer: Claude Code + GLM (documentation package; independent review remains
with Codex). Base `4d16c4a7`. This package closes
[YOLOE-DOC-R1](2026-09-28-yoloe-independent-review.md) (stale C++ status in the
conversion/evaluator guides) and the stale "Independent review remains pending"
statement that
[the B8 mobile planning independent review](2026-09-28-b8-mobile-planning-independent-review.md)
asked to update in the next coordinated author documentation pass.

## Scope discipline

Only the eight README files listed below plus this report and
[the remediation evidence directory](evidence/2026-09-28-review-status-docs-remediation/)
were added or changed. Product code, sample root/runtime READMEs outside the
list, other samples, reviewer reports and their evidence, plans and ledgers
were not touched. Pre-existing untracked files from concurrent packages
(`samples/llm/gemma4-e2b/...`, `docs/releases/unified-migration/*readme-depth*`,
and their evidence directories) present in the worktree are not part of this
package and were left unchanged.

No quantization recipe, command, parameter, threshold, table value or
historical result changed: the eight diffs touch prose status sentences only,
and all fenced command blocks are byte-identical to HEAD
([fenced-blocks-invariance.txt](evidence/2026-09-28-review-status-docs-remediation/fenced-blocks-invariance.txt)
is empty because the block diff is empty; no added/removed line in
[readmes.diff](evidence/2026-09-28-review-status-docs-remediation/readmes.diff)
contains a command or fence marker).

## Changes (eight files, one sentence each)

YOLOE — "C++ migration remains unfinished / C++ 迁移仍未完成" and the
equivalent closing "host checks do not close independent migration acceptance"
wording replaced by: the C++ runtime is implemented and its host runtime
composition/entrypoint scope was independently reviewed, while real SDK
compilation and board inference remain not-run; links point to
[the YOLOE independent review](2026-09-28-yoloe-independent-review.md).
Genuine float-S-publication, OE-compile, dataset and board gaps are retained;
no B9 completion is claimed.

- `samples/vision/yoloe/conversion/README.md:149` (known-gaps paragraph,
  closing sentence).
- `samples/vision/yoloe/conversion/README_cn.md:149` (same paragraph).
- `samples/vision/yoloe/evaluator/README.md:156` (boundaries closing
  sentence: "Native C++ migration and whole-branch independent acceptance
  remain separate work" → implemented/host-reviewed, real SDK/board not-run,
  whole-branch acceptance open).
- `samples/vision/yoloe/evaluator/README_cn.md:156` (same sentence).
- `samples/vision/yoloe/runtime/cpp/README.md:350` (verification-boundaries
  closing paragraph: independent review of the host runtime scope added; the
  remaining not-run list and the review's own limits — conversion/evaluator
  acceptance, full repository integration, complete B9 migration — stated).
- `samples/vision/yoloe/runtime/cpp/README_cn.md:335` (same paragraph).

UNetMobileNet — "Independent review remains pending / 独立评审仍待执行"
replaced by: independent review accepted this sample within the reviewed host
migration scope, linking
[the B8 mobile planning independent review](2026-09-28-b8-mobile-planning-independent-review.md);
that review does not close the whole B8 batch or certify a board. The
preceding sentence keeps dataset/real-SDK/board/performance not delivered.

- `samples/vision/unetmobilenet/evaluator/README.md:47` (boundaries section).
- `samples/vision/unetmobilenet/evaluator/README_cn.md:47` (same section).

Both languages carry the same status content per file. No sentence that was
already accurate was changed.

## Static verification performed (host only)

- Sample contract checker, run from the repository root with
  `../rdk_model_zoo/.venv/bin/python`, no exemptions flag:
  - `check.py --sample samples/vision/yoloe` → `1 samples, 0 violations,
    1 skips, 0 exemptions applied`
    ([checker-yoloe.txt](evidence/2026-09-28-review-status-docs-remediation/checker-yoloe.txt);
    the skip is the documented R-STAGE-PURITY CLI policy skip).
  - `check.py --sample samples/vision/unetmobilenet` → same summary
    ([checker-unetmobilenet.txt](evidence/2026-09-28-review-status-docs-remediation/checker-unetmobilenet.txt)).
- New relative links resolve in the worktree
  ([link-resolution.txt](evidence/2026-09-28-review-status-docs-remediation/link-resolution.txt));
  both linked review reports and the B8 review's evidence directory exist.

Not executed — any model or toolchain check: no weight download, export,
calibration, OE/Mapper compile, quantized-accuracy run, board or real-SDK
execution, and no toolchain provisioning. Quantization recipes are treated as
trusted source material per the 2026-09-28 user scope; real quantization
verification is not re-listed as a blocker for this package.

## Status

Documentation-only calibration applied; awaiting Codex review. This package
does not change board, dataset, performance or artifact availability states,
batch (B8/B9) acceptance, or any sample's migration status beyond the scoped
sentences above.
