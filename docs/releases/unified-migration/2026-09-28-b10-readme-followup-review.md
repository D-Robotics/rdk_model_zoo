# B10 README follow-up independent review — 2026-09-28

Status: **changes-required (documentation only)**. Runtime reviews remain valid in their bounded scope. H6 is not closed. Reviewer Codex; implementation Claude Code + GLM.

## B10-DOC-R1 — Paraformer test-data guide still calls native consumer unmigrated

Both `samples/speech/paraformer/test_data/README.md` (Layout and custom inputs) and its Chinese pair claim that the C++ consumer is still being migrated. This contradicts the implemented native executable, prepared-manifest/NPY reader and launcher, already independently host-checked in `2026-09-28-paraformer-independent-review.md`. Replace the stale claim with the actual prepared-feature handoff and a link to the native guide, preserving explicit SDK/board not-run boundaries. No claim of board acceptance is requested. Check all Paraformer README language pairs for similarly stale implementation status; do not change recipe semantics or broaden verification claims.

Non-blocking clarity: root/Python quickstart uses `outputs/paraformer-prepared`, evaluator prepares `outputs/paraformer-features`, while conversion example and native default use `outputs/paraformer_features`. These are allowed custom directories, not necessarily a runtime defect. Explain which documented preparation produces the conversion example's prerequisite or provide an explicit link to the native preparation command using the identical directory. Do not change CLI defaults merely to normalize spelling.

This review statically read conversion/evaluator contracts, source-attribution boundaries and current parameter tables. User-trusted quantization recipes were not executed. Separate batch evidence will record ordinary host checks; it does not certify OE, models or boards.
