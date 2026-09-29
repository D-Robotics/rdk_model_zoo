# Develop merge — 2026-09-29

User explicitly requested merging the accepted integration branch into develop
and committing. Codex performed the integration; no new product implementation.

Parents: develop `e0759d5a9bc93ff1b2a0cfece9d874be83250499` and accepted integration
`a6e9380b58b020f4fe4d568d59bd4fa1a27cd59a`. Fetch confirmed local develop matched
origin/develop before merging. The four develop-only commits have identical
stable patch IDs to commits already in the integration history. Their changes
are therefore retained, with later independently accepted revisions.

Git reported 23 Markdown conflicts. Each affected file was resolved to its
reviewed successor on the integration branch after the patch-identity check;
no blanket ours/theirs merge strategy was used. Before adding this report and
its evidence, the entire merged index tree exactly equalled the accepted
integration tree. Exact pairs, paths and tree IDs are in
[evidence/resolution.json](evidence/2026-09-29-develop-merge/resolution.json).
All 2,649 product/test/configuration hashes and 599 README hashes matched the
prior acceptance. Both ACT/PI0 submodules were initialized at their existing
pinned commits, without changing gitlinks.

Fresh checks ran in the develop checkout: 1,631 Python host tests passed with
zero skips; Catalog check passed its 130 tests, typecheck, source validation and
reproducible build. Sample contract: 51 samples, zero violations/exemptions;
policy skips retain their explicit scope. Seven Skills / 84 eval definitions
and reference sync validate. All 61 recorded commands exited zero. Commands,
environment selection and raw outputs are preserved in
[evidence](evidence/2026-09-29-develop-merge/). Existing optional dependency caches
were reused; this is not an independent clean-install or vendor-SDK verification.
No new board, actual quantization, live-model or release operation occurred.

This merge supersedes earlier reports' then-correct “not merged into develop”
status; historical reports and failures remain unchanged. Non-board acceptance
and remaining delivery limitations are described in the
[final review](2026-09-29-host-completion-independent-review.md).
Push completion is established by the actual remote ref observation, not by
this pre-commit report. No tag, Release, Hub or default-branch setting is changed.
