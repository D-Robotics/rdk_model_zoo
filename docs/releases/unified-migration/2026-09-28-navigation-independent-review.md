# Active README navigation — independent snapshot

Reviewer: Codex. Scope: root and platform entry READMEs plus active samples and
datasets. This is navigation evidence, not full README acceptance or H1 closure.

597 existing README files were scanned, containing 3595 inline link/image
references (including external references, whose reachability was not tested).
For inline local references outside fenced code, all target files exist and
all referenced Markdown anchors resolve under the explicit-ID/heading check.
[Corrected scan and file hashes](evidence/2026-09-28-navigation-independent-review/readme-links-corrected.json).

The first heuristic pass falsely treated twelve C++ lambda signatures inside
fenced examples as Markdown links. Those are valid code examples, not broken
navigation. The scanner was corrected to exclude fenced code; no customer file
was changed for these false positives. The initial output is retained in
[readme-links.json](evidence/2026-09-28-navigation-independent-review/readme-links.json).

Limits: inline Markdown only; no external URL reachability, reference-style
link resolution, Git-submodule interior review or proof of commands/algorithms.
This working-tree snapshot includes still-pending Claude README modifications.
Their recorded hashes allow a final delta check, but the scan does not accept
those packages. Source-depth restoration, figure accuracy, bilingual content,
variant contracts and historical result conditions are independently reviewed
in the corresponding package reports. Final repository validation remains H9.
