# VLA pinned integration — independent host review

Reviewer: Codex. Base f6b21941. Status: accepted for the parent repository's
ACT/Pi0 source-integration and documentation scope. No upstream implementation
was modified or executed by this review. This does not close all of H7.

Both initialized submodule HEADs match the registry, parent gitlinks and
.gitmodules URLs: ACT 326ea043be204de25223d95c7d918efe8672dc66;
Pi0/S600 a32de276bc1681a2b1531012de111eaa1c16acb6. Both worktrees are clean.
The parent integration suite passed two tests covering these pins, distinct
versions, guide initialization commands, manual manifest entries with no
published assets, and removal of old duplicate platform gitlinks.

Codex read the six bilingual parent guides, checked their local links and
command parity (ignoring translated comments), and compared target/version,
resource paths, CLI options/defaults and historical measurements with the pinned
upstream documents and parsers. S100 ACT and S600 ACT require different checkout
versions and LeRobot/data conventions; the S600 checkout also contains Pi0.
The recommended S600 calfix YAML exists at the documented root location.
Pi0 offline flags and actions.npy/result.json/engine.log agree with its parser
and writer. Live-control commands are distinguished from offline BPU execution.

ACT's 20-warmup/200-call 3.92/2.29/6.20 ms and 161.2 inf/s figures, and Pi0's
64 historical chunks plus one-input error figures, match the upstream records.
Guides correctly avoid presenting them as current migration results or task
success rates. Complete upstream source and workflow guides remain accessible
without reconstructing partial Model Zoo wrappers or changing the pinned layout.

[Fresh evidence](evidence/2026-09-28-vla-independent-review/integration-checks.json)
contains command output, pins, clean-state checks and hashes of all six parent
documents. Acceptance concerns source preservation and parent navigation only;
it does not certify vendor SDK compatibility, trained artifacts, quantization
recipes, board inference or robot operation. Those workflows were not run.
