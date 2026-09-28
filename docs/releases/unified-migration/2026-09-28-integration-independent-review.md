# Existing repair-branch integration — independent host review

Reviewer: Codex. Baseline 81c6e534. Disposition: H0 complete in its defined host
integration/tool-remediation scope. H3, H8, H9 and whole delivery remain open.
No merge, product edit, board connection or real quantization execution occurred.

## Branch inclusion

All nine existing B3/B6/B7 repair branches are represented in the current HEAD.
The signed-dtype branch is an ancestor; every commit reported by git cherry for
the other eight branches has a matching patch already in HEAD (no '+' entries).
[Exact branch tips and patch-equivalence results](evidence/2026-09-28-integration-independent-review/branch-inclusion.json).
This covers B3 capture, both B6 Python310/scheduling lines, and B7 bindings,
catalog inventory, dtype aliases, metadata, README and native capture branches.
Different cherry-pick commit IDs are not missing integrations. No redundant
merge or branch deletion is necessary. Current catalog-wide validation remains
under H8/H9; patch inclusion alone is not its behavioral acceptance.

## Tool remediation inspection

B3 temporarily evicts the reviewed dependency closure before importing pinned
source helpers, captures the actual loaded modules, then restores sys.modules
and sys.path. Its tests exercise warm-interpreter dependency drift and restoration.
YOLOv5 native run_capture snapshots the verified audit bytes before child execution
and archives those bytes afterwards, preserving the observer's empty-directory
contract. The comparator requires a matching archived audit digest and both
source and unified stdout/stderr, copies them into portable evidence, and fails
with a report for missing or tampered records. Current regression tests exercise
these exact earlier archival failures, not merely field presence in source code.
The earlier 2026-09-24 failed reviews remain historical; the 2026-09-26 fixes are
present and the host tooling portion is now independently rechecked.

## Fresh host verification

290 tests passed: B3 capture 30; YOLOv5 79; FCOS 38; YOLOWorld 19; LPRNet 23;
MODNet 13; ByteTrack 12; shared SAM 40; MobileSAM 17; EfficientSAM 19.
[Main logs](evidence/2026-09-28-integration-independent-review/host-regression.json),
[SAM logs](evidence/2026-09-28-integration-independent-review/sam-correct-path-recheck.json),
[current reviewed code hashes](evidence/2026-09-28-integration-independent-review/reviewed-code-hashes.json).
The review first invoked a nonexistent samples/vision/sam/tests directory; that
reviewer command error is retained in sam-regression.json. The corrected command
runs both real sample directories and all shared test_sam modules. No product fix
or dependency installation was used to turn that invocation into a passing run.

All checked runtime/tool/shared paths have no working diff. Parallel Claude work
on FCOS/ByteTrack documentation was not staged or claimed accepted here. The
historical board matrix, native numerical comparison gaps, MODNet manual asset
availability and ByteTrack S100P publication gap are unchanged. H3 still requires
its document/integration rollup; no board failure has been converted to a pass.
