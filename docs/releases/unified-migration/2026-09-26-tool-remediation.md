# B3 / B7 evidence-tool remediation — 2026-09-26

Status: implemented and host-verified in the integration branch. This is not a board-test result or final whole-branch independent acceptance.

## B3 fixed-source isolation

Integrated the previously checkpointed capture tool and fixed B3-TOOL-R1. A warm interpreter no longer reuses root `utils.py_utils` children when evaluating the pinned X5 implementation. The complete known dependency set is temporarily removed and restored; actual loaded dependency hashes must pass before either model is created. Four real source wrappers were exercised with an injected SDK, with module/SDK/path restoration checked. The standalone branch and integrated tree both pass all 30 tool tests.

[Original review plus remediation](2026-09-24-b3-tool-independent-review.md) preserves the original changes-required finding and its failure evidence. Board status remains not-run.

## Native capture archival

Integrated the existing YOLOv5 native tools, then repaired two outstanding archival gaps:

1. `run_capture.py` keeps the precise audit bytes verified before execution, records their SHA-256 and archives them after the child exits. This timing preserves the observer's required empty output directory. A real subprocess test changes the external audit during execution and proves the archive still contains the verified original bytes.
2. `compare_native.py` requires the archived source audit and matching hash, as well as the unified process's stdout/stderr beside its run record. It copies the audit and both sides' logs into the portable comparison output, where the existing digest index includes them. Missing files and tampering fail with a saved failed report.

The bilingual evaluator instructions now distinguish instrumentation's `--target s100` (shared S source group, including S600) from the actual unified build/comparison target. The parser never accepted `instrument.py --target s600`; the previous instruction was wrong. No S100P positive support was added.

[Structured evidence](evidence/2026-09-26-native-archive-remediation.json), [failure reproduction](evidence/2026-09-26-native-archive-red.log), [full YOLOv5 host test output](evidence/2026-09-26-native-archive-green.log).

## Verification and remaining work

The red run had 41 native tests and six failing cases, demonstrating all archival gaps. After repair the complete YOLOv5 suite passes 79 tests, including portable C++ checks and instrumentation compilation against host stubs. The contract checker reports 36 samples / 0 violations / 39 policy skips / 60 existing Ultralytics documentation exemptions. No model downloads, SSH, board builds or inference were performed.

Old captures lacking the audit copy and run-record hash do not satisfy the strengthened gate; do not fabricate missing process evidence. Preserve historical records and perform a scoped new capture when board access returns. The remaining non-board migration, README audit and final integration review stay active under the [host completion plan](../../superpowers/plans/2026-09-26-host-completion.md).
