# HIMLoco conversion/evaluator documentation migration

Base: `26ff8a2272c7f8af7d0f9afb0c57dac112cce06e`.
Author implementation record; independent whole-sample review remains open.

Seven source Python scripts are relocated byte-for-byte into the unified
conversion/evaluator directories. Four bilingual guides retain the original
recipes, input geometry, historical compiler/runtime metrics and licensing.
They add explicit repository-root commands, actual CLI defaults, outputs,
published-versus-rebuilt artifact identity, source attribution and navigation.
Root guides now link to these directories rather than the archived source.

Ruling: preserve the trusted conversion/evaluator implementations in this step;
no speculative numerical or toolchain rewrite is needed for documentation
migration. Duplicate helper consolidation can be assessed with the remaining
whole-sample work; this record does not claim deep-refactor completion.

Verification: seven scripts match archived source bytes and parse as Python;
15 existing HIMLoco host tests passed. Sample contract reports zero violations,
one ordinary CLI policy skip and zero exemptions. See
[evidence](evidence/2026-09-28-b10-himloco-docs/source-preservation.json) and
[contract](evidence/2026-09-28-b10-himloco-docs/contract.json).
No export, calibration, compilation, numerical model evaluation, downloads or
board execution was performed. User explicitly excludes recipe reruns as a
migration requirement. C++ migration and whole-sample review remain open.
