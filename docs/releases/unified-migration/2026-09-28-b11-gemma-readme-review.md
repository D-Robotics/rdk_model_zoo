# B11 Gemma bilingual README organization

Base: `f4f4f4d4`; author documentation review, not independent acceptance.

## What changed

Fourteen README files now cover sample overview, model, native runtime, conversion, evaluator,
third-party dependencies and example inputs. Existing architecture descriptions, five native
application parameter tables, HTTP examples, screenshots, source model sizes/checksums and
full conversion tutorials are retained. Top-level/sample indexes include Gemma as in progress:
50 in-repository samples plus the two separately pinned VLA integrations.

The model guide separates published S100P/S600 HBMs from manual S100 artifacts, explains
same-name files and nonempty-file reuse, and attributes the four HBM checksums to the fixed
source README rather than claiming newly computed hashes. Dependency documentation records
the tokenizers commit, Rust requirements and source installer behavior. Those runtime helpers
still need the planned implementation work; documentation does not certify their target/hash gates.

The evaluator guide corrects the native `--prompt` typo to `--prompt_id` and working directories,
uses the explicit build launcher, and states actual golden comparison criteria: exact integer
inputs, embedding max absolute error ≤1e-3, zero mask error. Golden input matching, BC cosine,
qualitative generation and historical throughput are distinguished. Conversion README changes
only organize and explain the existing recipe; 21 other conversion files (including complete
bilingual tutorials) remain byte-identical to source.

## Checks

- Migration contract: 50 samples, 0 violations, 51 policy skips, 0 exemptions.
  This resolves the 70 missing Gemma section anchors disclosed in the previous report;
  previous evidence is preserved. Anchors correspond to substantive content, not empty headings.
- Local links, explicit/heading anchors and image references checked in README files;
  final counts are in `checks.json` and `navigation.json`.
- Four real host help/preview invocations return 0 without running native SDK/model commands.
- No native runtime code changed; previous seven launcher tests remain the applicable code evidence.
- No quantization/download/export/calibration/BC/OE, native SDK build, board or robot run.

This completes the current documentation organization increment, not Gemma migration acceptance.
Native stage responsibilities, explicit model preparation and final source/whole-branch review
remain open. Docs will follow those implementation changes. B11 and full H0–H9 remain active.

Evidence: [documentation/source/preview checks](evidence/2026-09-28-b11-gemma-readmes/checks.json),
[complete navigation scan](evidence/2026-09-28-b11-gemma-readmes/navigation.json),
[migration contract](evidence/2026-09-28-b11-gemma-readmes/migration-contract.json).
