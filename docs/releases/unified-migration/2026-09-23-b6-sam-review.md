# B6 SAM host migration review

Status: host implementation and independent component review passed. Board=not-run, Closed=no, delivery=not-ready. Local development proceeds under the user’s 2026-09-22 instruction. The source audit’s earlier B5 scheduling hold was true at audit time; B5 host review is now passed.

## Scope and design

EfficientSAM and MobileSAM remain separate customer samples and share an explicitly bounded SAM implementation. Both have independent encoder and decoder pre_process/forward/post_process interfaces; pipeline.predict visibly composes six operations. No decoder execution is hidden in encoder post_process. Runtimes only adapt containers and validate metadata; source float32 casts live in stage post_process. No quantization metadata is silently applied as dequantization.

Assets are exact pairs per sample/target, resolved from existing X5/S manifests. Explicit paths require exact per-stage asset IDs. Both actual models bind before inference; hardware identity is gated before SDK import. X5 inputs use flat tensor maps, S uses named-model maps. Source-cast numeric output dtypes are validated against observed metadata; absent/unknown dtype is rejected. X5 decoder geometry is the published 1×3×128×128 / 1×3×1×1. S permits observed positive spatial dimensions with batch=1, candidates=3, and IoU 1×3 or 1×3×1×1; these are runtime compatibility conditions, not assertions of observed HBM metadata. MobileSAM box shape follows a finite source-supported set: X5 1×4 or 1×4×1×1, S 1×4.

EfficientSAM retains fixed exported two-point prompts, RGB/255, >=0 mask threshold. MobileSAM retains runtime one-box prompts in stretched 512 coordinates, ImageNet normalization and >0 threshold. Neither maps masks back to original geometry. Context and owned results must survive A/B/A calls and SDK buffer reuse.

## File responsibilities and acceptance

| Scope | Owner work | Checks |
| --- | --- | --- |
| samples/_shared/sam_binding.py, sam_runner.py | Exact manifest selection, native metadata, lazy adapter | All 8 pairs, mismatch/error and raw-purity fixtures |
| samples/_shared/sam_stages.py, sam_tensor_io.py; sample pipeline.py | Two stages, numeric transforms, thin named pipeline | Four source implementations, explicit stages vs predict, A/B/A, errors |
| sample model, runtime/main, run, visualization, test_data | Customer preparation/entry/IO | Offline CLI, fake download/shell, exact source assets |
| sample root/model/runtime/evaluator README pairs | Contract anchors, self-contained commands, executable API, honest results | Parser parity, link checks, API/recipe fixtures, independent review |
| sample conversion scripts/config/docs | Deep merge equivalent files; retain proven recipe differences | Source map hashes, real arguments/paths, prerequisites and historical results |
| indices, manifests if needed, ledger, evidence, plan | Integration and board handoff | Full host regression + sample checker, separate independent review |

No board, remote computer, real model download, conversion toolchain execution, commit, push or publication is part of B6 host execution. All download command tests must replace network calls or use an inert python3 shim. Source files and historic evidence remain intact.

## Final host execution (2026-09-23)

Two customer samples now share exact pair binding, SDK adaptation and explicitly composed encoder/decoder stages. Twenty bilingual customer READMEs cover root/model/runtime/conversion/evaluator; both source performance procedures remain available separately from the new migration comparator. Sixteen target-specific compilation configs and source provenance are retained. Source platform copies remain byte-identical (106/106 files); no C++ capability existed in this source scope.

The host full regression passed 781 tests before the final two boundary regressions. Affected scopes were rerun after each change: final shared 101, EfficientSAM 19 and MobileSAM 17 passed; current distinct coverage is 783, not a claim that the initial 781-run contained later changes. Contract checker: 30 samples / 0 violations / 33 policy skips / 84 existing B9 exemptions. The skips do not certify dynamic behavior. README local link check: 153 targets, zero failures.

During review, fixes addressed unregistered Torch wrapper weights, missing upstream checkpoint/ZIP fallback and S embedding dump capabilities, mutable bound metadata, failure evidence before runtime load, EfficientSAM's incorrect box documentation/API, MobileSAM dry-run shape claims, and incomplete conversion/performance instructions. See the separate [independent report](2026-09-23-b6-independent-host-review.md) for reviewer ownership and closure evidence.

Evidence: [full and affected command records](evidence/2026-09-23-b6-local-regression.json), [exact working-tree identity](evidence/2026-09-23-b6-working-tree-snapshot.json), [source inventory](evidence/2026-09-23-b6-sam-source-inventory.json). This work is uncommitted, based on develop `16c5d04d0c71ebd160400c14fb2be1c2eb2513f0`; no new commit or publication is claimed.

## Remaining execution boundary

Board=not-run, Closed=no, delivery=not-ready. The user requested local-only work and deferred board environment validation. Real SDK metadata/model execution, OE export/calibration/compile, representative calibration, dataset accuracy and current latency remain unverified. Missing pinned upstream revisions/checkpoint hashes and publisher artifact hashes remain explicit source prerequisites. No real model download, network/remote access, board execution or toolchain execution occurred during B6. The [handoff queue](2026-09-22-host-development-and-board-handoff.md) includes both samples on X5 8GB/4GB and S100/S100P/S600. B7 local development may proceed under the existing authorization; this does not close board acceptance.
