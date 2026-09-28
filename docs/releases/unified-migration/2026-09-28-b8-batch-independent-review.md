# B8 aggregate independent acceptance — non-board scope

Reviewer: Codex. Base 11c04d6d. Status: B8/H4 non-board migration accepted.
All eight sample scopes have independent content/code reviews and fresh aggregate
verification. No runtime or customer documentation was changed by this rollup.

| Sample | Preserved capability and reviewed boundary | Independent content review | Fresh tests |
| --- | --- | --- | --- |
| UNet | Five X5 backbones, 512×512 VOC mask, canonical evaluator, source conversion documentation | [Segmentation review](2026-09-28-b8-segmentation-independent-review.md) | 20 |
| UNetMobileNet | S100/S600 original-size masks, Python/native resources, six-level guides | [Mobile/planning review](2026-09-28-b8-mobile-planning-independent-review.md) | 19 |
| PP-LiteSeg | X5 decoded int32 IDs, source visualization, evaluator compatibility and recipes | [Segmentation review](2026-09-28-b8-segmentation-independent-review.md) | 18 |
| YOLO26 Depth | Twenty target/variant assets, separate NV12/log and RGB/raw profiles, Python/X5 native and offline metrics | [Depth review](2026-09-28-yolo26-depth-independent-review.md) | 38 |
| Depth Anything V2 | Source per-pixel normalization, explicit geometry fixes, relative float depth and display separation | [Independent review](2026-09-28-depth-anything-independent-review.md) | 15 |
| LaneNet | Python/native binary/embedding outputs, no invented lane-instance clustering, source figures | [Independent review](2026-09-28-lanenet-independent-review.md) | 22 |
| DiffusionDrive | Four named inputs, five source cases, native/shared transport and strict saved-output metrics | [Mobile/planning review](2026-09-28-b8-mobile-planning-independent-review.md) | 23 |
| PointNet | Point segmentation and source normalization, native resource fix and exact int32 ordering remediation | [Independent review including R1/R2 closure](2026-09-28-pointnet-independent-review.md) | 26 |

## Completion evidence

Rechecked all 304 file records across the seven accepted evidence snapshots.
Only two UNetMobileNet evaluator READMEs differ; their later status-only change
was independently accepted in the [status documentation review](2026-09-28-review-status-docs-independent-review.md).
All implementation hashes remain as reviewed. [Snapshot reconciliation](evidence/2026-09-28-b8-batch-independent-review/snapshot-recheck.json)
retains the precise differences rather than replacing old evidence.

Fresh full sample suites passed 181 tests in total under the current shared
implementation. [Full outputs](evidence/2026-09-28-b8-batch-independent-review/host-suites.json)
retain per-sample commands and results. [Documentation checks](evidence/2026-09-28-b8-batch-independent-review/docs-checks.json)
bind 90 bilingual README files; no local file link target is missing. All eight
sample contract checks returned zero violations, one existing CLI policy skip
each, and zero exemptions. These mechanical checks supplement the explicit
content/source reviews above; they do not substitute for them.

Across the reviewed scopes, model selection and source target/variant boundaries,
pre/inference/post responsibilities, file/visualization separation, Python/C++
source capability, API/CLI instructions, source recipes, evaluation scope,
historical figures and known discrepancies are accounted for. No remaining B8
non-board blocking finding is open. Dataset hub and repository-wide navigation
are H8/H1 integration work and remain tracked separately.

## Exact closure boundary

H4 closes under the user's current non-board scope. Board inference, full vendor
SDK builds, new dataset performance/accuracy and quantization execution are not
claimed. Trusted quantization recipes were reviewed as documents only; missing
new recipe execution is not a delivery blocker by user decision. Historical
failures/discrepancies remain in source-specific guides. Ledger board fields
stay not-run and overall Closed=no retains the separate full-delivery boundary.
This acceptance does not close H1/H8/H9 or the entire migration.
