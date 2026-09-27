# YOLO maintenance scope — user-directed deduplication

User decision: 2026-09-27; implementation/verification: 2026-09-28.
Base: ad2f63d. Independent Review=not-run, Board/real SDK/OE/dataset=not-run,
Closed=no for the full migration. H0–H9 continues under the revised scope.

## Scope that now controls the earlier reports

The user explicitly removed duplicate standalone S YOLO series from the migration
and selected the simpler maintained Ultralytics implementations/artifacts, with
runtime-dequantized floating outputs. A follow-up answer explicitly retained
YOLOE, YOLO-World and YOLO26 Depth because they provide distinct capabilities.
This supersedes the earlier B9 requirement to consolidate standalone YOLO11,
YOLO11Pose, YOLO11Seg and iMoonLab YOLOv13 Python/C++/conversion implementations.
YOLOv5s remains in the existing YOLOv5 sample. X5's original Ultralytics YOLOv13
family remains; the newly added S iMoonLab route is removed.

The four sample IDs, their ten artifacts and eight benchmark rows are removed
from active `docs/release/s` manifests. Summary totals were recomputed. The
standalone route module, exact-ID downloader branch, S-only iMoonLab family
addition and source-specific NMS overrides are removed. Runtime exact-ID support
still accepts maintained Ultralytics YOLO/YOLO26 records. Retired identities fail
explicitly; they do not silently select a different artifact. Normal family/task/
size preparation continues unchanged. Distinct YOLOE/World/Depth sources remain.

`platforms/s` contains historical snapshots, not maintained entrypoints. Original
records/source pins and prior review evidence are retained for provenance; no
rewrite of past passing/failing evidence makes it current acceptance. The exact
removed active records are stored in [retired-records.json](evidence/2026-09-27-yolo-deduplication/retired-records.json).
Their old independent CLI/source code is not brought into canonical samples, and
there is no remaining obligation to migrate its special integer-output behavior.

## Simpler runtime boundary

The YOLO-specific affine Quantization object, manual postprocessing transforms
and unused dequantize_tensor/dequantize_outputs helpers are removed. Maintained
DFL detection, segmentation and pose, plus LTRB detection, require finite floating
outputs with NONE/absent quantization metadata. Integer or SCALE-tagged outputs
fail at binding, with an instruction to select the maintained floating-output
artifact. This also rejects misleading SCALE descriptors on floating arrays;
there is no guess or hidden cast. Shared quantization support for unrelated,
nonduplicate samples is unchanged.

Forward still returns physical floating arrays without decoding, activation or
layout conversion. Postprocess validates/layout-normalizes, decodes and restores
geometry. Historical tests for now-discarded standalone variants were replaced
with retirement rejection and maintained floating-path coverage. Prior evidence
continues to describe its exact prior commit, not the current interface.

## Pose work retained from the interrupted increment

DFL pose now shares the strict runner/binder and per-image PreparedDetection
transport. It binds nine semantic class/box/keypoint heads by shape, validates
COCO-17/16-bin/square geometry, pairs skeletons with their NMS-selected boxes,
restores points with the actual integer resize/padding and returns owned arrays.
No previous-image state is cached. Default five-tuple visibility is one stable
sigmoid probability; the proposed standalone raw-logit compatibility mode was
removed when the user dropped that variant. The X5 legacy adapter retains its
three-tuple x/y/probability surface.

The original X5 pose source is preserved byte-for-byte as a licensed test fixture,
with source SHA/path/helper hashes. Four interior size/resize cases compare its
actual postprocessor with the maintained implementation. Other fixtures cover
extreme logits, context interleaving, buffer reuse, empty outputs, NMS pairing
and metadata rejection. These synthetic tests do not execute a real model.

## README, plan and verification

The root/model/runtime README pairs no longer advertise standalone downloads or
manual affine postprocessing. They describe the retained families, stage I/O,
floating-output requirement, ROI masks and point probability domain. Three
complete bilingual API examples (six snippets) execute with the actual runner/
binder and a fake SDK. The pose evaluator documentation now states its historical
v>0 serialization rule without falsely describing it as confidence filtering.
The spec, active host plan, legacy plan pointer and migration map record the user
scope override; the standalone source inventory remains explicitly historical.

Exact current host commands, exit codes and complete output are captured in
[host-results.json](evidence/2026-09-27-yolo-deduplication/host-results.json).
The [machine summary](evidence/2026-09-27-yolo-deduplication/result.json) distinguishes
host evidence, generated catalog checks and unrun board/SDK/OE/dataset work.
Full migration acceptance and independent review remain open. No board, HP or
SSH action was performed.

Verified totals: 375 host tests and 121 publisher tests passed. Catalog generation
and TypeScript checking passed (57 families, 595 configurations, 812 benchmark
observations). The migration checker covered 44 samples with zero violations,
46 documented policy skips and zero exemptions. README verification resolved 118
local links and executed six examples against a fake SDK. Catalog generation
still reports historical metrics with no published dataset (X5 125, S 35, X3 18);
these warnings do not constitute new accuracy validation.
