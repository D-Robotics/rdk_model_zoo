# YOLOE canonical evaluation — implementation self-check

Base: `4c2d7dae8f94e5a67d1886126495561ec93d722c`. This increment implements
canonical box/mask dataset evaluation and complete bilingual evaluator READMEs.
It is an implementation self-check, not independent whole-branch acceptance.
B9/H5 and H0–H9 remain open; native C++ migration remains outstanding.

## Behavior and documentation

The evaluator requires an explicit model digest, target protocol, dataset and
reviewed PF-to-dataset category mapping. Both category names and IDs are checked
against the fixed 4585-class vocabulary; every annotation category must be
covered. The example mapping contains only person and chair, not COCO-80.
Duplicate identities, path escapes, unreadable images and mismatched image/mask
geometry fail rather than dropping cases. Partial failures retain completed-image
records and explicitly partial predictions. Predictions-only mode never reports
AP. Empty detections are scored as empty; undefined COCO statistics remain -1.

ONNX CPU inference disables graph optimization and uses float RGB. It does not
simulate NV12, quantization or BPU execution. The implemented board backend uses
the existing float artifact identity/metadata gates but has not run on a board.
The runtime and evaluator share `runtime/python/postprocess.py`; the runtime
stage delegates to that decoder rather than duplicating numerical algorithms.
Separate bbox and segmentation outputs ensure mask area is derived from RLE.

The bilingual README retains original X5/S100 latency tables, measurement
conditions, X5 two-thread allocation failure and the limited source S100P
one-image comparison. These remain historical measurements. Commands, dependency
installation, complete options/defaults, mapping requirements, metric settings,
output files and failure boundaries are documented for people and Agents.

## Real model evidence

[Evidence directory](evidence/2026-09-28-yoloe-evaluation/) includes all 14 actual
ONNX CLI invocations with full logs, UTC times, return codes and persisted bbox/RLE
predictions. Eight exported models cover X5 E11 s/m/l, S100 E11s and E26 n/s/m/l/x
on both S target protocols. Every command exited 0. This is one bundled office
image with a generated identity mapping across 4585 PF classes and **no ground
truth**: it is not COCO AP, a hardware test or equivalence to historical HBM.

The shared annotation/map snapshots are stored once in `input-snapshots/`;
`evaluation.json` in each case binds their snapshot SHA-256. Original CLI input
files `annotations.json` and `mapping.json` are also retained. Model weights stay
in ignored local storage; model paths and digests are captured in each report.
`implementation-sha256.json` binds the tested Python implementation. The runner
uses local exported models and does not download assets or contact boards.

Detection counts: X5 11s/11m/11l = 38/26/25; S100 11s = 38; E26 n/s/m/l/x =
213/300/300/168/18 for each S protocol. These numbers describe this float check
only; they do not reproduce the source quantized-model counts.

## Checkpoint identity finding

[Checkpoint provenance](evidence/2026-09-28-yoloe-evaluation/checkpoint-provenance.json)
compares all five current E26 export checkpoint hashes to both source release
sidecars. All differ. File digests alone cannot establish whether parameter
tensors changed (serialization metadata may differ), but the source checkpoint
identity is not reproduced. Therefore differences from original HBM inference
cannot be attributed solely to quantization. The preceding export validation
remains a check of the explicitly hashed current checkpoint against its ONNX,
not proof of reproducing the original release checkpoint.

## Verification and remaining boundaries

All 461 host tests passed (YOLOE 29, exporter 5, evaluator 10, Ultralytics 141,
shared 153, ResNet 52, OCR 44 and checker 27). The migration contract check
reported 45 samples, zero violations, 47 declared skips and zero exemptions.
Initial README and contract checks found two nonexistent Chinese COCO links;
they were corrected to existing source pages and only the affected checks were
rerun. Initial failures and final results are both retained.

Host suite results and complete logs are captured in `host-results.json` beside
this evidence. Evaluator tests use the real pycocotools scorer with known
synthetic masks (perfect AP and empty predictions), including persisted engine
metrics, strict mapping, full/ROI geometry, invalid identity, SDK-free help,
partial failures and image-dimension mismatches. Synthetic AP validates scoring
mechanics only. Runtime/source comparisons guard the shared decoder refactor.

No actual board, real SDK/OE compilation, held-out dataset accuracy, accepted
full COCO mapping or new performance benchmark was validated. Native migration,
remaining samples/documentation and independent whole-branch review continue.
