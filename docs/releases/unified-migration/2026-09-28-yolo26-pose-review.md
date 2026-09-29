# YOLO26 pose stage consolidation and README review

Base: `02a3ecb`. Codex implementation/host verification. Independent whole-branch
review, board inference, real SDK/OE and dataset/performance verification are
not-run. This increment does not close H2 or the full migration.

## Implementation

YOLO26Pose now inherits the common pose preprocessing, raw forward, postprocessing,
prediction, scheduling delegation and per-image transport. The wrapper selects a
strict LTRBPoseContract rather than maintaining another pipeline. The contract
binds nine floating NHWC heads by shape at strides 8/16/32: one person logit,
four direct LTRB distances and 17 (x,y,visibility-logit) triplets. DFL binding,
integer/SCALE output, missing/ambiguous descriptors and malformed physical arrays
fail explicitly; the shared execution identity gate applies to direct library use.

pose_decode shares confidence filtering, NMS selection, ownership and geometry.
The now-unused canonical decode_pose_layer copy was removed; archived source
helpers remain for provenance. Only the box and keypoint formulas differ: DFL uses its 16-bin expectation and
`(2 * keypoint + anchor - 0.5) * stride`; YOLO26 uses direct distances and
`(keypoint + anchor) * stride`. Visibility receives one stable sigmoid in either
case. No loading, file I/O, drawing, label lookup or SDK operation enters these
numerical helpers. The model-facing stages still have the pre/forward/post/predict
surface requested by the migration design.

PreparedDetection carries immutable geometry; raw arrays borrow SDK storage and
remain physically unchanged. Returned five-tuples own float32 boxes/scores/points/
visibility and int64 IDs; empty arrays preserve their ranks. NMS selects matching
skeleton indices. The X5 compatibility adapter retains its list of records with
integer boxes; S retains the four-tuple with combined (N,17,3) points. Tests run
both actual compatibility adapters through the maintained stages.

## Explicit differences and documentation

Coordinate restoration now uses actual integer resize/padding, so rounded
letterbox cases can differ from the old ideal-scale inverse. Confidence must be
finite in (0,1), NMS in [0,1]. The old S helper's hidden confidence clamp to
[1e-6,1-1e-6] is removed: a valid supplied threshold is honored exactly. An
explicit source comparison demonstrates this intentional difference at 5e-7.
Stable sigmoid also handles extreme keypoint logits without overflow warnings.
These corrections are not described as universal bitwise source equivalence.

The library NMS default remains 0.65. CLI/legacy platform defaults remain separate;
README examples explicitly use S CLI's 0.45. Both languages explain the direct
point formula, source/SDK boundaries, stage inputs/outputs, lifetimes, confidence
domain and legacy results. A complete sixth API example executes every stage
and compares with predict using the actual runner/binder and a fake SDK.

## Verification and remaining scope

Two original source modules are preserved byte-for-byte with license headers,
source pins and SHA-256 metadata in tests/fixtures/yolo26_pose_sources.json. The
three helpers on each side are checked against their fixed source hashes before
reference execution. X5 pin: ac115717197920355fc390bb04299b20e6436864;
S pin: 380e1a2bf42041af54be6f34935e50197cfadff9. Tests run both original decoders
for two unrounded geometries × two resize modes and compare results while
respecting X5's integer-box interface.

Additional tests cover both metadata transports across four targets, reordered
heads, raw identity, once-activated extreme logits, direct point formulas,
overlap/NMS pairing, reusable buffers, empty arrays, interleaved contexts,
actual integer geometry and malformed metadata. Existing YOLO26 fixture loaders
were updated to inject through the common runner, retaining full metadata
validation; no production hardware gate was bypassed.

[Commands and complete logs](evidence/2026-09-28-yolo26-pose/host-results.json) and
[verified totals](evidence/2026-09-28-yolo26-pose/result.json) are host evidence.
YOLO26 segmentation and OBB still use the previous runtime and require their
own consolidation/audit. Native code, YOLOE, B10/B11 and H0–H9 remain open.

Final verification: 393 host tests passed (126 Ultralytics, 144 shared,
52 ResNet, 44 OCR, 27 checker). Migration checks: 44 samples, 0 violations,
46 documented policy skips, 0 exemptions. 124 local README links resolved;
12 bilingual examples executed against the fake SDK.
