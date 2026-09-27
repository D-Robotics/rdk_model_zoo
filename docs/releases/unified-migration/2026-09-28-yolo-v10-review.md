# YOLOv10 stage consolidation and README review

Base: `ab0fd96`. Host implementation/review only. Independent whole-branch review,
board inference, real SDK/OE and dataset/performance validation are not-run. H2
and full migration acceptance remain open.

## Result

The S YOLOv10 wrapper now inherits the existing DFL detector's preprocessing,
raw forward, postprocessing, prediction and scheduling delegation. It adds only
configuration and construction enforcing the source protocol: DFL logits,
16 bins, strides 8/16/32, square model input, and no NMS. A caller-supplied
contract or injected binding that enables NMS is rejected. This removes the
separate SDK loading, image preparation, positional-output decoding and pipeline
copy. Output shapes bind semantic class/box roles, not enumeration order.

X5's family dispatcher remains on the historical DFL + NMS path. Only S targets
select the NMS-free adapter. Active artifact lists, conversion recipes and native
code are unchanged by this increment.

The public stage transport now uses PreparedDetection (mapping-compatible, with
explicit per-image transform) and bound RawOutputs. The result is the common
owned, tuple-compatible DetectionResult. Legacy explicit original width/height
postprocess calls remain executable. All score-qualified anchors remain in
stride/grid order: there is no NMS, score sort or Top-K truncation. Reused SDK
buffers cannot change already returned result arrays.

## Deliberate behavior corrections

The common input/output boundary rejects malformed images, missing or ambiguous
output descriptors, integer/SCALE output, wrong shapes/dtypes and nonfinite arrays.
Direct execution goes through the common board-identity gate. This boundary still
needs real SDK/board confirmation; no synthetic test makes that claim.

Letterbox coordinate restoration uses the integer resized dimensions and actual
padding, replacing the old ideal-scale inverse. On images such as 59×37 resized
to 64×64, that corrects a measurable coordinate difference. The tests calculate
expected coordinates independently and explicitly demonstrate the difference
from source. Unrounded cases retain source numerical results.

Finite confidence thresholds use the shared [0,1] contract: zero retains all
finite-logit anchors and one retains none. Outside-range/nonfinite values fail
explicitly instead of relying on a logarithm warning. NMS thresholds are ignored
by the no-NMS contract. The bilingual README describes these semantics, buffer
lifetimes, result ordering and X5/S differences, with a complete S600 example.

## Verification evidence

The original S module is stored byte-for-byte with its license and source pin
`380e1a2bf42041af54be6f34935e50197cfadff9`; source and three archived helper
SHA-256 values are checked before source-reference execution. The host comparison
runs original preprocessing on two image geometries × both resize modes, and
original postprocessing on two unrounded geometries × both resize modes.
No SDK constructor is called in those source comparisons.

Other tests exercise raw identity, overlapping boxes without suppression, source
traversal order, reused buffers, empty outputs, two interleaved image contexts,
confidence endpoints, and forbidden NMS contracts. Existing platform fixtures now
inject the fake SDK through the common runner and retain full descriptor binding.
The first source fixture omitted source cfg.anchor_sizes and failed; the fixture
was corrected without altering the pinned source or production decoder.

[Complete commands and outputs](evidence/2026-09-28-yolo-v10/host-results.json) and
[verified summary](evidence/2026-09-28-yolo-v10/result.json) distinguish host
results from unrun external validation. OBB, YOLO26 segmentation/pose, native
capabilities, YOLOE and all remaining H0–H9 obligations continue.

Final host result: 385 tests passed (118 Ultralytics, 144 shared,
52 ResNet, 44 OCR, 27 checker). The migration checker covered 44 samples with
0 violations, 46 documented policy skips and 0 exemptions. All 122 local README
links resolved; 10 bilingual stage examples executed using the actual runner/binder
and a fake SDK. No full migration or board acceptance is implied.
