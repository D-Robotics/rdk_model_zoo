# YOLO26 segmentation stage consolidation and README review

Base: `9da5acf`. Codex implementation and host verification. Independent whole-
branch review, board inference, real SDK/OE and dataset/performance validation
remain not-run. H2 and the complete migration are not closed.

## Shared stages with distinct mask mathematics

YOLO26Seg now inherits the common segmentation preprocessing, raw forward,
postprocessing, prediction and scheduling delegation. LTRBSegmentationContract
binds ten outputs by complete geometry/dtype metadata: class logits, direct
four-value boxes and 32 coefficients at strides 8/16/32, plus stride-4 prototypes.
Head layout is NHWC; prototypes accept proven NHWC or NCHW. Integer/SCALE data,
DFL bindings, ambiguous roles, missing descriptors and malformed/nonfinite arrays
are rejected. Direct library execution uses the common hardware-identity gate.

Box decode, filtering, classwise NMS, mask-coefficient selection, result ownership
and explicit per-image geometry share the existing stage implementation. The
mask algorithm intentionally remains protocol-specific. DFL retains cropped
binary masks and optional morphology. YOLO26 combines prototypes/coefficients,
uses stable sigmoid, bilinearly upsamples probabilities to model input size,
crops to each model-space box, removes actual integer padding, resizes to original
size, thresholds >0.5 and takes the clipped original-image ROI. No morphology
is applied to YOLO26. Processing each original-size probability map separately
avoids retaining an N×H×W stack just to return ROI masks.

The now-unused canonical decode_seg_layer copy is removed. Historical source
helpers and the legacy process_mask utility used by existing regression tests
remain; maintained YOLO26 inference uses the context-aware decoder above.

## Public behavior and compatibility

The canonical result remains (float32 boxes, float32 scores, int64 IDs, boolean
ROI mask list). Masks and result arrays own storage. Empty detection returns
(0,4)/(0,)/(0,)/[]; degenerate masks are (0,0). Forward leaves physical dtype,
layout and SDK buffer identity unchanged. PreparedDetection carries immutable
geometry instead of depending on the last image. Explicit width/height calls
continue to work.

The X5 compatibility adapter was updated to accept the explicit transform used
by shared predict and still reconstructs a boolean (N,H,W) full-image stack.
A failing regression first reproduced its missing-dimensions exception after
stage migration; the complete failure is retained in legacy-red.log. Its actual
predict path now runs through the shared stages and is tested. S retains ROI
masks. No native implementation or asset manifest changed in this increment.

Confidence is finite in (0,1), NMS in [0,1]. The common decoder honors valid
thresholds without the prior hidden [1e-6,1-1e-6] clamp and selects the class in
raw logit space before sigmoid, avoiding saturated scores changing class IDs.
It uses actual integer geometry; rounded inverse coordinates can differ from
the prior ideal-scale rule. NMS defaults remain library 0.65 and explicit CLI
X5 0.70/S 0.45. Both README languages explain these differences, output placement,
buffer lifetime and a complete seventh task API example.

## Source evidence and verification limits

Original S/X5 source modules are preserved byte-for-byte with license headers and
SHA-256 metadata in tests/fixtures/yolo26_seg_sources.json. Reference tests verify
three archived helper hashes per source before execution. Pins are X5
ac115717197920355fc390bb04299b20e6436864 and S
380e1a2bf42041af54be6f34935e50197cfadff9.

Tests execute original S postprocessing for two geometries × two resize modes,
comparing ROI masks exactly and numerical results within 1e-6/1e-5. The old X5
source hard-coded a 640 prototype scale and resized masks directly to original
size without removing letterbox padding. At a synthetic 640 model input and
640×320 image, a test reproduces its wrong full-mask result while boxes agree.
The canonical implementation already used the S-style mask correction before
this increment and continues to do so; no claim of universal original-X5 mask
equivalence is made.

Additional fixtures cover both NV12 transports across four target profiles,
reordered NHWC/NCHW prototypes, raw identity, border clipping, empty results,
saturated classification logits, NMS/coefficient pairing, buffer reuse,
interleaved geometry and malformed metadata. These are synthetic host tests,
not artifact/runtime certification. [Commands/logs](evidence/2026-09-28-yolo26-segmentation/host-results.json)
and [verified summary](evidence/2026-09-28-yolo26-segmentation/result.json) record
actual exits. OBB still requires consolidation; native, YOLOE, B10/B11 and all
remaining H0–H9 requirements continue.

Final host results: 401 tests passed (134 Ultralytics, 144 shared, 52 ResNet,
44 OCR, 27 checker). Migration checker: 44 samples, 0 violations, 46 documented
policy skips, 0 exemptions. 126 local README links and 14 bilingual fake-SDK
examples passed. Full migration acceptance remains open.

The byte-exact archived X5 source contains 11 inherited trailing-whitespace
lines. Full staged whitespace checking reports them (rc=2); checking maintained
changes with only that source fixture excluded passes (rc=0). Original bytes are
intentionally preserved for SHA verification, not reformatted. Both check outputs
are retained in whitespace-results.json and its logs.
