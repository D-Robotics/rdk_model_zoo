# YOLO26 OBB common runner and explicit geometry

Implementation self-check, 2026-09-28. Base `fbd56ade607da40a320a198cdbe55808150791cf`.
This is not the final independent migration review. H2 and H0–H9 remain open.

## Change and preserved behavior

`YOLO26OBB` now exposes configuration/construction and pre/forward/post/predict
stages, reusing the detection input/raw stages and `ModelRunner`. `LTRBOBBContract`
binds the nine floating heads by semantic roles, rejects incompatible metadata,
and preserves borrowed raw buffers. The new `obb_decode.py` owns numeric decoding,
rotated IoU/NMS and inverse geometry. The last production dependency on
`Yolo26Runtime` is gone; the obsolete `yolo26_common.py` has been removed. Its
output-order tests now exercise production metadata binding rather than a
parallel role-order implementation.

Preserved: absolute direct LTRB offsets, radians from the maintained exporter,
angle-sign and degree-offset overrides, optional width/height regularization;
X5 classwise rotated NMS, angle wrapping and per-component clipping; S OpenCV
class-agnostic rotated NMS without extra wrapping/clipping. Owned records retain
`rrect`, `score`, `id`. No manual output dequantization was introduced.

Intentional corrections: actual integer resize/padding and explicit image context
replace idealized geometry; no cached last-image context. Nonfinite metadata/data,
thresholds and angle controls fail explicitly. OpenCV intersection errors propagate
instead of becoming zero overlap. Axis-wise width/height scaling retains the prior
interface's approximation under unequal X/Y scales; this is documented, not claimed
to be an exact polygon transform.

## Evidence scope

The byte-exact reference fixture is the pre-refactor unified OBB implementation at
the base commit, with path and SHA-256 in its adjacent JSON. Tests invoke its real
postprocessor, bypassing SDK construction only. Eight platform/resize/regularize
combinations compare selected IDs, scores and rotated rectangles. This proves
continuity with the maintained implementation, **not** equivalence to the original
S standalone sigmoid-angle decoder. That prior difference is explicitly documented.
Separate tests cover all four target bindings, raw-buffer identity/result ownership,
interleaved non-square images and integer geometry, classwise versus agnostic NMS,
angle controls, empty results, metadata errors and target conflicts.

Both runtime READMEs include the same executable OBB example, raw-buffer lifetime,
result shape, units, default values, platform policy table, approximate inverse
geometry and verification limits. The bundled image demonstrates the interface,
not DOTA accuracy.

## Validation

[Complete command records](evidence/2026-09-28-yolo26-obb/host-results.json)
and [machine summary](evidence/2026-09-28-yolo26-obb/result.json):

- Ultralytics 141, shared 144, ResNet 52, OCR 44, checker 27: **408 tests passed**.
- Migration checker: 44 samples, 0 violations, 46 policy skips, 0 exemptions.
- 128 local README links resolved; 16 bilingual Python examples executed through
  the actual shared runner with injected fake SDKs. No inference stage is mocked.
- Initial missing-contract test failure is retained in `red.log`. During test
  migration, a pose fixture incorrectly supplied a `classes` argument and the new
  reference loader initially lacked its import path; both harness errors were
  corrected before the full captured green run.

Board, real SDK, OpenExplore conversion and DOTA evaluation: **not-run**.
Independent whole-branch review: **not-run**. No hardware connection attempted.
Native audit, remaining documentation, YOLOE, B10/B11/H8 and final integration
remain in the full completion plan; this increment does not close those tasks.
