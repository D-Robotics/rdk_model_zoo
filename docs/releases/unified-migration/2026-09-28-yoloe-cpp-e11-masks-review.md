# YOLOE native E11 ROI masks and binary-mask boundary fixes

Base: `acd7a3783e3efa2230137dbf701410e3443542a2`. Native S E11 ROI restoration
now complements E26's distinct mask algorithm. This is implementation self-check,
not full native-runtime acceptance. NV12/SDK ownership, identity gates, public
stages/application, remaining migration and whole-branch review remain open.

## Behavior and fixes

E11 clips boxes to actual image content before cropping the 160×160 prototype,
combines 32 channels, thresholds raw values strictly above 0.5, resizes the binary
crop using Lanczos4, and optionally applies a 5×5 rectangular opening. This
preserves the source S algorithm while using the canonical actual-axis geometry
and visible-content crop correction. It does not use E26's full-canvas logit
interpolation or the X5 Python full-image probability-mask path.

Two existing Python helper defects were exposed while verifying the ROI contract:

- A zero-sized box was padded to at least one pixel, contradicting the evaluator's
  exact ROI dimensions. A failing fixture returned `(1,7)/(8,1)/(1,1)` instead of
  `(0,7)/(8,0)/(0,0)`. The shared DFL mask helper now preserves both zero axes.
- Lanczos4 can overshoot 0/1 uint8 data to 2. A fixed 8×8 pattern resized to 17×17
  reproduces it; the failing test is retained. Both native E11 and the shared
  Python helper now convert positive values to 1 after optional morphology,
  preserving the final foreground support while enforcing binary output.

These fixes affect the shared Python DFL ROI helper used by canonical YOLOE S11
and Ultralytics DFL segmentation. X5 YOLOE full-image masks and E26's distinct
mask algorithm are not routed through this helper. Source snapshots stay intact.
Bilingual YOLOE and Ultralytics API READMEs now document the exact shape/binary
contract; native documentation also explains E11's threshold, interpolation,
morphology default and differences from other protocols.

## Verification

The native image test first failed to compile before `restore_e11_masks` existed,
then passed with the new implementation. Fixtures cover E11 0.5 equality, empty
axes, wrong protocol, full-mask restoration and the Lanczos overshoot pattern,
in addition to existing E26/geometry checks. The Python regression initially
failed on zero-axis shape and then on overshoot before each respective fix.

The real E11s mask comparison uses the exact 38 native-decoded candidate boxes
and coefficients saved in the preceding E11 evidence. Identical input values
are supplied to both mask implementations so this checks mask math, not an
unqualified full Python/C++ pipeline equivalence. With morphology off and on,
all 38 ROI shapes and pixels match exactly. Restored boxes are compared at
rtol=atol=1e-6. Native/Python arrays, per-mask hashes, model/prototype/candidate
identity references, complete command/output/timing records are retained in
[evidence](evidence/2026-09-28-yoloe-cpp-e11-masks/). E26 mask regression uses the
same exact-pixel criterion from the previous increment.

The full required host regression, evaluator/export suites, sanitizer C++ tests,
README command/link checks and migration contract results are captured alongside
the comparisons. No board, SDK/OE build, quantized-model equivalence, dataset AP
or full native application acceptance is claimed. H0–H9 remain open.

## Final checks and evidence corrections

All 463 Python tests passed: YOLOE 29, exporter 5, evaluator 10, Ultralytics 143,
shared 153, ResNet 52, OCR 44 and checker 27. Five native sanitizer tests passed;
E11 38 masks in each morphology setting and E26 213 masks had zero pixel/shape
and restored-box differences in their respective comparisons. Migration checks
reported 45 samples, zero violations, 47 declared skips and zero exemptions;
106 YOLOE local links passed.

The original source-equivalence fixture failed because the fixed source genuinely
contains Lanczos values of 2 (five changed pixels in the first reported mask).
The revised fixture pins that overshoot exists, then requires exact equality to
the source foreground encoded as 0/1; X5 remains compared without normalization.
This is an explicit binary-representation correction, not unchanged bitwise
source output. The initial failure log and status records are retained.

The native and Python evidence runners initially reused three log filenames.
Those three checks were captured again under unique `verified-*` names, with
new timestamps in `verification-reruns.json`; final records reference this shared
verified rerun. Native runner logging now has a separate prefix for subsequent
runs. Numerical comparisons and other suite logs had distinct destinations.

The retained `initial-yoloe.log` and `yoloe.log` contain NumPy's original
`AssertionError: ` trailing space. Final diff whitespace checking excludes only
these two raw failed-output logs; source/document checks are unchanged.
