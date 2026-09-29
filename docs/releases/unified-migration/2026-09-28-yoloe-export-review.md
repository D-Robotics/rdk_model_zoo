# YOLOE canonical checkpoint export — implementation self-check

Base: `fb9c2e96acbcda364eee18f553268bc60eb392ee`. Canonical PF checkpoint export
is implemented in `samples/vision/yoloe/conversion/export.py` with separate raw
head adapters in `export_heads.py`. This is an implementation self-check, not a
fresh independent whole-branch review. YOLOE, B9 and H0–H9 remain open.

## Source behavior and fixes

The entry explicitly loads a local checkpoint, verifies the family, YAML scale,
three PF branches, reg_max/end2end, strides, mask dimension and exact ordered
4585-class vocabulary. It does not download a checkpoint or run a compiler.
E11 uses cv2/cv3/cv5, DFL16 and opset 11; E26 uses one2one branches, direct LTRB
and opset 17. Linear vocabulary weights are applied as dense 1x1 convolutions;
original parameters are not replaced. Backbone routing and model.<index> node
names are preserved, including X5 attention paths.

For both families every raw anchor and decoded pre-Top-K tensor is checked against
the upstream static PF head, with proposal filtering disabled and flags restored
afterward. E26 additionally requires identical selected anchor/class sets and
compares selected values by identity; output order changes are explicitly recorded. The ONNX
checker and canonical shape/vocabulary contract run before comparing all ten
ONNX Runtime CPU outputs against PyTorch. The criterion is elementwise
abs(actual-reference) <= 0.002 + 0.002*abs(reference), not a blanket maximum
absolute error of 0.002. The upstream head comparison uses rtol/atol 1e-4. ONNX Runtime graph optimization
is disabled so the exporter validates its graph without additional CPU fusions.
`export.json` distinguishes float_checked from compiled-model, dataset and board
validation (all not-run), and records input/checkpoint/ONNX/vocabulary identities,
versions and maximum absolute errors. It has no hard-coded hardware march.

Reviewing the complete X5 source README exposed an omission in the previous
preparation increment: E11l requires a second int16 attention override,
`/model.10/m/m.1/attn/Softmax`. A new regression first failed against the old
configuration and now verifies both overrides for 11l, one for 11s/m, and a
specific warning when an expected node is absent. This corrects the prior
increment; its historical record is retained, not silently rewritten.

## Real model and host evidence

The evidence directory records actual export commands, complete stdout/stderr,
UTC times, return codes, exact implementation hashes and official release asset
identities. Checkpoints and ONNX weight files remain in ignored local storage;
they are not committed as repository artifacts. The E26 release supplies SHA-256
digests and the downloaded files are checked against them. E11 publisher digests
are unavailable, so local hashes and official source URLs are recorded without
claiming independent publisher digest verification.

See [real export records](evidence/2026-09-28-yoloe-export/real-export-results.json)
and per-variant export metadata in the same directory for the current completed
matrix. Validation uses the bundled office_desk image on CPU. These are real
checkpoint/ONNX runs, not fake compiler/SDK tests, but cover one image and do not
establish dataset accuracy or compiled hardware behavior.

Synthetic head tests additionally cover mixed Linear/Conv vocabulary branches,
source parameter preservation and rejection of incompatible heads. Entry tests
verify dependency-free help and rejection of invalid paths/options before loading.
The previous preparation tests retain real ONNX structure validation and explicitly
fake compiler tests. Bilingual conversion documentation now describes the canonical
commands, required export environment, numerical criteria, output files and actual
verification boundaries. Root navigation is synchronized.


## Final checks

All eight checkpoint exports passed the stated raw/identity/numerical checks:
E11s/m/l and E26n/s/m/l/x. All 14 concrete target preparations then passed using
those actual ONNX files and one bundled image, with no absent-attention warning;
X5 E11l matched both required Softmax nodes. This is a preparation smoke, not a
representative calibration dataset or OE compile. Full preparation commands,
YAML, calibration digests and records are retained beside the export evidence.

451 host tests passed (YOLOE 29, export-specific 5, Ultralytics 141, shared 153,
ResNet 52, OCR 44, checker 27). The migration checker passed 45 samples with zero
violations, 47 declared skips and zero exemptions. README checks passed 58 YOLOE
local links, 128 Ultralytics links, 16 Ultralytics examples, two YOLOE library
examples and both bilingual preparation-command examples. See
[host-results.json](evidence/2026-09-28-yoloe-export/host-results.json) and
[result.json](evidence/2026-09-28-yoloe-export/result.json) for counts and file hashes.

The export environment was Python 3.14.7, PyTorch 2.14.0, Ultralytics 8.4.127,
ONNX 1.23.0, ONNX Runtime 1.30.0, NumPy 2.5.3 and OpenCV 4.14.0. The shared
source requirements permit other dependency versions, but those combinations
have not been accepted here. No publisher input or native implementation changed;
publisher/native checks were not repeated in this increment.

## Findings from real checkpoints

The first full matrix caught an E11m CPU optimization difference: with the same
ONNX, optimized ORT had one box_16 element outside the original rtol/atol 2e-3;
unoptimized ORT had zero violations. A fused PyTorch diagnostic also had zero
violations against optimized ORT. The isolated optimization comparison is retained
in numerics-comparison.json and its script. The final exporter uses ORT_DISABLE_ALL,
records that choice, retains the original tolerance and makes no acceptance claim
for optimized execution. Initial failures remain under initial/.

The next matrix caught E26m/l/x Top-K order changes when Linear vocabulary math
was represented as Conv2d. Diagnostics verified all raw anchors and pre-Top-K
decoded values at the original rtol/atol 1e-4; boxes and coefficients were exact,
classification maximum absolute errors were approximately 7.25e-5/5.34e-5/5.34e-5.
All selected anchor/class sets were identical. There were 2/4/2 reordered rows,
respectively, with nonzero floating score differences: these are not exact ties.
The final check covers all raw outputs and requires the exact selected identity
set, then compares selected rows by identity while recording order drift. It
rejects every boundary substitution regardless of score proximity. A dedicated
negative test proves that rejection. This expands raw-tensor coverage without
claiming the old strict ranking criterion passed. The prior strict failures are
preserved under before-keyed-topk/, with independent diagnostics in topk-comparison.json.

During failure handling review, an attempted but failed ONNX comparison was found
to retain the initial not-run field. The exporter now marks running stages failed;
untouched stages remain not-run. The original failed record is preserved as-is.

## Outstanding scope

Canonical evaluator workflows, S native C++ migration and their bilingual customer
README layers remain required. Real OE compilation, compiled metadata, dataset
accuracy and board tests are not-run. The full H0–H9 plan, final independent review
and GitHub integration remain open; this increment does not close the user goal.
