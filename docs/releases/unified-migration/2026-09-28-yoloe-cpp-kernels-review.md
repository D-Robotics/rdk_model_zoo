# YOLOE native float binding and E26 candidate kernels

Base: `bcbcaa8719092372d7c5507147b6a0a4f246ef4b`. This implementation increment
prepares the native pipeline; it does not claim the C++ runtime is complete.
B9/H5/H0–H9 remain open, including all other migration and README work.

## Source audit and implementation

S E11 combines DFL64, classwise NMS and cropped prototype masks. S E26 uses
LTRB4, deterministic static Top-K and full-canvas mask restoration. These
algorithms are not interchangeable. Both source implementations contain manual
quantized output reading; canonical task math remains float-only as requested.

`runtime/cpp/common/float_heads.h` discovers ten unique semantic roles by shape,
requiring 4585 classes and explicit E11/E26 box geometry. It reuses Ultralytics'
existing physical FLOAT32 NHWC plan/copy helpers rather than introducing another
stride, allocation or dequantization implementation. Binding by shape alone is
not an SDK precision gate; the documented caller obligations remain explicit.

`e26_decode.h` extracts the source static candidate algorithm from the pinned S
snapshot (380e1a2bf42041af54be6f34935e50197cfadff9), preserving exact-tie ordering,
single/multi-label semantics, strict logit threshold and no-NMS behavior. The
public wrapper checks exact compact vector lengths before pointer access.
Finite inputs that overflow decoded boxes now raise an error, matching the
canonical Python boundary. Mask generation and SDK calls are not in this kernel.

The root native README in both languages describes the actual available modules,
shape/precision contract, host commands, semantics, memory costs and outstanding
integration. It links the historical source programs without misrepresenting
them as canonical float entry points.

## Verification

Two native C++17 tests run under AddressSanitizer and UndefinedBehaviorSanitizer.
They cover reversed output order, wrong class/family/count, duplicate roles,
physical row/cell padding, short allocations, quantized/nonfinite rejection,
exact-tie ordering, aligned coefficients, multi-label selection, strict threshold,
invalid caps, malformed vector lengths and finite-input box overflow. Both test
files were first compiled before their implementation headers existed, producing
missing-header failures; then compiled and executed successfully with the new
headers. This is an implementation self-check, not independent review.

The [real-output comparison](evidence/2026-09-28-yoloe-cpp-kernels/comparison.json)
uses the actual exported E26n ONNX, bundled office image and unoptimized CPU
backend. Full raw tensors were supplied to the sanitizer-instrumented native
probe. Single-label mode produced 213 candidates, multi-label mode 300; both
matched Python class IDs and order exactly. Boxes, scores and coefficients
passed rtol=atol=1e-6 (observed maximum absolute difference 0 for these checks).
Complete native JSON, stderr, argv, UTC, model/image/tensor hashes and source
hashes are retained. Raw tensors/weights remain in ignored local storage.
This proves candidate decoding for the tested inputs, not masks, geometry, SDK
compatibility, other model sizes, dataset accuracy or hardware execution.

The 29 existing YOLOE tests passed. The migration contract check finished with
45 samples, zero violations, 47 declared skips and zero exemptions; all 100
YOLOE local README links passed. Initial checks reported missing native README
section anchors, then incorrect section order; complete sections and their order
were corrected without exemptions. Both failed records are retained alongside
final passing results. Documented bilingual C++ build/run commands were extracted
and executed under sanitizers. Host test/build and README checks are captured in
the accompanying evidence.
No existing runtime behavior is routed through these native helpers yet. The
remaining integration includes E11 decoding, masks, geometry, SDK ownership,
identity gates, public stages and native application/build instructions. Board,
SDK/OE execution and whole-branch independent review remain not-run.
