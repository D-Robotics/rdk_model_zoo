# YOLOE native geometry and E26 mask restoration

Base: `6d45800b5808ed64b02f6df5325acc0b641973d4`. This increment adds actual
OpenCV image preparation and E26 ROI masks to the native mathematical modules.
It is an implementation self-check. The complete native entry, E11 mask path,
SDK resource management and identity gates are still pending; H0–H9 remain open.

## Implementation

`geometry.h` records original/resized dimensions, all four padding extents,
protocol and resize mode. E11 uses truncated letterbox dimensions, 127 padding
and optional nearest-neighbor stretch; E26 uses ties-to-even rounded dimensions,
114 padding and letterbox only. Both ensure a minimum resized dimension of one.
`restore_box` uses actual per-axis scales, matching the canonical Python correction
rather than reconstructing the ideal gain from the archived native implementation.
Geometry is revalidated before inversion, rejecting a mismatched/modified context.

`image_ops.h` owns image-library operations, separate from candidate math and
future inference stages. E26 combines prototype logits and coefficients, checks
for overflow, linearly resizes logits, thresholds at zero, crops in model space,
unpads, restores binary pixels with nearest-neighbor, and copies the original
box ROI. It retains 0/1 uint8 masks and independent storage, including explicit
zero-sized dimensions. No sigmoid/NMS/morphology/dequantization is introduced.

## Actual C++ image-library environment

The host initially lacked CMake and C++ OpenCV development files. CMake was
installed only in ignored task-local dependencies. Official OpenCV tag 4.14.0
(the installed Python cv2 version) was downloaded and built locally with
core/imgproc/imgcodecs, without tests/examples/Java/Python modules or accelerator
backends requested by this build. No system package or board was modified.
Source URL/archive digest, configure/build/install logs and static-library hashes
are retained in the [evidence](evidence/2026-09-28-yoloe-cpp-masks/).
The C++ tests use actual OpenCV rather than a stub. ASan/UBSan instrument sample
test code; the separately built release OpenCV library itself is not sanitized.
The static link emitted a duplicate-core-library warning, retained in build logs.

## Validation and discovered fix

Five native C++ tests cover float binding, E11/E26 candidates, geometry and
OpenCV pixels/masks. The new geometry test was first compiled without its header
and failed, then passed after implementation. Image tests exercise real padding,
round/truncate differences, stretch, tiny images, invalid/forged contexts,
positive/negative prototypes, degenerate ROIs and combination overflow.

The real comparison consumes hash-verified E26n ONNX outputs from the preceding
candidate evidence. It runs all 213 single-label candidates on the bundled office
image, comparing restored boxes and full ROI pixels against canonical Python.
The first comparison failed at instance 16: native empty shape `(0,0)` versus
Python `(0,58)`. No later-instance claim is made from that aborted run. The initial
native records and source are retained under `initial-empty-shape/`; the fix
constructs the empty Mat with its explicit clipped dimensions, with a regression
fixture. After rebuilding, all 213 mask shapes/pixels match exactly and restored
box maximum difference is zero. The comparison criterion was not relaxed.
Both native and Python mask arrays are persisted in compressed NPZ along with
per-mask hashes, commands, timestamps, model-output reference and image identity.

The test CMake project can build four SDK-free tests without OpenCV, or all five
with `YOLOE_TEST_OPENCV=ON`; README commands, dependencies, ownership, protocol
rules and remaining integration are documented in both languages.

Final verification: all five native tests and 29 existing YOLOE Python tests
passed. The migration contract check reported 45 samples, zero violations,
47 declared skips and zero exemptions; all 104 YOLOE local README links passed.
The complete bilingual build/run blocks were extracted and executed against the
local OpenCV installation. CMake/CTest 3.20+ is documented because `--test-dir`
was introduced in 3.20 (verified against installed official CTest help).

## Scope

This proves the tested host geometry/mask behavior, not all inputs, quantized HBM
equivalence, dataset accuracy, real SDK/OE compilation or board execution. The
native C++ entry is still in progress. E11 masks, NV12 packing, SDK ownership,
identity gates, remaining samples/README work and independent review continue.

Raw `opencv-configure.log` contains upstream-generated trailing whitespace and
is retained byte-for-byte. The final diff whitespace check excludes that one
evidence log; no source/document rule or repository-wide exemption is changed.
