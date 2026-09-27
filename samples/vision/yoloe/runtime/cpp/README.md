# YOLOE C++ runtime migration

English | [简体中文](README_cn.md)

This directory currently contains the reusable float-output binding, E11/E26 candidate decoders, BGR geometry and E11/E26 ROI mask restoration for the canonical native runtime. **A complete board executable is not yet available here.** Use the [Python runtime](../python/README.md) for the implemented canonical entry, subject to its artifact/SDK requirements. Source C++ programs remain in the [S E11 snapshot](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) and [S E26 snapshot](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md); their quantized artifacts and manual dequantization do not satisfy this new float contract.

<a id="supported-boards"></a>
## Target scope

The source native capabilities are S100 E11s and S100/S100P E26 n/s/m/l/x. This increment is host-only and does not enable any board executable. S600 remains unsupported. X5 has E11 Python publications; the common shape binder does not create an X5 native implementation.

## Modules and model contract

| Module | Responsibility |
| --- | --- |
| `common/float_heads.h` | Bind ten logical roles by unique shape, independently of physical output order |
| `common/geometry.h` | Explicit E11/E26 resize geometry and actual-scale inverse boxes |
| `common/image_ops.h` | OpenCV BGR preparation and E11/E26 ROI mask restoration |
| `common/candidate.h` | Shared owning candidate result for both families |
| `common/e11_decode.h` | DFL16, classwise NMS and aligned E11 coefficients |
| `common/e26_decode.h` | Select E26 PF candidates, decode LTRB boxes and retain aligned mask coefficients |
| `tests/test_float_heads.cc` | Shape/precision/stride/allocation and finite-value boundary tests |
| `tests/test_e11_decode.cc` | DFL, same/different-class suppression, score/IoU equality and malformed outputs |
| `tests/test_e26_decode.cc` | Candidate ordering, single/multi-label selection, thresholds and invalid outputs |
| `tests/decode_probe.cc` | Host-only utility for comparison with saved raw float tensors; not an inference application |

Float reads reuse Ultralytics' `common/task_outputs.h`: `nhwc_float_plan` requires unquantized FLOAT32, batch-one NHWC, valid physical row/cell strides and a sufficient allocation. `copy_float_output` removes physical padding and rejects nonfinite values. `bind_heads` only identifies logical roles; shape matching by itself does not prove dtype, memory safety, model family or vocabulary identity.

| Role | Shape at stride 8 / 16 / 32 |
| --- | --- |
| Class logits | `[1,80,80,4585]` / `[1,40,40,4585]` / `[1,20,20,4585]` |
| E11 DFL boxes | Same spatial shapes, 64 channels |
| E26 direct LTRB boxes | Same spatial shapes, 4 channels |
| Mask coefficients | Same spatial shapes, 32 channels |
| Prototype | One `[1,160,160,32]` tensor |

The caller explicitly selects E11 (64 box channels) or E26 (4); an incompatible family, missing/duplicate role, wrong vocabulary width or extra output is rejected. SDK input/output ownership is not implemented in this directory yet. The existing [conversion guide](../../conversion/README.md) describes preparing float output models; no compatible S float HBM has been compiled or verified in this migration.

<a id="dependencies"></a>
## Dependencies

Prerequisites: a C++17 compiler and the repository checkout. The four geometry/candidate tests do not need OpenCV or a board SDK. The fifth image/mask test needs OpenCV C++ core/imgproc development libraries. The documented build/test commands need CMake/CTest 3.20+ (`ctest --test-dir`); set `OpenCV_DIR` to your installed OpenCV CMake package directory if it is not discoverable. Python opencv-python alone does not provide this C++ development environment. From the repository root:

<a id="build"></a>
## Build host tests

```bash
mkdir -p /tmp/yoloe-native-tests
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  -I samples/vision/ultralytics_yolo/runtime/cpp \
  samples/vision/yoloe/runtime/cpp/tests/test_float_heads.cc \
  -o /tmp/yoloe-native-tests/float-heads
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  samples/vision/yoloe/runtime/cpp/tests/test_e26_decode.cc \
  -o /tmp/yoloe-native-tests/e26-decode
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  -I samples/vision/ultralytics_yolo/runtime/cpp \
  samples/vision/yoloe/runtime/cpp/tests/test_e11_decode.cc \
  -o /tmp/yoloe-native-tests/e11-decode
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  samples/vision/yoloe/runtime/cpp/tests/test_geometry.cc \
  -o /tmp/yoloe-native-tests/geometry
```

Build all five tests with a real OpenCV installation (no board SDK):

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-opencv \
  -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-opencv --parallel 4
```

<a id="run"></a>
## Run host tests

```bash
/tmp/yoloe-native-tests/float-heads
/tmp/yoloe-native-tests/e26-decode
/tmp/yoloe-native-tests/e11-decode
/tmp/yoloe-native-tests/geometry
```

Successful tests exit 0 with no output. The decoder test allocates the full 4585-class tensor geometry, so allow several hundred MB with sanitizers. A thrown assertion/contract error or sanitizer diagnostic is a failure. These commands compile actual C++ math and float-memory utilities; they do not establish SDK ABI compatibility.

For the CMake build, run all five checks with failure output:

```bash
ctest --test-dir /tmp/yoloe-native-opencv --output-on-failure
```

<a id="parameters"></a>
## Candidate decoding

`decode_e11` accepts the same semantic arrangement with 64 box channels. It reuses Ultralytics' numerically stabilized 16-bin DFL expectation and sigmoid. Defaults are score 0.25 and classwise NMS IoU 0.7; score must be in `(0,1)`, NMS in `[0,1]`. It keeps one class per anchor and applies no E26 candidate cap. Coefficients travel with each candidate through NMS.

The original native E11 boundaries are retained: score **greater than or equal to** the threshold is accepted; a same-class box is suppressed only when IoU is **greater than** the NMS threshold. Python's existing NMS also suppresses equality, so exact-boundary equivalence is not claimed. Native output is ordered by ascending class, then descending score; exact score ties prefer the original scale/anchor index. This makes source unordered-map/OpenMP merging deterministic, but does not reproduce its unspecified tie order or NumPy's tie ordering. Finite tensor values and exact lengths are validated before decoding, including the prototype not yet consumed by candidate math.

`decode_e26` accepts ten finite compact float vectors in semantic order: class/box/coefficients for each stride, then prototype. Every vector length is checked before reading. Defaults are score threshold 0.25, maximum 300 candidates and one class per anchor. The valid threshold range is `(0,1)` and the candidate cap is `1..8400`.

Selection preserves the source static Top-K contract. Exact ties prefer lower scale, anchor, then class; multi-label expansion breaks ties by the selected anchor rank and class. The raw threshold comparison is strict. Boxes are decoded in the 640×640 model canvas, scores use sigmoid, and mask coefficients remain aligned. No IoU NMS, geometry restoration, mask generation or dequantization occurs in this module. Nonfinite decoded boxes are rejected, including finite input distances that overflow during scaling. Multi-label expansion currently uses memory proportional to the selected anchor count times 4585 classes; large caps are materially more expensive than the default.

`prepare_bgr` returns owned 640×640 BGR pixels and explicit geometry. E11 letterbox truncates resized dimensions and pads with 127; its optional stretch uses nearest-neighbor. E26 letterbox uses ties-to-even rounding and padding 114, rejecting stretch. Both letterbox paths use linear interpolation and clamp tiny resized dimensions to at least one pixel. Inverse boxes use actual horizontal/vertical scales, correcting the archived native ideal-gain reconstruction on rounded dimensions.

`restore_e26_masks` combines prototype logits and coefficients, checks finite sums, linearly resizes logits to 640×640, thresholds at zero and crops in model coordinates, removes the recorded padding, uses nearest-neighbor to restore source dimensions, then copies the clipped integer-truncated ROI. It returns owning float boxes and `CV_8UC1` masks with values 0/1, preserving empty/degenerate instance slots. Reversed/nonfinite boxes, malformed geometry, wrong prototype length, nonfinite coefficients/prototypes or arithmetic overflow fail. It has no sigmoid, NMS, morphology or dequantization.

`restore_e11_masks` implements the S11 ROI protocol: clip the model box to actual image content, truncate its bounds at prototype scale, combine raw prototype values and coefficients, threshold strictly above 0.5, resize the binary crop with Lanczos4, and optionally apply a 5×5 rectangular opening (`do_morph=false` by default for the library). Final positive values are normalized to 1 because Lanczos can overshoot uint8 binary data to 2. Empty boxes retain exact zero-sized axes. These last two corrections also apply to the shared Python DFL ROI helper; normal foreground support is unchanged. This is distinct from X5 Python's full-image probability-mask path. [E11 mask evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-masks-review.md) compares both morphology settings with identical real native candidate inputs.

<a id="interface-lifecycle"></a>
## Interface and lifetime

The numerical headers expose pure functions and own no SDK resources. `bind_heads` returns output indices; it does not retain references. `decode_e11` and `decode_e26` borrow their ten input vectors only for the duration of the call and return owning detections with copied boxes/scores/coefficients. Callers may release input tensors after return. Internal pointer views never escape. OpenCV result matrices use reference-counted owned storage and do not alias the caller's image/prototype; retain a returned result for as long as its pixels are needed. Invalid arguments throw `std::invalid_argument`; allocation failures may propagate. There is no model loading, implicit hardware selection or hidden cross-call geometry state.

<a id="results-interpretation"></a>
## Verification and next integration

[Implementation evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md) records the earlier E26 checks. The [E11 extension evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md) records all three native tests, E26 regression and real E11 s/m/l ONNX comparisons. The E26n candidate comparison covers both single- and multi-label decoding against Python. [Geometry/mask evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md) separately records actual OpenCV compilation and full ROI pixel comparison. Labels/order are compared exactly; boxes, scores and coefficients use stated numerical tolerances. Masks are not compared by this candidate-only test.

The remaining native work is explicit: NV12 packing, SDK resource ownership, target and artifact identity gates, public pre/infer/post/predict stages, library/CLI entry, and complete board build/run documentation. Board, real SDK, OE compilation and native dataset accuracy remain unverified. This directory is not a completed C++ migration or a replacement for the archived source programs yet.
