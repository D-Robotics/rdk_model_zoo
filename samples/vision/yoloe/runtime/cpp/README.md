# YOLOE C++ runtime migration

English | [简体中文](README_cn.md)

This directory currently contains the reusable float-output binding and E26 candidate decoder for the canonical native runtime. **A complete board executable is not yet available here.** Use the [Python runtime](../python/README.md) for the implemented canonical entry, subject to its artifact/SDK requirements. Source C++ programs remain in the [S E11 snapshot](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) and [S E26 snapshot](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md); their quantized artifacts and manual dequantization do not satisfy this new float contract.

<a id="supported-boards"></a>
## Target scope

The source native capabilities are S100 E11s and S100/S100P E26 n/s/m/l/x. This increment is host-only and does not enable any board executable. S600 remains unsupported. X5 has E11 Python publications; the common shape binder does not create an X5 native implementation.

## Modules and model contract

| Module | Responsibility |
| --- | --- |
| `common/float_heads.h` | Bind ten logical roles by unique shape, independently of physical output order |
| `common/e26_decode.h` | Select E26 PF candidates, decode LTRB boxes and retain aligned mask coefficients |
| `tests/test_float_heads.cc` | Shape/precision/stride/allocation and finite-value boundary tests |
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

The caller explicitly selects E11 (64 box channels) or E26 (4); an incompatible family, missing/duplicate role, wrong vocabulary width or extra output is rejected. E11 candidate/mask decoding and SDK input/output ownership are not implemented in this directory yet. The existing [conversion guide](../../conversion/README.md) describes preparing float output models; no compatible S float HBM has been compiled or verified in this migration.

<a id="dependencies"></a>
## Dependencies

Prerequisites: a C++17 compiler and the repository checkout. The two unit tests do not need OpenCV or a board SDK. From the repository root:

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
```

<a id="run"></a>
## Run host tests

```bash
/tmp/yoloe-native-tests/float-heads
/tmp/yoloe-native-tests/e26-decode
```

Successful tests exit 0 with no output. The decoder test allocates the full 4585-class tensor geometry, so allow several hundred MB with sanitizers. A thrown assertion/contract error or sanitizer diagnostic is a failure. These commands compile actual C++ math and float-memory utilities; they do not establish SDK ABI compatibility.

<a id="parameters"></a>
## Candidate decoding

`decode_e26` accepts ten finite compact float vectors in semantic order: class/box/coefficients for each stride, then prototype. Every vector length is checked before reading. Defaults are score threshold 0.25, maximum 300 candidates and one class per anchor. The valid threshold range is `(0,1)` and the candidate cap is `1..8400`.

Selection preserves the source static Top-K contract. Exact ties prefer lower scale, anchor, then class; multi-label expansion breaks ties by the selected anchor rank and class. The raw threshold comparison is strict. Boxes are decoded in the 640×640 model canvas, scores use sigmoid, and mask coefficients remain aligned. No IoU NMS, geometry restoration, mask generation or dequantization occurs in this module. Nonfinite decoded boxes are rejected, including finite input distances that overflow during scaling. Multi-label expansion currently uses memory proportional to the selected anchor count times 4585 classes; large caps are materially more expensive than the default.

<a id="interface-lifecycle"></a>
## Interface and lifetime

The two headers expose pure functions and own no SDK resources. `bind_heads` returns output indices; it does not retain references. `decode_e26` borrows its ten input vectors only for the duration of the call and returns owning detections with copied boxes/scores/coefficients. Callers may release input tensors after return. Internal pointer views never escape. Invalid arguments throw `std::invalid_argument`; allocation failures may propagate. There is no model loading, implicit hardware selection or hidden cross-call geometry state.

<a id="results-interpretation"></a>
## Verification and next integration

[Implementation evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md) records native host tests and real E26n ONNX output comparisons against Python for single- and multi-label decoding. Labels/order are compared exactly; boxes, scores and coefficients use stated numerical tolerances. Masks are not compared by this candidate-only test.

The remaining native work is explicit: E11 decoding, mask restoration, image/NV12 geometry, SDK resource ownership, target and artifact identity gates, public pre/infer/post/predict stages, library/CLI entry, and complete board build/run documentation. Board, real SDK, OE compilation and native dataset accuracy remain unverified. This directory is not a completed C++ migration or a replacement for the archived source programs yet.
