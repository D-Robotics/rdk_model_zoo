# YOLOE C++ runtime migration

English | [简体中文](README_cn.md)

This directory provides a C++ stage library with owned NV12 inputs, E11/E26 decoding and ROI masks, backed by an explicitly supplied inference runner. **A complete board executable is not yet available here.** Use the [Python runtime](../python/README.md) for the implemented canonical entry, subject to its artifact/SDK requirements. Source C++ programs remain in the [S E11 snapshot](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) and [S E26 snapshot](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md); their quantized artifacts and manual dequantization do not satisfy this new float contract.

<a id="supported-boards"></a>
## Target scope

The source native capabilities are S100 E11s and S100/S100P E26 n/s/m/l/x. This increment is host-only and does not enable any board executable. S600 remains unsupported. The new SDK adapter also accepts X5 E11s/m/l with the X5 stack; this is implemented but has not been built against a real SDK or run on a board.

## Modules and model contract

| Module | Responsibility |
| --- | --- |
| `common/float_heads.h` | Bind ten logical roles by unique shape, independently of physical output order |
| `inc/yoloe.h`, `src/yoloe.cpp` | Task construction and pre_process/infer/post_process/predict orchestration |
| `inc/runner.h`, `inc/pipeline_io.h` | Backend contract, owned input/output batches and instance identity |
| `inc/sdk_runner.h`, `src/sdk_runner.cpp` | Model loading, SDK ownership and shared input/output transport after required preflight |
| `inc/config.h` | Protocol-specific configuration validation |
| `common/nv12.h` | BGR-to-I420 conversion and shared split-NV12 packing |
| `common/postprocess.h` | Family dispatch and aligned instance result assembly |
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

The caller explicitly selects E11 (64 box channels) or E26 (4); an incompatible family, missing/duplicate role, wrong vocabulary width or extra output is rejected. SDK ownership now reuses Ultralytics `PackedModelOwner`, `Nv12Input` and `TaskOutputs`; semantic YOLOE roles are validated before allocation. Quantized outputs are rejected, not manually dequantized. The existing [conversion guide](../../conversion/README.md) describes preparing float output models; no compatible S float HBM has been compiled or verified in this migration.

<a id="dependencies"></a>
## Dependencies

Prerequisites: a C++17 compiler and the repository checkout. The four geometry/candidate tests do not need OpenCV or a board SDK. The image/mask and stage tests need OpenCV C++ core/imgproc development libraries. The documented build/test commands need CMake/CTest 3.20+ (`ctest --test-dir`); set `OpenCV_DIR` to your installed OpenCV CMake package directory if it is not discoverable. Python opencv-python alone does not provide this C++ development environment. From the repository root:

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

Build all eight tests with a real OpenCV installation (no board SDK):

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-opencv \
  -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-opencv --parallel 4
```

Build the reusable stage library and all eight tests from its own CMake project:

```bash
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-stage-core \
  -DYOLOE_BUILD_TESTS=ON -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-stage-core --parallel 4
```

The output is `libyoloe_core.a`, not a board executable. For a normal embedding build, omit `YOLOE_BUILD_TESTS` and `YOLOE_SANITIZERS` (both default OFF in the library project). A consumer can add this directory with CMake `add_subdirectory` and link `yoloe_core`; include paths and OpenCV dependencies are propagated. `YOLOE_TEST_OPENCV` controls the standalone test project; it does not remove OpenCV from the stage library.

<a id="run"></a>
## Run host tests

```bash
/tmp/yoloe-native-tests/float-heads
/tmp/yoloe-native-tests/e26-decode
/tmp/yoloe-native-tests/e11-decode
/tmp/yoloe-native-tests/geometry
```

Successful tests exit 0 with no output. The decoder test allocates the full 4585-class tensor geometry, so allow several hundred MB with sanitizers. A thrown assertion/contract error or sanitizer diagnostic is a failure. These commands compile actual C++ math and float-memory utilities; they do not establish SDK ABI compatibility.

For the CMake build, run all eight checks with failure output:

```bash
ctest --test-dir /tmp/yoloe-native-opencv --output-on-failure
```

Run the library project's eight tests:

```bash
ctest --test-dir /tmp/yoloe-stage-core --output-on-failure
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

`YOLOE` exclusively owns a `std::unique_ptr<Runner>`. Construction validates configuration and backend protocol; invalid construction releases the supplied backend. The backend must return ten independently owned compact semantic FLOAT32 vectors. It must perform hardware/artifact identity and SDK metadata validation before exposing inference; the base interface itself is not proof of a real SDK implementation. `SdkRunner` implements the low-level SDK boundary described below. Test runners remain explicit host fixtures.

`pre_process` returns an owned compact Y plane (409600 bytes) and interleaved UV plane (204800 bytes), plus actual geometry. `infer` invokes the runner exactly once and returns owned raw outputs carrying that geometry. `post_process` returns aligned `Instance` values with box/score/label/ROI mask. Prepared/raw batches from another task are rejected, including another task with the same protocol; do not cache a last-image geometry yourself. Raw outputs remain valid across subsequent calls, and result masks do not borrow SDK buffers. Use one task per inference thread; concurrent backend use is not guaranteed.

The following function is compiled in host verification. The application must supply an actual matching backend; it does not download or synthesize a model:

```cpp
#include "yoloe.h"
yoloe::Result process_image(yoloe::Config config,
                            std::unique_ptr<yoloe::Runner> backend,
                            const cv::Mat& image) {
    yoloe::YOLOE task(config, std::move(backend));
    auto prepared = task.pre_process(image);
    auto raw = task.infer(prepared);
    return task.post_process(raw);
    // task.predict(image) composes the same three operations.
}
```

Configuration defaults to E11, score 0.25, NMS unset (E11 resolves 0.7), morphology off, letterbox, max_det 300 and single_label true. E11 rejects E26-only overrides; E26 rejects an explicitly supplied NMS threshold, morphology or stretch. Select `config.protocol = yoloe::Protocol::E26` only with a matching backend. These values describe library behavior; source demo CLI defaults may differ.

The numerical headers expose pure functions and own no SDK resources. `bind_heads` returns output indices; it does not retain references. `decode_e11` and `decode_e26` borrow their ten input vectors only for the duration of the call and return owning detections with copied boxes/scores/coefficients. Callers may release input tensors after return. Internal pointer views never escape. OpenCV result matrices use reference-counted owned storage and do not alias the caller's image/prototype; retain a returned result for as long as its pixels are needed. Invalid arguments throw `std::invalid_argument`; allocation failures may propagate. There is no model loading, implicit hardware selection or hidden cross-call geometry state.

## SDK backend library

`SdkRunner(SdkModel, SdkPreflight)` owns exactly one model and its input/output
allocations. `SdkModel` contains `path`, `target` and `variant`. Allowed pairs are
X5 E11s/m/l, S100 E11s/E26n/s/m/l/x and S100P E26n/s/m/l/x; the compiled SDK stack
must match. It checks a nonempty file, one named model, target-specific 640×640
NV12 input and ten finite, unquantized FLOAT32 NHWC output roles. Output order
may vary; returned vectors always use the semantic order above. No SDK buffer
escapes the adapter. Destruction frees outputs, inputs, then the packed model;
failed construction also releases anything already acquired. Inference uploads
Y/UV directly, cleans input caches, submits/waits/releases the task, invalidates
output caches and copies output data. It performs no decode or image rendering.

**Preflight is mandatory and has no default.** The application supplies a
`void(const SdkModel&)` callback that verifies the actual board, selected
publication or custom float SHA-256, and vocabulary/conversion provenance. It
runs before any SDK call; throwing stops construction. The adapter's shape,
target/variant and stack checks cannot prove those identities. A no-op callback
is used only by host fixtures and is not a valid production policy. The
canonical launcher supplying this policy is still pending; this API alone is
not a ready-to-run customer entry.

The second complete API example also compiles in host verification. Its caller
must supply the described policy, not a placeholder that silently accepts:

```cpp
#include "sdk_runner.h"
#include "yoloe.h"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                yoloe::SdkPreflight preflight) {
    auto backend = std::make_unique<yoloe::SdkRunner>(model, std::move(preflight));
    yoloe::Config config;
    config.protocol = backend->protocol();
    yoloe::YOLOE task(config, std::move(backend));
    return task.predict(image);
}
```

To build the SDK library on a matching board SDK installation (not executed in
this host-only round):

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-lib \
  -DYOLOE_BUILD_SDK=ON
cmake --build /tmp/yoloe-board-lib --parallel 4
```

This produces `libyoloe_core.a` and `libyoloe_sdk.a`, not an executable. With
CMake `add_subdirectory`, link the application to `yoloe_sdk`. Set
`YOLOE_DNN_INCLUDE_DIR` to the directory containing `dnn/hb_dnn.h` and
`YOLOE_DNN_LIBRARY` to the matching DNN library if discovery fails. UCP headers
also require `YOLOE_UCP_LIBRARY`; exposing both hbSys and UCP headers is rejected.
Do not use test-double include paths for a deployable library. OpenCV development
files are still required. The default host build leaves `YOLOE_BUILD_SDK=OFF`.

The OpenCV host configuration now runs eight tests: the existing six plus X5
and UCP adapter tests with production code instrumented by ASan/UBSan. They
cover preflight rejection before SDK calls, metadata/precision rejection before
allocation, partial allocation and failed initialization cleanup, task/cache
errors, semantic output order and independence across inference calls. These
use narrow API doubles, not vendor SDK headers/libraries.

<a id="results-interpretation"></a>
## Verification and next integration

[SDK adapter evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-sdk-runner-review.md) records resource/metadata tests and their host-only limits.

[Stage/NV12 evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-stages-review.md) covers actual byte comparisons, ownership/error paths and compiled API examples; it does not certify a board backend.

[Implementation evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md) records the earlier E26 checks. The [E11 extension evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md) records all three native tests, E26 regression and real E11 s/m/l ONNX comparisons. The E26n candidate comparison covers both single- and multi-label decoding against Python. [Geometry/mask evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md) separately records actual OpenCV compilation and full ROI pixel comparison. Labels/order are compared exactly; boxes, scores and coefficients use stated numerical tolerances. Masks are not compared by this candidate-only test.

The remaining native work is explicit: a canonical implementation of the target/artifact preflight policy, CLI entry and complete executable build/run documentation. The adapter API does not close those requirements. Board, real SDK, OE compilation and native dataset accuracy remain unverified. This directory is not a completed C++ migration or a replacement for the archived source programs yet.
