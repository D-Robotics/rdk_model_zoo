# YOLOE C++ runtime

English | [简体中文](README_cn.md)

Run E11/E26 prompt-free instance segmentation through `run.sh`, or embed the C++ three-stage library. The launcher selects an exact model, checks the local board and file identity, optionally builds the native executable, and retains logs and image/mask results. **The implementation has host verification only: real SDK compilation and board inference remain not-run.** See the [Python runtime](../python/README.md) for the other canonical entry and its different X5 mask protocol.

<a id="supported-boards"></a>
## Target scope

| Target | Variants / default | Model requirement | Current validation |
| --- | --- | --- | --- |
| X5 | 11s/m/l; default 11s | Published floating-output BIN, matching X5 SDK | Host only; SDK/board not-run |
| S100 | 11s, 26n/s/m/l/x; default 11s | Locally converted floating-output HBM with explicit SHA-256 | Host only; compatible HBM/SDK/board not verified |
| S100P | 26n/s/m/l/x; default 26n | Locally converted floating-output HBM with explicit SHA-256 | Host only; compatible HBM/SDK/board not verified |
| S600 | None | No published route; explicitly rejected | Unsupported |

The 14 publication identities describe model selection, **not 14 runnable native artifacts**. Published S files have quantized outputs and are rejected by this entry. Use the [conversion workflow](../../conversion/README.md) to prepare floating outputs; renaming a quantized file does not change its contract. Source S E11 (historical `../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) and S E26 (historical `../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) programs remain historical references, with their original capabilities and measurements.

## Select, build and run

Commands below start at the repository root. Python 3.10+ and NumPy are required for the launcher; Python OpenCV is not needed by this native entry. `PYTHON` selects the interpreter used by `run.sh`. It also works from another working directory. Relative user paths and the default output directory are resolved from the caller's directory; child processes run from the repository root.

These host-safe commands inspect the active manifests and prepare a command without downloading, building, checking a board or executing inference:

```bash
bash samples/vision/yoloe/runtime/cpp/run.sh --list-models
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s --dry-run
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n --dry-run
```

A dry-run requires an explicit target. Its `executed`, `downloaded` and `runtime_metadata_verified` fields stay false. S publications report that a local floating-output conversion is required; dry-run success is not permission to execute that quantized file.

On an X5 with its matching SDK and C++ dependencies, explicitly download the model, then build and run. These board commands were **not executed** in this host-only migration:

```sh
python3 samples/vision/yoloe/model/download.py --target x5 --variant 11s
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s \
  --build --output outputs/yoloe_cpp_x5_11s_run1
```

For S, first complete and retain the conversion evidence. Supply the expected digest from that evidence, not a digest invented to bypass validation. Replace both placeholders below with the actual artifact path and its 64 hexadecimal SHA-256 digits:

```sh
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n \
  --model-path /path/to/converted-float.hbm \
  --local-float-sha256 REPLACE_WITH_EXPECTED_64_HEX_SHA256 \
  --build --output outputs/yoloe_cpp_s100p_26n_run1
```

The SHA binds local bytes; it does not prove conversion provenance or SDK compatibility. All ten output tensors must still pass runtime metadata checks. No compatible S floating-output HBM is supplied here. The launcher never downloads implicitly and has no hardware-identity override or silent target fallback.

`--build` configures Release mode and creates `runtime/cpp/build/<target>/yoloe_demo`; without it, the existing executable at that path is used. Use `--binary /absolute/path/yoloe_demo` for a separately built executable; it cannot be combined with `--build`. To configure SDK discovery explicitly on a matching SDK installation:

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-cli \
  -DYOLOE_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release \
  -DYOLOE_DNN_INCLUDE_DIR=/path/to/sdk/include \
  -DYOLOE_DNN_LIBRARY=/path/to/sdk/lib/libdnn.so
cmake --build /tmp/yoloe-board-cli --parallel 2
```

For UCP, also provide `YOLOE_UCP_LIBRARY` when discovery cannot find the matching `libhbucp`. Use `--binary /tmp/yoloe-board-cli/yoloe_demo` in the selected run command. The normal launcher verifies identity and inputs before building. Missing vendor dependencies fail explicitly; host test doubles must never be used as deployment dependencies.

## Launcher parameters

| Option | Default / effect |
| --- | --- |
| `--target` | `auto`; detected local board for execution, explicit target required for dry-run |
| `--variant`, `--asset-id` | Target defaults above; an asset ID is the exact active manifest reference, conflicting choices fail |
| `--model-path` | Selected model's canonical path; override for an existing file |
| `--local-float-sha256` | Custom floating-output digest; requires `--model-path` |
| `--test-img` | Sample `test_data/office_desk.jpg`; decoded as BGR |
| `--label-file` | Sample `test_data/classes.names`; fixed ordered 4585 labels, exact digest required |
| `--output` | `outputs/yoloe_cpp`; must be a new directory, never overwrites an earlier run |
| `--build`, `--binary` | Explicit build or separately built native executable; mutually exclusive |
| `--score-thres` | 0.25, strictly between 0 and 1 |
| `--nms-thres` | E11 only, default 0.7 in [0,1]; E26 rejects even an explicit default |
| `--resize-type` | 1 = letterbox; E11 also permits 0 = stretch; E26 requires 1 |
| `--max-det` | E26 only, 300 by default, range 1..8400; E11 accepts only the unchanged default |
| `--multi-label` | E26 only; default one class per selected anchor |
| `--no-morph` | Disable S E11 CLI's default 5×5 opening; X5 E11/E26 already disable morphology |
| `--no-contour` | Omit contour drawing; masks and detections are unchanged |
| `--list-models`, `--dry-run` | Read-only inspection modes; mutually exclusive |

The native binary additionally requires explicit `--target`, `--variant`, `--model-path`, `--model-sha256`, `--test-img`, `--label-file` and `--output`. Normally let the launcher supply these. The native `--model-sha256` is the verified byte digest, not a publication selector. Run the binary with `--help` alone for its options. No CPU/BPU-core or scheduling-priority option is exposed; the SDK adapter uses its default scheduling contract.

## Results and failure records

```text
outputs/yoloe_cpp_x5_11s_run1/
  launch-report.json
  configure.stdout.log / configure.stderr.log   # only with --build
  build.stdout.log / build.stderr.log           # when configuration succeeds
  native.stdout.log / native.stderr.log
  result/
    report.json
    annotated.png
    masks/000000.png ...
```

`launch-report.json` records the selected publication reference, local/published model kind, expected/observed digests, effective options, binary digest, exact subprocess argv/cwd, UTC times, return codes and result-file digests. Raw stdout/stderr bytes are retained, including non-UTF-8 output. An observed digest is not publisher authentication when the manifest has no checksum; `publisher_checksum_verified` remains false. For custom models, the publication ID names the architecture reference, not the provenance of the new file.

Native `report.json` uses schema `rdk-model-zoo/yoloe-native-run/v1`, with target/variant, model/image/vocabulary hashes, image shape, effective configuration and aligned instances. Each instance contains a zero-based class ID, fixed label, score, original-image `[x1,y1,x2,y2]` box and ROI mask path/shape. Stored PNG masks use 0/255; in-memory masks use 0/1. Zero-area instances keep their exact empty axes and `mask: null`, without an empty PNG. `annotated.png` is a presentation overlay, not an evaluation mask. **All native masks use ROI layout, including X5; Python X5 uses full-image probability masks.** Do not interchange their evaluation inputs without adapting the protocol.

The native report is written last. A zero exit without a valid identity-matched report and its files is a launcher failure. Host fixtures label their reports `host-fixture`; the normal launcher rejects them as SDK evidence. Metadata verification is recorded only after a successful native result. Build/inference/result-validation failures retain logs and a failed launch record; image saving can leave partial output. Preflight failures before output-directory creation print an error without creating a run folder. Pick a new output path for a retry. Native validation errors exit 2; the launcher retains a positive native exit code and maps signal termination to 2.

## Modules and model contract

| Module | Responsibility |
| --- | --- |
| `launcher.py`, `run.sh` | Publication selection, explicit build, process logs and run records |
| `src/main.cpp`, `src/cli_options.cpp`, `src/cli_io.cpp` | CLI orchestration, option parsing, image/label I/O and saved results |
| `common/float_heads.h` | Bind ten logical roles by unique shape, independently of physical output order |
| `inc/yoloe.h`, `src/yoloe.cpp` | Task construction and pre_process/infer/post_process/predict orchestration |
| `inc/runner.h`, `inc/pipeline_io.h` | Backend contract, owned input/output batches and instance identity |
| `inc/sdk_runner.h`, `src/sdk_runner.cpp` | Model loading, SDK ownership and shared input/output transport after required preflight |
| `inc/model_identity.h`, `inc/preflight.h`, `src/preflight.cpp` | Explicit model selection and local board/model/vocabulary verification using shared native helpers |
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

Prerequisites: a C++17 compiler and the repository checkout. The four geometry/candidate tests do not need OpenCV or a board SDK. The image/mask and stage tests need OpenCV C++ core/imgproc/imgcodecs development libraries. The documented build/test commands need CMake/CTest 3.20+ (`ctest --test-dir`); set `OpenCV_DIR` to your installed OpenCV CMake package directory if it is not discoverable. Python opencv-python alone does not provide this C++ development environment. From the repository root:

On macOS, sanitizer builds of the OpenCV-enabled test project (`YOLOE_TEST_OPENCV=ON` with `YOLOE_SANITIZERS=ON`) additionally resolve the real TBB that OpenCV was built against through CMake's standard TBB CONFIG package and link it directly to the OpenCV-linked test executables. Under ASan, an executable that loads libtbb only transitively through `libopencv_core` aborts at process exit in `tbb::detail::r1::__TBB_InitOnce::~__TBB_InitOnce`; a minimal empty `main` linked against OpenCV core alone reproduces the crash. The direct TBB reference keeps ASan+UBSan enabled; if CMake cannot discover the package, point `CMAKE_PREFIX_PATH` at the prefix that installed TBB. Linux and OpenCV-off builds take no such branch and need no TBB. This notes host sanitizer behavior only; no board verification is implied.

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

Build all eleven tests with a real OpenCV installation (no board SDK):

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-opencv \
  -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-opencv --parallel 4
```

Build the reusable stage library and all eleven tests from its own CMake project:

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

For the CMake build, run all eleven checks with failure output:

```bash
ctest --test-dir /tmp/yoloe-native-opencv --output-on-failure
```

Run the library project's eleven tests:

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
is used only by host fixtures and is not a valid production policy. The built-in `make_preflight(expected_model_sha256, label_path)` supplies local
board and byte verification. The canonical launcher selects the publication or
custom float file before invoking the executable with its verified digest.

`make_preflight` reads real local sysfs/device-tree using shared target rules.
It rejects unknown/mismatched boards, missing/empty models, malformed or
mismatching model digests and any vocabulary other than the fixed ordered PF
file. The 64-digit expected model digest is supplied explicitly, not silently
computed from the same file and accepted. Model target/variant/SDK-stack and
tensor contracts are additionally checked by `SdkRunner`. No identity override
is exposed. For custom conversion, a matching digest proves bytes, not compiler
provenance; the caller must retain conversion evidence. No compatible S float
HBM is supplied by this API.

The second complete API example compiles in host verification. Its caller
supplies the selected model's expected digest and the vocabulary path:

```cpp
#include "preflight.h"
#include "sdk_runner.h"
#include "yoloe.h"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                const std::string& expected_model_sha256,
                                const std::string& label_path) {
    auto gate = yoloe::make_preflight(expected_model_sha256, label_path);
    auto backend = std::make_unique<yoloe::SdkRunner>(model, std::move(gate));
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

This produces `libyoloe_core.a`, `libyoloe_preflight.a` and `libyoloe_sdk.a`, not an executable. With
CMake `add_subdirectory`, link the application to `yoloe_sdk`. Set
`YOLOE_DNN_INCLUDE_DIR` to the directory containing `dnn/hb_dnn.h` and
`YOLOE_DNN_LIBRARY` to the matching DNN library if discovery fails. UCP headers
also require `YOLOE_UCP_LIBRARY`; exposing both hbSys and UCP headers is rejected.
Do not use test-double include paths for a deployable library. OpenCV development
files are still required. The default host build leaves `YOLOE_BUILD_SDK=OFF`.

The OpenCV host configuration runs eleven tests: six numerical/stage checks,
X5/UCP adapter checks, preflight, CLI I/O and fixture help. The explicit fixture
also runs the actual executable entry with synthetic outputs; it is not a SDK backend. The preflight library itself needs no OpenCV
or SDK. The adapter tests run with production code instrumented by ASan/UBSan. They
cover preflight rejection before SDK calls, metadata/precision rejection before
allocation, partial allocation and failed initialization cleanup, task/cache
errors, semantic output order and independence across inference calls. These
use narrow API doubles, not vendor SDK headers/libraries.

<a id="results-interpretation"></a>
## Verification boundaries

[Native preflight evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-native-preflight-review.md) covers registry parity, model/vocabulary rejection and shared streaming hashes.

[SDK adapter evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-sdk-runner-review.md) records resource/metadata tests and their host-only limits.

[Stage/NV12 evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-stages-review.md) covers actual byte comparisons, ownership/error paths and compiled API examples; it does not certify a board backend.

[Implementation evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md) records the earlier E26 checks. The [E11 extension evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md) records all three native tests, E26 regression and real E11 s/m/l ONNX comparisons. The E26n candidate comparison covers both single- and multi-label decoding against Python. [Geometry/mask evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md) separately records actual OpenCV compilation and full ROI pixel comparison. Labels/order are compared exactly; boxes, scores and coefficients use stated numerical tolerances. Masks are not compared by this candidate-only test.

The canonical selection/CLI/output implementation is present. [Native entry evidence](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-native-cli-review.md) separates real OpenCV host execution, Python process-policy tests and SDK API doubles. An independent review has accepted this host runtime composition and entrypoint scope ([review record](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-independent-review.md)). Board inference, real SDK compilation, OE-produced floating S artifacts and native dataset accuracy remain not-run; that review does not close conversion/evaluator acceptance, full repository integration or the complete B9 migration.
