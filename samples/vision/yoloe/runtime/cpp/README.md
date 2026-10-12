English | [简体中文](README_cn.md)

# YOLOE C++ runtime


Run E11/E26 prompt-free instance segmentation through `run.sh`, or embed the C++ three-stage library. The launcher selects an exact model, checks the local board and file identity, optionally builds the native executable, and retains logs and image/mask results. Real SDK compilation and board inference follow the build/run sections below. See the [Python runtime](../python/README.md) for the Python entry and its different X5 mask protocol.

<a id="overview"></a>
## C++ inference

Run prompt-free YOLOE instance segmentation with the matching X5 or S model. The native pipeline decodes candidates, restores masks and writes predictions and visualization files.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/  # Model and CLI headers: detect.hpp, cli.hpp
├── src/  # Five-file runtime: detect.cpp, cli.cpp, main.cpp
├── CMakeLists.txt  # Build configuration
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards

| Target | Variants / default | Model requirement | Support |
| --- | --- | --- | --- |
| X5 | 11s/m/l; default 11s | Locally converted floating-output model meeting the NHWC output contract, with explicit SHA-256 | supported |
| S100 | 11s, 26n/s/m/l/x; default 11s | Locally converted floating-output HBM with explicit SHA-256 | supported |
| S100P | 26n/s/m/l/x; default 26n | Locally converted floating-output HBM with explicit SHA-256 | supported |
| S600 | None | No published route; explicitly rejected | not-supported |

Published artifacts run through the [Python runtime](../python/README.md) on their matching targets: the published X5 11s/m/l BINs fail this entry's output binding (`Task output requires NHWC aligned geometry.`), and the published S HBMs have quantized outputs. This entry runs locally converted models whose ten outputs are unquantized FLOAT32 NHWC — prepare them with the [conversion workflow](../../conversion/README.md), record their SHA-256. The 14 publication identities remain the model-selection keys; for a locally converted model the identity names its architecture reference.

## Select, build and run

Commands below start at the repository root. Python 3.10+ and NumPy are required for the launcher; Python OpenCV is not needed by this native entry. `PYTHON` selects the interpreter used by `run.sh`. It also works from another working directory. Relative user paths and the default output directory are resolved from the caller's directory; child processes run from the repository root.

These host-safe commands inspect the active manifests and prepare a command without downloading, building, checking a board or executing inference:

```bash
bash samples/vision/yoloe/runtime/cpp/run.sh --list-models
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s --dry-run
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n --dry-run
```

A dry-run requires an explicit target. Its `executed`, `downloaded` and `runtime_metadata_verified` fields stay false. S publications report that a local floating-output conversion is required before execution.

On an X5 with its matching SDK and C++ dependencies, the published 11s/m/l
BINs are rejected by this entry's output binding, so running them stays with
the Python runtime (`python3 samples/vision/yoloe/runtime/python/main.py
--target x5 --variant 11s`). The native entry binds only models whose ten
outputs are unquantized FLOAT32 NHWC, so on X5 — as on S — first run the
conversion workflow and take the expected digest from its recorded evidence.
Replace both placeholders below with the actual artifact path and its 64
hexadecimal SHA-256 digits, using the matching target and variant:

```sh
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n \
  --model-path /path/to/converted-float.hbm \
  --local-float-sha256 REPLACE_WITH_EXPECTED_64_HEX_SHA256 \
  --build --output outputs/yoloe_cpp_s100p_26n_run1
```

Provide the converted model and its SHA-256 with the matching target and variant. Its ten output tensors must satisfy the FLOAT32 NHWC contract. Prepare S floating-output HBMs using the [conversion workflow](../../conversion/README.md). Use `--build` to build the native executable on the matching board.

`--build` configures Release mode and creates `runtime/cpp/build/<target>/yoloe_demo`; without it, the existing executable at that path is used. Use `--binary /absolute/path/yoloe_demo` for a separately built executable; it cannot be combined with `--build`. To configure SDK discovery explicitly on a matching SDK installation:

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-cli \
  -DYOLOE_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release \
  -DYOLOE_DNN_INCLUDE_DIR=/path/to/sdk/include \
  -DYOLOE_DNN_LIBRARY=/path/to/sdk/lib/libdnn.so
cmake --build /tmp/yoloe-board-cli --parallel 2
```

For UCP, also provide `YOLOE_UCP_LIBRARY` when discovery cannot find the matching `libhbucp`. Use `--binary /tmp/yoloe-board-cli/yoloe_demo` in the selected run command. The normal launcher verifies identity and inputs before building. Missing vendor dependencies fail explicitly; install the matching vendor SDK headers and libraries.

## Launcher parameters

| Option | Default / effect |
| --- | --- |
| `--target` | `auto`; detected local board for execution, explicit target required for dry-run |
| `--variant`, `--asset-id` | Target defaults above; an asset ID is the exact active manifest reference, conflicting choices fail |
| `--model-path` | Selected model's default path; override for an existing file |
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

Native `report.json` uses schema `rdk-model-zoo/yoloe-native-run/v1`, with target/variant, model/image/vocabulary hashes, image shape, effective configuration and aligned instances. Each instance contains a zero-based class ID, fixed label, score, original-image `[x1,y1,x2,y2]` box and ROI mask path/shape. Stored PNG masks use 0/255; in-memory masks use 0/1. Zero-area instances keep their exact empty axes and `mask: null`, without an empty PNG. `annotated.png` is the presentation overlay; use the stored ROI masks for evaluation. **All native masks use ROI layout, including X5; Python X5 uses full-image probability masks.** Do not interchange their evaluation inputs without adapting the protocol.

After a successful run, read `report.json` and the result files. Build, inference or output failures leave available logs and failure records in the output directory; errors before directory creation print to the terminal. Retry with a new output path. Native validation errors return 2; the launcher preserves positive native exit codes and maps signal termination to 2.

## Modules and model contract

| Module | Responsibility |
| --- | --- |
| `launcher.py`, `run.sh` | Publication selection, explicit build, process logs and run records |
| `src/main.cpp` | CLI orchestration: preflight gate → inputs → `YOLOE` built from model/config/gate → `predict` → saved results |
| `src/cli.cpp`, `inc/cli.hpp` | Strict option parsing, image/label byte-exact loading, fresh-output policy and report/mask writing |
| `src/detect.cpp`, `inc/detect.hpp` | The whole YOLOE model in one unit: geometry, NV12 preparation, `bind_heads` role binding, E11/E26 decodes, ROI mask restoration, board/model/vocabulary preflight, the `preprocess`/`infer`/`postprocess`/`predict` stages and the board DNN adapter (compiled with `YOLOE_HAS_SDK`) |

Float reads reuse Ultralytics' shared vocabulary header `inc/yolo.hpp`: `nhwc_float_plan` requires unquantized FLOAT32, batch-one NHWC, valid physical row/cell strides and a sufficient allocation, and `copy_float_output` removes physical padding and rejects nonfinite values. `bind_heads` binds the ten logical output roles by unique shape; dtype, stride and allocation requirements are enforced by these helpers, and model-family and vocabulary identity by the explicit selection rules below.

| Role | Shape at stride 8 / 16 / 32 |
| --- | --- |
| Class logits | `[1,80,80,4585]` / `[1,40,40,4585]` / `[1,20,20,4585]` |
| E11 DFL boxes | Same spatial shapes, 64 channels |
| E26 direct LTRB boxes | Same spatial shapes, 4 channels |
| Mask coefficients | Same spatial shapes, 32 channels |
| Prototype | One `[1,160,160,32]` tensor |

The caller explicitly selects E11 (64 box channels) or E26 (4); an incompatible family, missing/duplicate role, wrong vocabulary width or extra output is rejected. SDK ownership reuses the Ultralytics shared DNN backend (`inc/backend.hpp` + `src/backend.cpp`): `PackedModelOwner`, `Nv12Input`, `TaskOutputs` and the stack-portable synchronous inference; semantic YOLOE roles are validated before allocation. Quantized outputs are rejected, not manually dequantized. The existing [conversion guide](../../conversion/README.md) describes preparing float output models; prepare the matching float-output S HBM before native execution.

<a id="dependencies"></a>
## Dependencies

Install a C++17 compiler, CMake 3.20+ and OpenCV C++ core/imgproc/imgcodecs development libraries. Set `OpenCV_DIR` to the installed OpenCV CMake package if discovery fails. Board inference also requires the matching DNN SDK and, for UCP headers, `libhbucp`.

<a id="build"></a>
## Build the application library

From the repository root, build the core static library:

```bash
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-stage-core
cmake --build /tmp/yoloe-stage-core --parallel 4
```

The output is `libyoloe_core.a`. Add this directory with CMake `add_subdirectory` and link `yoloe_core`; include paths and OpenCV dependencies propagate to the application. For board inference, use the `YOLOE_BUILD_SDK=ON` library build below or the `YOLOE_BUILD_CLI=ON` executable build above.

<a id="run"></a>
## Run

Prepare a floating-output model for the selected board and use the `run.sh` commands under “Select, build and run” above. X5, S100 and S100P require a locally converted model with ten unquantized FLOAT32 NHWC outputs and its SHA-256. Read the reports, image and ROI masks from the output directory; fields are described in the results section.

<a id="parameters"></a>
## Candidate decoding

`decode_e11` accepts the same semantic arrangement with 64 box channels. It reuses Ultralytics' numerically stabilized 16-bin DFL expectation and sigmoid. Defaults are score 0.25 and classwise NMS IoU 0.7; score must be in `(0,1)`, NMS in `[0,1]`. It keeps one class per anchor and applies no E26 candidate cap. Coefficients travel with each candidate through NMS.

E11 boundaries: score **greater than or equal to** the threshold is accepted; a same-class box is suppressed only when IoU is **greater than** the NMS threshold (the Python NMS also suppresses IoU equality, so the two languages can differ exactly at that boundary). Native output is ordered by ascending class, then descending score; exact score ties prefer the lower scale/anchor index. Finite tensor values and exact lengths are validated before decoding, including the prototype not yet consumed by candidate math.

`decode_e26` accepts ten finite compact float vectors in semantic order: class/box/coefficients for each stride, then prototype. Every vector length is checked before reading. Defaults are score threshold 0.25, maximum 300 candidates and one class per anchor. The valid threshold range is `(0,1)` and the candidate cap is `1..8400`.

Selection applies a static Top-K: candidates are ranked by score, exact ties prefer lower scale, anchor, then class, and multi-label expansion breaks ties by the selected anchor rank and class. The raw threshold comparison is strict. Boxes are decoded in the 640×640 model canvas, scores use sigmoid, and mask coefficients remain aligned. No IoU NMS, geometry restoration, mask generation or dequantization occurs in this module. Nonfinite decoded boxes are rejected, including finite input distances that overflow during scaling. Multi-label expansion currently uses memory proportional to the selected anchor count times 4585 classes; large caps are materially more expensive than the default.

`prepare_bgr` returns owned 640×640 BGR pixels and explicit geometry. E11 letterbox truncates resized dimensions and pads with 127; its optional stretch uses nearest-neighbor. E26 letterbox uses ties-to-even rounding and padding 114, rejecting stretch. Both letterbox paths use linear interpolation and clamp tiny resized dimensions to at least one pixel. Inverse boxes use actual horizontal/vertical scales, using the actual resized dimensions.

`restore_e26_masks` combines prototype logits and coefficients, checks finite sums, linearly resizes logits to 640×640, thresholds at zero and crops in model coordinates, removes the recorded padding, uses nearest-neighbor to restore source dimensions, then copies the clipped integer-truncated ROI. It returns owning float boxes and `CV_8UC1` masks with values 0/1, preserving empty/degenerate instance slots. Reversed/nonfinite boxes, malformed geometry, wrong prototype length, nonfinite coefficients/prototypes or arithmetic overflow fail. It has no sigmoid, NMS, morphology or dequantization.

`restore_e11_masks` implements the S11 ROI protocol: clip the model box to actual image content, truncate its bounds at prototype scale, combine raw prototype values and coefficients, threshold strictly above 0.5, resize the binary crop with Lanczos4, and optionally apply a 5×5 rectangular opening (`do_morph=false` by default for the library). Final positive values are normalized to 1 because Lanczos can overshoot uint8 binary data to 2, so the C++ ROI mask is a strict 0/1 `CV_8UC1` image. Empty boxes retain exact zero-sized axes. The Python S11 runtime returns the Lanczos output unchanged as uint8 (isolated pixels can read 2; nonzero is foreground). This is distinct from X5 Python's full-image probability-mask path. `do_morph` toggles the 5×5 opening on the same candidate inputs.

<a id="interface-lifecycle"></a>
## Interface and lifetime

`YOLOE` exclusively owns a `std::unique_ptr<Runner>`. Construction validates configuration and backend protocol; invalid construction releases the supplied backend. The native constructor `YOLOE(SdkModel, Config, SdkPreflight)` builds and owns the board adapter itself; the runner-injecting constructor serves custom backends and tests. The backend must return ten independently owned compact semantic FLOAT32 vectors. It must perform hardware/artifact identity and SDK metadata validation before exposing inference; the base interface carries none of that validation itself. `SdkRunner` implements the low-level SDK boundary described below.

`preprocess` returns an owned compact Y plane (409600 bytes) and interleaved UV plane (204800 bytes), plus actual geometry. `infer` invokes the runner exactly once and returns owned raw outputs carrying that geometry. `postprocess` returns aligned `Instance` values with box/score/label/ROI mask. Prepared/raw batches from another task are rejected, including another task with the same protocol; do not cache a last-image geometry yourself. Raw outputs remain valid across subsequent calls, and result masks do not borrow SDK buffers. Use one task per inference thread; concurrent backend use is not guaranteed.

The following function shows the embedding API; the application supplies the matching backend:

```cpp
#include "detect.hpp"
yoloe::Result process_image(yoloe::Config config,
                            std::unique_ptr<yoloe::Runner> backend,
                            const cv::Mat& image) {
    yoloe::YOLOE task(config, std::move(backend));
    auto prepared = task.preprocess(image);
    auto raw = task.infer(prepared);
    return task.postprocess(raw);
    // task.predict(image) composes the same three operations.
}
```

Configuration defaults to E11, score 0.25, NMS unset (E11 resolves 0.7), morphology off, letterbox, max_det 300 and single_label true. E11 rejects E26-only overrides; E26 rejects an explicitly supplied NMS threshold, morphology or stretch. Select `config.protocol = yoloe::Protocol::E26` only with a matching backend. These are the library defaults; the demo CLI applies its own defaults where options are omitted.

The decode/geometry functions expose pure interfaces and own no SDK resources. `bind_heads` returns output indices; it does not retain references. `decode_e11` and `decode_e26` borrow their ten input vectors only for the duration of the call and return owning detections with copied boxes/scores/coefficients. Callers may release input tensors after return. Internal pointer views never escape. OpenCV result matrices use reference-counted owned storage and do not alias the caller's image/prototype; retain a returned result for as long as its pixels are needed. Invalid arguments throw `std::invalid_argument`; allocation failures may propagate. There is no model loading, implicit hardware selection or hidden cross-call geometry state.

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
runs before any SDK call; throwing stops construction.  The built-in
`make_preflight(expected_model_sha256, label_path)` supplies local board and
byte verification. The launcher selects the publication or
custom float file before invoking the executable with its verified digest.

Call `make_preflight(expected_model_sha256, label_path)` with the prepared model’s 64-digit SHA-256 and fixed ordered PF vocabulary. It checks local board identity, model existence and checksum, and vocabulary. `SdkRunner` checks the target, variant, SDK stack and tensor contract. For S targets, first prepare a compatible floating-output HBM using the conversion guide and save its conversion records.

The second complete API example shows the preflight-gated native path; the
task constructs and owns its board adapter:

```cpp
#include "detect.hpp"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                const std::string& expected_model_sha256,
                                const std::string& label_path) {
    yoloe::Config config;
    config.protocol = model.variant.rfind("26", 0) == 0 ? yoloe::Protocol::E26
                                                        : yoloe::Protocol::E11;
    yoloe::YOLOE task(model, config,
                      yoloe::make_preflight(expected_model_sha256, label_path));
    return task.predict(image);
}
```

To build the SDK library on a matching board SDK installation:

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
Use vendor SDK include paths for deployment. OpenCV development
files are still required. The default host build leaves `YOLOE_BUILD_SDK=OFF`.

<a id="results-interpretation"></a>
## Deployment and result usage

Build the native entry against the selected board SDK and OpenCV, prepare a float-output model with the [conversion guide](../../conversion/README.md), then run the documented launcher with the matching profile and the fixed 4585-entry class vocabulary.

Each run records its run outputs: `result/report.json` (schema `rdk-model-zoo/yoloe-native-run/v1`) lists the aligned instances — zero-based class IDs, fixed labels, scores, original-image `[x1,y1,x2,y2]` boxes and ROI mask paths — while `launch-report.json` retains the selected publication, expected/observed digests, exact argv/cwd and exit codes. `annotated.png` is a presentation overlay; use the stored masks and report fields for downstream processing.

All native masks use ROI layout on every target, including X5; Python X5 uses full-image probability masks, so adapt the protocol before mixing evaluation inputs. Stored PNG masks use 0/255 and in-memory masks 0/1. Dataset-level accuracy comes from the [evaluator](../../evaluator/README.md) workflows.
