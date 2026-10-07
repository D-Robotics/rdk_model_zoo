# Paraformer native inference

[简体中文](README_cn.md) · [Sample overview](../../README.md)

This directory provides the native S100 executable and launcher, CPU CIF/text
kernels, three-model application composition, UCP SDK adapter, production preflight
and prepared-manifest/NPY reader. Build and run it on S100 with the matching
board SDK as described below. The Python frontend can prepare feature files for
the native pipeline; C++ runs the three published stages, CPU CIF and source
text decoding.

<a id="supported-boards"></a>
## Supported boards

| Target | Status | Reason |
| --- | --- | --- |
| S100 | supported | Three published HBM models; native pipeline and launcher |
| X5 / S100P / S600 | not-supported | No matching published Paraformer package |

<a id="dependencies"></a>
<a id="environment"></a>
## Environment

The numerical library requires CMake 3.18 or newer and a C++17 compiler. It does
not include vendor SDK, JSON, NumPy, Torch or audio-library headers. Host builds
run on macOS arm64 with Apple Clang. Real frontend generation
uses the separately documented [Python environment](../python/README.md#environment).

<a id="build"></a>
## Build and run host checks

From repository root, using a new build directory:

```bash
cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-core -DCMAKE_BUILD_TYPE=Release -DPARAFORMER_BUILD_TESTS=ON -DPARAFORMER_SANITIZERS=ON
cmake --build /tmp/rdk-paraformer-core -j 2
ctest --test-dir /tmp/rdk-paraformer-core --output-on-failure
```

Success means four CTest checks pass: numerical contract, synthetic three-model
composition, SDK control flow with an isolated API double, and group preflight. Address/undefined-behavior sanitizers are enabled for Clang/GNU in
this command. Release tests retain assertions. The production artifact is the
static `paraformer_contract` library; test executables are not an inference CLI.

<a id="run"></a>
<a id="quickstart"></a>
## Run the native sample

Use repository root as cwd. The launcher needs Python with NumPy and PyYAML; it
never loads a Python inference SDK. Actual C++ execution requires S100 with the
matching UCP SDK. Feature generation additionally requires the separate verified
[FunASR environment](../python/README.md#environment). There is no automatic
model download, dependency installation, board fallback or test-backend fallback.

Preview the declared model set and native command on any host:

```bash
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --list-models
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --target s100 --dry-run
```

With the FunASR interpreter active, prepare the two bundled WAVs. This CPU step
works without a board or model files. The output directory must be new:

```bash
python samples/speech/paraformer/runtime/python/main.py --target s100 --preprocess-only --output-dir outputs/paraformer_features
```

On S100, explicitly prepare the published package and run the real native backend.
The following commands require the model download/board environment:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --target s100 --build --manifest outputs/paraformer_features/prepared-manifest.json
```

`--build` configures Release with real SDK, I/O and CLI enabled, tests disabled,
then builds `runtime/cpp/build/s100/paraformer_demo`. Provide SDK/JSON dependencies
as described below; CMake errors are retained. On later runs omit `--build` to use
that binary, or provide `--binary /absolute/path/to/paraformer_demo`. Choose a new
output directory per run. Prefix selection uses `--max-utts 1`; zero means all.

<a id="parameters"></a>
### Launcher options

| Option | Default / behavior |
| --- | --- |
| `--target` | `auto` reads actual local identity; choices auto/x5/s100/s100p/s600, only S100 has published assets |
| `--list-models` | Host-only list; auto lists the declared S100 set, not detected support |
| `--dry-run` | Preview only; requires explicit target; mutually exclusive with list |
| `--manifest` | `outputs/paraformer_features/prepared-manifest.json` |
| `--vocab-file` | `samples/speech/paraformer/model/s100/tokens.json` |
| `--max-utts` | 0 = all; positive prefix count, negative values rejected |
| `--output-dir` | `outputs/paraformer_cpp`; must not already exist |
| `--build` | Explicit configure/build; mutually exclusive with binary override |
| `--binary` | Otherwise the sample's `build/s100/paraformer_demo` |
| `--encoder-model-path`, `--predictor-model-path`, `--decoder-model-path` | Published model paths by default; overrides require all three paths and all three matching asset IDs |
| `--encoder-asset-id`, `--predictor-asset-id`, `--decoder-asset-id` | Exact IDs in the preflight table; supply all three or none |
| `--help` | Show parser help; no SDK/model loading |

Relative launcher arguments resolve from invocation cwd; child commands run at
repository root with absolute paths. `PYTHON` selects the launcher's interpreter.
Preprocessing is explicit: this launcher consumes prepared features and never
silently recomputes them or modifies the manifest.

The direct executable accepts `--help`; otherwise `--target s100`, `--manifest`,
`--vocab-file`, `--output-dir`, and each stage's `--<stage>-model-path`,
`--<stage>-asset-id`, `--<stage>-sha256` are required. Only `--max-utts` defaults to
0. The launcher derives those hashes from actual selected files and forwards the
complete arguments; manual direct invocation must supply all of them. Duplicate,
unknown or missing options fail with rc=2. There is no direct auto/list/dry-run.

### Results, logs and failures

A successful launcher run returns rc=0 and creates:

- `launch-report.json`: selected publication IDs, publisher versus observed
  digests, binary digest, exact child argv/cwd/UTC times, status and result digest.
- `configure.*.log`, `build.*.log` when building, `binary-help.*.log` and
  `native.*.log`: complete stdout/stderr per launched process.
- `result/result.json`: target/backend, three model digests and full physical
  tensor metadata, manifest/vocabulary identity and per-utterance results.

Per-utterance records retain input annotations/reference text, feature digest,
valid/original frames, truncation, recognized text, token IDs/count, decoder
execution and stage timings.
Stage timing excludes frontend, file I/O and other work outside the runner/CIF
calls. Exact-empty CIF skips decoder with zero IDs/text and null decoder timing.

Both layers reject existing output directories. Preflight/parser failures before
a directory is created return rc=2 with stderr and no new result directory. Later
native errors write `result/failed.json` containing partial completed records,
current utterance and the error; no successful result is written. The launcher
records failure and retains logs, including failed builds. If report writing itself
fails, stderr says so. `inference_attempted` becomes true before a pipeline call;
`inference_executed` is false before calls, null on a failed first call (some model
work may have occurred), and true after any completed utterance. Partial output
must not be interpreted as whole-run success.

Before accepting success, the launcher rejects `host-fixture`, checks exit code
and result existence, validates model identity, physical shapes/types/roles and
byte strides against the Python binding, checks every selected input/result and
text/token/timing consistency, and rehashes inputs/models/vocabulary. These report-consistency checks bind the report to the run. A host CLI fixture is
never installed as the public binary or accepted as native success.

## Build options and current entry points

| Option | Default | Effect |
| --- | --- | --- |
| `PARAFORMER_BUILD_TESTS` | `OFF` | Build/register four host tests |
| `PARAFORMER_SANITIZERS` | `OFF` | Enable ASan/UBSan on Clang/GNU and propagate link flags |
| `PARAFORMER_BUILD_CLI` | `OFF` | Build `paraformer_demo`; requires both SDK and I/O options enabled |
| `PARAFORMER_BUILD_IO` | `OFF` | Build prepared-manifest/NPY library using nlohmann JSON; add its host test when tests are enabled |
| `PARAFORMER_BUILD_SDK` | `OFF` | Build `paraformer_sdk` using actual vendor headers/libraries |
| `CMAKE_BUILD_TYPE` | CMake default | Documentation checks use `Release` |

The complete CLI and launcher flags are described below. In an embedding CMake project, add this directory
with `add_subdirectory` and link `paraformer_contract`; its public include path
contains `contract.h` and `pipeline.h`. Do not compile this numerical library with
fast-math: float32 operation order is part of source parity. Clang/GNU builds
explicitly disable floating-point contraction for the library.

<a id="interface-lifecycle"></a>
<a id="stage-io"></a>
## Numerical and stage contracts

All vectors represent contiguous flattened batch-one tensors. Float arrays are
owned; inputs are not modified. Shapes and finite values are checked before use.

| Interface | Inputs | Result |
| --- | --- | --- |
| `cif` | weights 401, hidden 401×512, explicit valid frames 0–400 | acoustic 100×512, int32 count capped at 100 |
| `decode` | logits 100×8404, count 0–100, 8,404 unique nonempty ordered strings | text and selected IDs |
| encoder callback | features 400×560 | context 400×512 |
| predictor callback | context 400×512 | weights 401 and hidden 401×512 |
| decoder callback | context, acoustic, int32 count, zero bias 512 | logits 100×8404 |

CIF masks weights at/after valid_frames before accumulation. No-fire input returns
zero acoustic values/count; the first 100 emissions are retained. It preserves
source float64 cumulative sums rounded to float32 and one fire per time step.
It is not a general multi-fire integrator for weights above one. Native CIF has
no implicit unmasked calibration mode.

Greedy text decoding uses the valid token prefix and first ID on equal scores,
filters tokens enclosed by `<...>`, removes every `@@` and concatenates without
separators. Repeated tokens remain repeated; it is not CTC. Returned IDs retain
special tokens that text rendering filters out. All logit rows are required to be
finite, even if the valid count uses only a prefix.

`Pipeline(encoder, predictor, decoder, vocabulary).predict(features, valid_frames)`
is application composition, not a model forward hiding CPU CIF between SDK calls.
It requires valid_frames 1–400, executes encoder → predictor → CIF → decoder, and
bypasses decoder when CIF count is zero. Callables must return owned raw arrays
synchronously. DecoderInput references are only valid during the callback; do not
retain them. The caller must keep any captured SDK resources alive and coordinate
access if they are not thread-safe. The numerical library does not load models, select
boards, perform file I/O, set scheduling or compile a fake SDK fallback.

<a id="results-interpretation"></a>
<a id="results"></a>
## Results and timings

`Prediction` contains text, token IDs/count, `decoder_executed` and `Timings`.
Encoder/predictor/CIF milliseconds are numeric; decoder milliseconds are an
`std::optional<double>` and remain absent on bypass. A failed stage throws and
does not return a successful prediction. Wrong frontend geometry/frame count
fails before model callbacks; wrong encoder output fails before predictor.

Timings cover runner calls and CPU CIF, excluding frontend, file I/O, validation
outside those calls and text rendering. They are not end-to-end latency and host
callbacks do not measure BPU performance. Pipeline outputs contain no CER or
accuracy estimate. The native executable applies group identity checks and writes result/failure
files as documented below.

<a id="integration-example"></a>
## Runnable numerical API example

After the build above, still from repository root:

```bash
cat > /tmp/rdk-paraformer-core/example.cc <<'CPP'
#include "contract.h"
#include <iostream>
int main() {
    std::vector<float> weights(401, 0.f), hidden(401 * 512, 0.f);
    weights[0] = .75f; weights[1] = .75f; weights[2] = .5f;
    for (int h = 0; h < 512; ++h) {
        hidden[h] = 2.f; hidden[512+h] = 6.f; hidden[1024+h] = 10.f;
    }
    const auto result = paraformer::cif(weights, hidden, 3);
    std::cout << result.token_count << " " << result.acoustic[0] << " "
              << result.acoustic[512] << "\n";
}
CPP
c++ -std=c++17 -fsanitize=address,undefined -Isamples/speech/paraformer/runtime/cpp/inc /tmp/rdk-paraformer-core/example.cc /tmp/rdk-paraformer-core/libparaformer_contract.a -o /tmp/rdk-paraformer-core/example
/tmp/rdk-paraformer-core/example
```

Expected output is `2 3 8`. [test_pipeline.cc](tests/test_pipeline.cc) also gives
an executable callback-composition example with explicit synthetic model outputs.
The default two WAVs can already be converted to features through the Python
`--preprocess-only` command; the optional feature I/O library below reads that prepared manifest in C++.

<a id="troubleshooting"></a>
## Verification and limits

The CTest suite exercises fractional integration, padding, empty input, the
100-emission cap, invalid contracts, repeated/special/BPE tokens, equal-score
ties, callback order, zero-token bypass and malformed intermediate tensors.

A build failure for sanitizer runtime libraries requires matching compiler/linker
support; `PARAFORMER_SANITIZERS=OFF` builds without instrumentation but does not
reproduce the sanitizer check. The SDK adapter validates shape/type/name/strides
from runtime metadata at load. SDK compilation, board inference, OE and CER run
through their respective guides; the identity/artifact preflight factory is
documented in the [preflight](#preflight) section.

<a id="sdk-adapter"></a>
## S100 SDK adapter

The optional `paraformer_sdk` library requires real S-series UCP headers
`dnn/hb_dnn.h`, `hb_ucp.h`, `hb_ucp_sys.h` and the `dnn`/`hbucp` libraries. On a
matching SDK development environment, configure a separate directory with
`cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-sdk -DPARAFORMER_BUILD_SDK=ON`
and build with `cmake --build /tmp/rdk-paraformer-sdk -j 2`. For a nonstandard SDK
prefix, provide `CMAKE_PREFIX_PATH` or the explicit CMake cache paths
`PARAFORMER_DNN_INCLUDE`, `PARAFORMER_UCP_INCLUDE`, `PARAFORMER_UCP_SYS_INCLUDE`,
`PARAFORMER_DNN_LIBRARY`, `PARAFORMER_UCP_LIBRARY`. No SDK installation is automatic.
Missing real SDK dependencies fail configuration; host API doubles are included
only by `test_sdk`, never a fallback for this library.

Construct `SdkRunner(SdkModel{path, "s100", Stage::Encoder}, preflight)` separately
for each stage. The callback is mandatory and runs before any SDK call; it must
reject wrong local board identity and a mismatched stage/publication/model digest.
A no-op callback is suitable only for the isolated host test. Use the concrete `make_preflight` factory below. The launcher below wires this into the complete native CLI. Each artifact must contain
exactly one named model. The runner binds physical names independently of order:

| Stage | Input roles → physical names | Output roles → physical names |
| --- | --- | --- |
| encoder | features → `speech` | context → `/encoder/after_norm/Add_1_output_0` |
| predictor | context → `/encoder/after_norm/Add_1_output_0` | alphas → `/predictor/Add_output_0`; hidden → `/predictor/Concat_5_output_0` |
| decoder | context → encoder context name; count → `token_num`; bias → `bias_embed`; acoustic → `onnx::Shape_8609` or `shape_8609` | logits → `logits`; optional count → `token_num` |

Shapes are those in the stage table above. The count uses int32; all other tensors
use float32. Physical quantization, unknown/duplicate roles, both acoustic aliases
at once, extra tensors, wrong dimensions, overlapping/misaligned byte strides or
insufficient allocation are rejected. The optional decoder count output is
validated and returned when present. No manual dequantization is added.

`RawTensors` maps semantic roles to `variant<vector<float>, vector<int32_t>>`.
`infer` validates the entire input set before touching SDK buffers, clears padding,
copies by actual per-axis byte strides, cleans caches, performs one synchronous
model call, invalidates caches and returns owned compact arrays. Later calls do
not overwrite prior results. Float inputs must be finite; count must be 0–100.
Outputs are raw; pipeline/decoding validate finite values before numerical use.
Use one runner serially, or protect it externally against concurrent calls.
Scheduling currently uses the shared synchronous transport's default priority
and any BPU core; no custom scheduling API is claimed.

This complete embedding function compiles against the public header. It requires
caller-provided model group/vocabulary and validated features; it constructs the
real preflight callback itself. It is not a
synthetic inference result or a self-contained board application:

```cpp
#include "preflight.h"
#include <algorithm>
#include <utility>
std::vector<float> encode_features(const paraformer::ModelGroup &models,
                                  const std::string &vocabulary,
                                  const std::vector<float> &features) {
    auto verify = paraformer::make_preflight(models, vocabulary);
    const auto encoder_model = std::find_if(models.begin(), models.end(),
        [](const auto &a) { return a.model.stage == paraformer::Stage::Encoder; });
    paraformer::SdkRunner encoder(encoder_model->model, std::move(verify));
    auto outputs = encoder.infer({{"features", features}});
    return std::move(std::get<std::vector<float>>(outputs.at("context")));
}
```

Packed models, tensor allocations and inference tasks reuse the shared
owners/transport. The multi-input transport permits decoder's four inputs;
the image transport retains its one/two-input restriction.

<a id="preflight"></a>
## Three-model preflight

Link `paraformer_preflight` for the SDK-independent checks, or `paraformer_sdk`
which links it transitively. `ModelGroup` is an array of three `ModelArtifact`
records. Each record contains `SdkModel{path, "s100", stage}`, `asset_id`, and a
64-digit expected SHA-256. Order is arbitrary; exactly one encoder, predictor and
decoder is required. `expected_asset_id(stage)` returns the fixed publication ID:

| Stage | Asset ID |
| --- | --- |
| Encoder | `s:paraformer:s100/paraformer_large_encoder_400x560_s100.hbm` |
| Predictor | `s:paraformer:s100/paraformer_large_predictor_400x512_s100.hbm` |
| Decoder | `s:paraformer:s100/paraformer_large_decoder_400x512_s100.hbm` |

Pass the vocabulary path separately. Its required SHA-256 is
`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`.
Use observed model digests captured during package preparation for reproducible
file identity. The publisher does not record model digests, so a local expected
digest binds the local bytes; shape/type/name validation still runs at load.
Replace the expected digest only after re-preparing the package, never to
silence a mismatch.

`make_preflight(group, vocabulary)` reads actual local identity through the shared
platform reader and immediately verifies the entire group before any runner is
constructed. S100P board aliases override a generic S100 SoC identity. Unknown
identity, other targets, wrong stage/asset IDs, duplicate stages, missing/empty or
non-regular files, symbolic/hard-link aliases between stages, changed model bytes
and a different vocabulary fail. Hex model digests are case-insensitive.

The returned callback validates the runner's stage/target/path and rechecks local
identity plus all three model digests and vocabulary before each model load.
Thus a bad decoder file is caught before encoder is loaded. Rechecking incurs
three-file hashing at factory creation and at each runner construction, not at
every inference. Keep artifacts unchanged throughout loading and execution;
preflight does not lock files against concurrent replacement.

`verify_group(group, vocabulary, actual)` is the explicit-identity lower-level
checker used in host tests; customer deployment code should use `make_preflight`
to obtain actual identity rather than supplying a fabricated identity. No bypass
switch or implicit S100P fallback is provided.

<a id="prepared-features"></a>
## Read Python-prepared features

`paraformer_feature_io` reads the actual `prepared-manifest.json` and `.npy`
files produced by the [Python `--preprocess-only` flow](../python/README.md).
It does not recompute or approximate FunASR features, resample audio, load models
or modify the original manifest. Install/provide the nlohmann JSON header library
first (3.11.3 verified); NumPy is needed for feature generation, not by this C++
reader. No dependency is installed by CMake. For a nonstandard header location,
set `CMAKE_PREFIX_PATH` or `-DPARAFORMER_JSON_INCLUDE=/path/to/include`.

Build from repository root in a separate directory:

```bash
cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-io -DCMAKE_BUILD_TYPE=Release -DPARAFORMER_BUILD_IO=ON -DPARAFORMER_BUILD_TESTS=ON -DPARAFORMER_SANITIZERS=ON
cmake --build /tmp/rdk-paraformer-io -j 2
ctest --test-dir /tmp/rdk-paraformer-io --output-on-failure
```

Six tests should pass, including reading the two persisted real frontend feature
files. `feature_probe` is a host verification tool, not an inference executable.
A complete feature reader embedding function is:

```cpp
#include "feature_io.h"
std::vector<float> first_features(const std::string &manifest) {
    const auto items = paraformer::load_prepared_manifest(manifest, 1);
    return paraformer::load_features(items.front());
}
```

Link `paraformer_feature_io` in your CMake application. For full inference pass the
selected item's `valid_frames` alongside its loaded values to `Pipeline::predict`;
never substitute 400 for a short utterance. The example returns only values to
illustrate reading and does not execute a model.

`load_prepared_manifest(path, max_utts=0)` returns `FeatureItem` records. Zero means
all records; a positive value selects the first N. The entire manifest is checked
structurally before prefix selection. Only selected feature files need exist when
loaded. `feature_file` resolves relative to the manifest directory (absolute paths
are also accepted), independently of later cwd changes. Records require:

| Field | Contract |
| --- | --- |
| `utt_id` | Unique nonempty filename stem; no slash, backslash, NUL, dot/dot-dot or surrounding ASCII whitespace |
| `feat_length` | Integer 1–400; no missing-length fallback |
| `original_frames` | Positive integer, at most the C++ signed-int limit |
| `truncated` | Boolean equal to `original_frames > 400` |
| `feature_file` | Nonempty path to a regular NPY file |
| `feature_sha256` | 64 hex digits; must match bytes read, case-insensitive |
| `text` | Optional reference string, not recognized output |

`feat_length` must equal `min(original_frames, 400)`. Unknown annotation fields are
preserved in `original_record_json`, a semantic JSON serialization (not the
original whitespace). `reference_text` is an optional string. FeatureItem also
provides the resolved path, normalized digest and explicit frame/truncation fields.
A user-supplied digest identifies the local bytes; record the frontend version
alongside the digest when comparing runs.

`load_features(item)` hashes the same owned bytes it parses and returns a compact
float vector of 224,000 elements. Supported inputs are NPY versions 1.0/2.0/3.0,
C-order shape `[1,400,560]`, explicit little/big-endian float32 (`<f4`/`>f4`); endian
conversion preserves values. All values, including padded rows, must be finite.
The header is parsed as data with arbitrary key order and either quote style;
Python expressions are never evaluated. Header length is capped at 64 KiB.
Fortran order, other types/shapes/versions, duplicate/unknown header keys, trailing
syntax or data, truncated payloads, malformed metadata and changed bytes fail
with exceptions. The reading API returns no partial success for a failed file.
